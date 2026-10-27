"""ComfyUI-AccelLog: 每次采样运行时自动汇报实际使用的注意力加速后端。

纯启动补丁，不在工作流里注册任何节点：
  1. 给所有注意力后端调用包一层计数器（不改动原有逻辑，只计数）；
  2. 包裹 comfy.samplers.CFGGuider.sample，每次采样结束输出一行汇总。

覆盖的加速路径：
  - sageattention.sageattn            (XB sage 节点、T8 sage 的最终内核)
  - attention_sage / attention3_sage  (comfy 内置 sage 后端)
  - attention_comfy_kitchen_int8 / sol 及 comfy_kitchen.int8_attention / sol_attn (Kitchen)
  - flash / xformers / pytorch / sub_quad / split / basic (其余后端)
环境变量 ACCEL_LOG=0 可关闭。删除整个文件夹即可完全恢复。
"""

import functools
import logging
import os
import threading

_LOG = logging.getLogger("AccelLog")
_ENABLED = os.environ.get("ACCEL_LOG", "1").strip().lower() not in ("0", "false", "no", "off")

_COUNTS = {}                 # label -> 本次采样的调用次数
_LOCAL = threading.local()   # 防止同一逻辑调用的嵌套重复计数
_T8_KEY = "qwen_image21_t8"

_LABELS = {
    "attention_sage": "SageAttention",
    "attention3_sage": "SageAttention3",
    "attention_comfy_kitchen_int8": "Kitchen INT8",
    "attention_comfy_kitchen_sol": "Kitchen Sol",
    "attention_flash": "FlashAttention",
    "attention_xformers": "xFormers",
    "attention_pytorch": "PyTorch注意力",
    "attention_sub_quad": "SubQuadratic",
    "attention_split": "SplitCross",
    "attention_basic": "Basic",
    "sageattn": "SageAttention",
    "sageattn3_blackwell": "SageAttention3",
    "int8_attention": "Kitchen INT8",
    "int8_attention_from_prequantized": "Kitchen INT8",
    "sol_attn": "Kitchen Sol",
}

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _wrap_backend(label, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not getattr(_LOCAL, "busy", False):
            _LOCAL.busy = True
            try:
                _COUNTS[label] = _COUNTS.get(label, 0) + 1
            finally:
                _LOCAL.busy = False
        return fn(*args, **kwargs)

    wrapper._acclog_wrapped = True
    for attr in ("container_function",):
        try:
            if hasattr(fn, attr):
                setattr(wrapper, attr, getattr(fn, attr))
        except Exception:
            pass
    return wrapper


def _patch_attention_module():
    import comfy.ldm.modules.attention as attn

    names = [
        "attention_sage", "attention3_sage",
        "attention_comfy_kitchen_int8", "attention_comfy_kitchen_sol",
        "attention_flash", "attention_xformers",
        "attention_pytorch", "attention_sub_quad", "attention_split", "attention_basic",
        "sageattn", "sageattn3_blackwell",
    ]
    patched, skipped = [], []
    for name in names:
        fn = getattr(attn, name, None)
        if not callable(fn) or getattr(fn, "_acclog_wrapped", False):
            continue
        try:
            setattr(attn, name, _wrap_backend(_LABELS.get(name, name), fn))
            patched.append(_LABELS.get(name, name))
        except Exception as e:
            skipped.append("%s(%s)" % (name, e))
    try:
        for key, fn in list(attn.REGISTERED_ATTENTION_FUNCTIONS.items()):
            if not callable(fn) or getattr(fn, "_acclog_wrapped", False):
                continue
            fn_name = getattr(fn, "__name__", "") or ""
            attn.REGISTERED_ATTENTION_FUNCTIONS[key] = _wrap_backend(_LABELS.get(fn_name, key), fn)
            patched.append("registry:%s" % key)
    except Exception as e:
        skipped.append("registry(%s)" % e)
    return patched, skipped


def _patch_sageattention():
    try:
        import sageattention
        fn = getattr(sageattention, "sageattn", None)
        if callable(fn) and not getattr(fn, "_acclog_wrapped", False):
            sageattention.sageattn = _wrap_backend("SageAttention", fn)
            return ["sageattention.sageattn"]
    except Exception as e:
        return ["sageattention 跳过(%s)" % e]
    return []


def _patch_comfy_kitchen():
    out = []
    try:
        import comfy_kitchen
        for name in ("int8_attention", "int8_attention_from_prequantized", "sol_attn"):
            fn = getattr(comfy_kitchen, name, None)
            if callable(fn) and not getattr(fn, "_acclog_wrapped", False):
                setattr(comfy_kitchen, name, _wrap_backend(_LABELS.get(name, name), fn))
                out.append("comfy_kitchen.%s" % name)
    except Exception as e:
        out.append("comfy_kitchen 跳过(%s)" % e)
    return out


def _startup_default():
    try:
        import comfy.model_management as mm
        if mm.comfy_kitchen_attention_enabled():
            return "Comfy Kitchen INT8 (--use-ck-attention)"
        if mm.sage_attention_enabled():
            return "SageAttention (--use-sage-attention)"
        if mm.flash_attention_enabled():
            return "FlashAttention (--use-flash-attention)"
        if mm.xformers_enabled():
            return "xFormers"
        return "核心默认 (SDPA/SubQuad)"
    except Exception:
        return None


def _identify_override(override):
    try:
        if getattr(override, "_accel_switch_owned", False):
            return "顶栏切换(%s)" % getattr(override, "_accel_switch_label", "?")
        q = getattr(override, "__qualname__", "") or ""
        m = getattr(override, "__module__", "") or ""
        if q.startswith("make_override"):
            return "T8 节点(sage)"
        if "set_model_optimized_attention" in q:
            fn = None
            try:
                fn = override.__closure__[0].cell_contents
            except Exception:
                pass
            fn_name = (getattr(fn, "__name__", "") or "").lower()
            if "kitchen" in fn_name:
                return "ModelAttentionBackend 节点(Kitchen INT8)"
            if "sage" in fn_name:
                return "ModelAttentionBackend 节点(SageAttention)"
            if "pytorch" in fn_name:
                return "ModelAttentionBackend 节点(PyTorch)"
            return "ModelAttentionBackend 节点"
        low = ("%s.%s" % (m, q)).lower()
        if "sage" in low:
            return "XB sage 节点"
        try:
            cells = [c.cell_contents for c in (override.__closure__ or ())]
        except Exception:
            cells = []
        for c in cells:
            if callable(c) and "sage" in (getattr(c, "__name__", "") or "").lower():
                return "XB sage 节点"
        if "kitchen" in low:
            return "Kitchen 节点(%s)" % (m or q)
        return "自定义注意力 override(%s.%s)" % (m, q) if (m or q) else "自定义注意力 override"
    except Exception:
        return "自定义注意力 override"


def _describe_source(guider):
    sources = []
    try:
        opts = guider.model_options.get("transformer_options", {}) or {}
    except Exception:
        opts = {}
    if isinstance(opts, dict):
        override = opts.get("optimized_attention_override")
        if override is not None:
            sources.append(_identify_override(override))
        t8 = opts.get(_T8_KEY)
        if isinstance(t8, dict):
            sage = t8.get("sage")
            if sage == "sage_kitchen":
                sources.append("T8 节点(sage/kitchen 混合)")
            elif sage:
                sources.append("T8 节点(sage)")
            else:
                sources.append("T8 节点(仅块缓存)")
    if not sources:
        sources.append(_startup_default() or "未识别")
    return " + ".join(sources)


def _emit_report(guider):
    if not _ENABLED:
        return
    parts = ["%s×%d" % (label, n) for label, n in sorted(_COUNTS.items(), key=lambda kv: -kv[1]) if n > 0]
    head = " + ".join(parts) if parts else "未检测到已知加速（核心默认路径，未计数）"
    _LOG.info("[加速日志] %s | 来源: %s", head, _describe_source(guider))


def _install_sampler_hook():
    try:
        import comfy.samplers
        if getattr(comfy.samplers.CFGGuider.sample, "_acclog_wrapped", False):
            return True
        original = comfy.samplers.CFGGuider.sample

        @functools.wraps(original)
        def sample(self, *args, **kwargs):
            _COUNTS.clear()
            try:
                return original(self, *args, **kwargs)
            finally:
                try:
                    _emit_report(self)
                except Exception:
                    _LOG.exception("加速日志报告生成失败")

        sample._acclog_wrapped = True
        comfy.samplers.CFGGuider.sample = sample
        return True
    except Exception as e:
        _LOG.warning("CFGGuider.sample 钩子挂载失败（每次运行的加速日志将不可用）: %s", e)
        return False


def _install():
    if not _ENABLED:
        _LOG.info("ACCEL_LOG=0，加速日志已禁用")
        return
    patched, skipped = _patch_attention_module()
    patched += _patch_sageattention()
    patched += _patch_comfy_kitchen()
    hooked = _install_sampler_hook()
    _LOG.info("注意力加速日志已启用 | 采样钩子: %s | 启动默认: %s",
              "CFGGuider.sample" if hooked else "挂载失败",
              _startup_default() or "未识别")
    if skipped:
        _LOG.info("跳过项: %s", ", ".join(skipped))


_install()
