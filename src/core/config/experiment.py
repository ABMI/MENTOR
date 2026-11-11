"""Experiment parameters — the one place to edit when trying a new setup.

EXPERIMENT is the baseline. A run changes only what differs through overrides
(see run.ipynb), e.g. {"adjudicator": {"thinking": False}}.
Every resolved value is printed on the [run.config] log line.
"""
import copy


# ── model presets (vLLM, OpenAI-compatible): role configs refer to these by name
MODELS = {
    "qwen3.6-27b": {
        "model_name": "Qwen/Qwen3.6-27B-FP8",
        "base_url": "http://localhost:8000/v1",   # your vLLM server
        "api_key": "token-not-needed",
        "temperature": 0,
        "enable_thinking": False,
    },
}


# ── experiment ───────────────────────────────────────────────────────────────
# Role keys:
#   model        name in MODELS
#   thinking     None = use the preset's enable_thinking
#   temperature  None = use the preset's temperature
#   prompt       variant name in core/prompt/<role>.py PROMPTS
EXPERIMENT = {
    "search": {
        "min_patients": 70000,      # first iteration's patient-count threshold
        "min_floor": 70000,         # stop when the next threshold would go below this
        "step": 10000,              # threshold decrease per iteration
    },
    "curator": {
        "model": "qwen3.6-27b",
        "thinking": None,
        "temperature": None,
        "prompt": "v2",
        "mode": "summary",          # summary (LLM-summarized labels) / raw (labels as fetched) / name_only
    },
    "suggestor": {
        "model": "qwen3.6-27b",
        "thinking": None,
        "temperature": None,
        "prompt": "v2",
        "max_retries": 3,
        "use_hierarchy": True,      # parent concepts in the validate/review prompts
        "hierarchy_parent_levels": 3,
    },
    "adjudicator": {
        "model": "qwen3.6-27b",
        "thinking": True,
        "temperature": None,
        "prompt": "v2",
        "max_retries": 3,
    },
    "diagnostician": {
        "model": "qwen3.6-27b",
        "thinking": None,
        "temperature": None,
        "prompt": "v2",
        "max_retries": 3,
        "target_nco_count": 100,    # the LLM judges the effective (de-duplicated) NCO count against this
    },
    "openfda": {
        "max_labels": 1000,
        "min_interval": 0.3,        # seconds between requests
        "single_ingredient": True,  # use single-ingredient labels only (fallback: all); False = all labels
    },
    "cem": {
        "base_url": "https://atlas-demo.ohdsi.org/WebAPI",
        "source_key": "CEM",
        "poll_interval": 1,         # seconds
        "max_wait": 300,            # seconds of polling before a round is abandoned
        "log_every": 10,            # seconds between [cem.poll] wait lines
        "request_timeout": 180,
        "retry_interval": 1800,     # seconds to wait before a new round (new concept set) after a failed one
        "max_rounds": None,         # None = keep retrying until a round succeeds
    },
}

LLM_ROLES = ("curator", "suggestor", "adjudicator", "diagnostician")


def llm_config(role_cfg):
    """Preset + role-level thinking/temperature -> the dict create_llm() takes."""
    if role_cfg["model"] not in MODELS:
        raise KeyError(f"unknown model: {role_cfg['model']} (allowed: {', '.join(MODELS)})")
    cfg = dict(MODELS[role_cfg["model"]])
    if role_cfg.get("thinking") is not None:
        cfg["enable_thinking"] = role_cfg["thinking"]
    if role_cfg.get("temperature") is not None:
        cfg["temperature"] = role_cfg["temperature"]
    return cfg


def _merge(base, overrides, path=""):
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        key = f"{path}{k}"
        if k not in base:
            raise KeyError(f"unknown config key: {key} (allowed: {', '.join(base)})")
        if isinstance(base[k], dict) and isinstance(v, dict):
            out[k] = _merge(base[k], v, key + ".")
        else:
            out[k] = v
    return out


def build_config(overrides=None):
    """EXPERIMENT + overrides, with every role's llm resolved and prompt names checked."""
    from core.prompt import get_prompt

    cfg = _merge(EXPERIMENT, overrides or {})
    for role in LLM_ROLES:
        cfg[role]["llm"] = llm_config(cfg[role])
        get_prompt(role, cfg[role]["prompt"])  # fail fast on a bad variant name
    return cfg
