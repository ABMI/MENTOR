from typing import TypedDict, Dict, Any, List, Literal
from langgraph.graph import START, END, StateGraph
from agents.mentor.curator import build_curator
from agents.mentor.suggestor import build_suggestor, count_results
from agents.mentor.diagnostician import build_diagnostician
from core.model_factory import create_llm
from core.config.experiment import LLM_ROLES
from core.prompt import get_prompt, prompt_hash
from core.utils import log, span, span_start, span_end
from pathlib import Path
import subprocess


class AgentState(TypedDict, total= False):
    outcomes: Dict[int, Any]
    ingredient_concept_ids: List
    ingredient_concepts: Dict[int, Any]
    diagnostics: int
    verdict: str
    is_all_known_ingredient: bool
    cem_result: Dict[int, Any]
    cem_result_detail: Dict[int, Any]
    minimum_min_patients: int
    init_min_patients: int
    min_patients: int
    step: int
    
  
def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent,
                              capture_output=True, text=True, timeout=5)
    except Exception:
        return None


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def log_run_config(cfg, overrides=None):
    """One line with every resolved experiment value, so runs can be compared by grep."""
    head = _git("rev-parse", "--short", "HEAD")
    diff = _git("diff", "--quiet", "HEAD", "--", ":/src", ":/main.py")
    kv = {
        "commit": head.stdout.strip() if head and head.returncode == 0 else None,
        "dirty": diff.returncode == 1 if diff else None,
        "overrides": sorted(_flatten(overrides or {})),
    }
    for section, values in cfg.items():
        values = dict(values)
        if section in LLM_ROLES:
            llm = values.pop("llm")
            values.update(
                model_name=llm["model_name"],
                thinking=bool(llm.get("enable_thinking", False)),
                temperature=llm["temperature"],
                prompt_hash=prompt_hash(get_prompt(section, values["prompt"])),
            )
        kv.update(_flatten(values, f"{section}."))
    log("ok", "run.config", **kv)


def build_nodes(db, cfg):
    curator_llm = create_llm(cfg["curator"]["llm"])
    curator = build_curator(curator_llm, db, cfg)

    suggestor_llm = create_llm(cfg["suggestor"]["llm"])
    adjudicator_llm = create_llm(cfg["adjudicator"]["llm"])
    suggestor = build_suggestor(suggestor_llm, adjudicator_llm, db, cfg)

    diagnostician_llm = create_llm(cfg["diagnostician"]["llm"])
    diagnostician = build_diagnostician(diagnostician_llm, db, cfg)

    def curator_node(state):
        with span("curator", llm=True, ingredients=state["ingredient_concept_ids"]) as end:
            out = curator.invoke(state)
            end.update(valid=out.get("is_all_known_ingredient", False))
        return out

    # iter = suggestor + diagnostician; opened here, closed in diagnostician_node
    def suggestor_node(state):
        n = (state["init_min_patients"] - state["min_patients"]) // state["step"] + 1
        span_start("iter", llm=True, n=n, min_patients=state["min_patients"])
        n_before = len(state.get("outcomes") or {})
        with span("suggestor", llm=True) as end:
            out = suggestor.invoke(state)
            excluded, included, pending = count_results(out["outcomes"].values(), "final_result")
            no_cover = sum(1 for v in out["outcomes"].values() if v.get("cem_result") == 2)
            end.update(outcomes=len(out["outcomes"]), new=len(out["outcomes"]) - n_before,
                       targets=len(out["outcomes"]) - no_cover, no_cover=no_cover,
                       excluded=excluded, included=included, pending=pending - no_cover)
        return out

    def diagnostician_node(state):
        with span("diag", llm=True) as end:
            out = diagnostician.invoke(state)
            nco_count = count_results(out["outcomes"].values(), "final_result")[1]
            end.update(verdict=out["verdict"], nco_count=nco_count)
            if out["verdict"] == "insufficient":
                end.update(next_min_patients=out["min_patients"])
        span_end("iter", min_patients=state["min_patients"], nco_count=nco_count, verdict=out["verdict"])
        return out

    return curator_node, suggestor_node, diagnostician_node

    
def route_curator(state: AgentState) -> Literal["suggestor", "__end__"]:
    if state.get("is_all_known_ingredient"):
        return "suggestor"
    return END


def route_diagnostician(state: AgentState) -> Literal["suggestor", "__end__"]:
    if state.get("diagnostics"):
        return END
    return "suggestor"

    
def build_graph(curator, suggestor, diagnostician):
    builder = StateGraph(AgentState)
    builder.add_node("curator", curator)
    builder.add_node("suggestor", suggestor)
    builder.add_node("diagnostician", diagnostician)

    builder.add_edge(START, "curator")
    builder.add_conditional_edges(         
        "curator",
        route_curator,
        {"suggestor": "suggestor", END: END}, 
    )
    builder.add_edge("suggestor", "diagnostician")
    builder.add_conditional_edges(
        "diagnostician",
        route_diagnostician,
        {"suggestor": "suggestor", END: END},
    )
    return builder

    
def build_mentor(db, cfg):
    nodes = build_nodes(db, cfg)
    graph = build_graph(*nodes)
    return graph.compile()