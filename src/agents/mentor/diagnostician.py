from typing import TypedDict, Dict, Any
from langgraph.graph import START, END, StateGraph
from core.prompt import get_prompt
from core.utils import parse_json_dict
from core.utils import log, secs
import time


class AgentState(TypedDict):
    ingredient_concepts: Dict[int, Any]
    outcomes: Dict[int, Any]
    diagnostics: int
    verdict: str
    min_patients: int
    minimum_min_patients: int
    step: int


def is_valid_result(result):
    return (
        isinstance(result, dict)
        and result.get("result") in (0, 1)
        and "reason" in result
    )


def build_nodes(llm, db, cfg):
    diagnose_prompt = get_prompt("diagnostician", cfg["diagnostician"]["prompt"])
    max_retries = cfg["diagnostician"]["max_retries"]
    target_nco_count = cfg["diagnostician"]["target_nco_count"]

    def diagnose(state: AgentState):
        outcomes = state["outcomes"]
        drug_infos = list(state["ingredient_concepts"].values())
        minimum_min_patients = state["minimum_min_patients"]
        min_patients = state["min_patients"]
        step = state["step"]

        t0 = time.perf_counter()
        nco_list = []
        for concept_id, info in outcomes.items():
            if info["final_result"] == 1:
                nco_list.append({"name": info["name"], "patient_count": info["patient_count"]})

        if len(nco_list) != 0:
            nco_text = "\n".join(f"- {item['name']}" for item in nco_list)
            variables = {
                "drug_infos": drug_infos,
                "target_nco_count": target_nco_count,
                "nco_count": len(nco_list),
                "nco_list": nco_text,
            }

            result = None
            for attempt in range(max_retries):
                try:
                    response = llm.invoke(diagnose_prompt, variables, attempt=f"{attempt+1}/{max_retries}")
                    result = parse_json_dict(response)
                    if is_valid_result(result):
                        break
                    log("warn", "diag.diagnose", level="WARN", attempt=f"{attempt+1}/{max_retries}",
                        error="missing result/reason keys")
                except Exception as e:
                    log("warn", "diag.diagnose", level="WARN", attempt=f"{attempt+1}/{max_retries}", error=e)
                    result = None

            if is_valid_result(result):
                diagnostics = result["result"]
                reason = result["reason"]
                llm_verdict = "sufficient" if diagnostics else "insufficient"
            else:
                log("error", "diag.diagnose", level="ERROR", reason="max retries exceeded")
                diagnostics = 0
                reason = "diagnosis_failed_after_retries"
                llm_verdict = "failed"
        else:
            diagnostics = 0
            reason = "nco_count is zero"
            llm_verdict = "skipped"

        log("ok", "diag.diagnose", nco_count=len(nco_list), target=target_nco_count, llm_verdict=llm_verdict,
            reason=reason, time=secs(time.perf_counter() - t0))

        next_min_patients = min_patients - step
        if diagnostics:
            verdict = "sufficient"
        elif next_min_patients < minimum_min_patients:
            diagnostics = 1
            verdict = "exhausted"
        else:
            verdict = "insufficient"

        return {
            "diagnostics": diagnostics,
            "verdict": verdict,
            "min_patients": next_min_patients,
        }

    return (diagnose,)


def build_graph(diagnose):
    builder = StateGraph(AgentState)

    builder.add_node("diagnose", diagnose)
    builder.add_edge(START, "diagnose")
    builder.add_edge("diagnose", END)
    return builder


def build_diagnostician(llm, db, cfg):
    nodes = build_nodes(llm, db, cfg)
    graph = build_graph(*nodes)
    return graph.compile()
