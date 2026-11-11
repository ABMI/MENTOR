from typing import TypedDict, Dict, Any, List
from langgraph.graph import START, END, StateGraph
from core.prompt import get_prompt
from core.utils import parse_json_dict
from core.cem import get_negative_controls
from core.concept_pool import load_person_count_with_descendants
from core.config.cdm import config as cdm_config
from core.concept_hierarchy import format_concept_neighbors
from core.utils import log, secs
import time


CONCEPT_TABLE = cdm_config["concept_table"]
PERSON_COUNT_WITH_DESCENDANTS_PATH = cdm_config["person_count_with_descendants_path"]


class AgentState(TypedDict):
    ingredient_concept_ids: List
    ingredient_concepts: Dict[int, Any]
    outcomes: Dict[int, Any]
    cem_result: Dict[int, Any]
    cem_result_detail: Dict[int, Any]
    init_min_patients: int
    min_patients: int
    step: int

    
def is_valid_result(result):
    return (
        isinstance(result, dict)
        and "result" in result
        and "reason" in result
    )

    
def count_results(items, key):
    values = [v.get(key) for v in items]
    return values.count(0), values.count(1), len(values) - values.count(0) - values.count(1)


CEM_SOURCES = [
    ("Publications", "Pmid"),
    ("Product labels", "Splicer"),
    ("Spontaneous reports", "Faers"),
]
CEM_LEVELS = ["exact", "descendant", "parent", "ancestor"]
CEM_FLAGS = [
    ("indication_or_contraindication", "indCi"),
    ("too_broad", "tooBroad"),
    ("drug_induced", "drugInduced"),
    ("pregnancy_related", "pregnancy"),
    ("not_prevalent", "notPrevalent"),
    ("optimized_out", "optimizedOut"),
]


def format_cem_verdict(cem_result):
    return {
        1: "negative control candidate (no evidence of association found)",
        0: "NOT a negative control candidate (evidence of association found)",
    }[int(cem_result)]


def format_cem_evidence(row):
    if not row:
        return "(no CEM record for this outcome)"
    lines = ["Evidence counts (exact / descendant / parent / ancestor):"]
    for label, key in CEM_SOURCES:
        lines.append(f"- {label}: " + " / ".join(str(row.get(f"{level}{key}Count", 0)) for level in CEM_LEVELS))
    lines.append("Flags: " + ", ".join(f"{name}={row.get(key, 0)}" for name, key in CEM_FLAGS))
    return "\n".join(lines)


def format_first_judgment(result):
    if not result or result.get("result") not in (0, 1):
        return "(unavailable: the first-pass judgment failed)"
    verdict = "valid negative control" if result["result"] == 1 else "NOT a valid negative control"
    return f"result = {result['result']} ({verdict})\nreason: {result.get('reason', '')}"


def build_nodes(llm, llm2, db, cfg):
    validate_prompt = get_prompt("suggestor", cfg["suggestor"]["prompt"])
    review_prompt = get_prompt("adjudicator", cfg["adjudicator"]["prompt"])
    max_retries = cfg["suggestor"]["max_retries"]
    review_retries = cfg["adjudicator"]["max_retries"]
    use_hierarchy = cfg["suggestor"]["use_hierarchy"]
    parent_levels = cfg["suggestor"]["hierarchy_parent_levels"]
    cem_cfg = cfg["cem"]

    def get_candidate_concepts(state: AgentState):
        min_patients = state["min_patients"]
        outcomes = state.get("outcomes", {})
        loaded_outcomes = load_person_count_with_descendants(db, min_patients, PERSON_COUNT_WITH_DESCENDANTS_PATH)
        new_outcome_count = 0
        for concept_id, outcome_info in loaded_outcomes.items():
            if concept_id not in outcomes:
                outcomes[concept_id] = outcome_info
                new_outcome_count += 1
        log("ok", "suggestor.filter", new_candidates=new_outcome_count)
                
        return {"outcomes": outcomes}

    def format_evidence(d):
        sources = {
            "Publication": "Pmid",
            "Product Label": "Splicer",
            "FAERS": "Faers",
        }
        return {
            label: {
                "Exact": d.get(f"exact{key}Count", 0),
                "Desc": d.get(f"descendant{key}Count", 0),
                "Parent": d.get(f"parent{key}Count", 0),
                "Ancestor": d.get(f"ancestor{key}Count", 0),
            }
            for label, key in sources.items()
        }
        
    def add_cem_result(state: AgentState): 
        outcomes = state["outcomes"]
        ingredient_concept_ids = state["ingredient_concept_ids"]

        if not state.get("cem_result"):
            cem_result, cem_result_detail = get_negative_controls(ingredient_concept_ids, **cem_cfg)
        else:
            log("skip", "cem.poll", reason="cached")
            cem_result = state["cem_result"]
            cem_result_detail = state["cem_result_detail"]

        t0 = time.perf_counter()
        count = [0,0,0]

        for concept_id in outcomes.keys():
            if concept_id in cem_result:
                outcomes[concept_id]["cem_result"] = cem_result[concept_id]
                outcomes[concept_id]["cem_result_detail"] = format_evidence(cem_result_detail[concept_id])
            else:
                outcomes[concept_id]["cem_result"] = 2
            count[outcomes[concept_id]["cem_result"]]+=1

        # analysis targets = outcomes CEM has a verdict for; no_cover outcomes are skipped downstream
        log("ok", "cem.evaluate", outcomes=len(outcomes), targets=count[0] + count[1],
            exclude=count[0], include=count[1], no_cover=count[2], time=secs(time.perf_counter() - t0))
        return {
            "outcomes": outcomes,
            "cem_result": cem_result,
            "cem_result_detail": cem_result_detail
        }
    
    def validate(state: AgentState):
        outcomes = state["outcomes"]
        ingredient_concepts = state["ingredient_concepts"]
        drug_infos = list(ingredient_concepts.values())
        t0 = time.perf_counter()
        validated = []
        no_cover = 0

        # no CEM record: out of analysis scope, neither validated nor reviewed
        todo = []
        for concept_id, outcome_info in outcomes.items():
            if "final_result" in outcome_info:
                continue
            if outcome_info["cem_result"] == 2:
                outcome_info["final_result"] = None
                no_cover += 1
            else:
                todo.append((concept_id, outcome_info))

        for i, (concept_id, outcome_info) in enumerate(todo, start=1):

            hierarchy = "(not provided)"
            if use_hierarchy:
                hierarchy = format_concept_neighbors(
                    db, concept_id, parent_levels=parent_levels, show_siblings=False, show_children=False
                )
                outcomes[concept_id]["hierarchy"] = hierarchy
            
            result = None
            for attempt in range(max_retries):
                try:
                    response = llm.invoke(
                        validate_prompt,
                        {"drug_infos": drug_infos, "outcome_name": outcome_info["name"], "hierarchy": hierarchy},
                        item=f"{i}/{len(todo)}",
                        concept=outcome_info["name"],
                        attempt=f"{attempt+1}/{max_retries}",
                    )
                    result = parse_json_dict(response)
                    if is_valid_result(result):
                        break
                    log("warn", "suggestor.validate", level="WARN", concept=outcome_info["name"],
                        attempt=f"{attempt+1}/{max_retries}", error="missing result/reason keys")
                except Exception as e:
                    log("warn", "suggestor.validate", level="WARN", concept=outcome_info["name"],
                        attempt=f"{attempt+1}/{max_retries}", error=e)
                    result = None
    
            if not is_valid_result(result):
                log("error", "suggestor.validate", level="ERROR", concept=outcome_info["name"],
                    reason="max retries exceeded")
                result = {"result": None, "reason": "validation_failed_after_retries"}
    
            outcomes[concept_id]["llm_validate_result"] = result
            validated.append(outcomes[concept_id])
            cem_result = outcomes[concept_id]["cem_result"]
            llm_validate_result = outcomes[concept_id]["llm_validate_result"]["result"]
    
            if cem_result == llm_validate_result and cem_result in (0, 1):
                outcomes[concept_id]["final_result"] = cem_result
            else:
                outcomes[concept_id]["final_result"] = 2

        excluded, included, pending = count_results(validated, "final_result")
        log("ok", "suggestor.validate", n=len(validated), excluded=excluded, included=included,
            pending=pending, skipped_no_cover=no_cover, time=secs(time.perf_counter() - t0))
        return {"outcomes": outcomes}

    # CEM and the first-pass judgment disagree (or the first pass failed); no_cover outcomes are never included
    def get_discordant_cases(outcomes):
        return {k: v for k, v in outcomes.items() if v.get("final_result") == 2 and v["cem_result"] in (0, 1)}

    def review_discordant(outcomes, ingredient_concepts, cem_rows):
        t0 = time.perf_counter()
        target_items = get_discordant_cases(outcomes)
        drug_infos = list(ingredient_concepts.values())

        for i, (concept_id, outcome_info) in enumerate(target_items.items(), start=1):

            outcome_context = {"outcome_name": outcome_info["name"]}
            if use_hierarchy:
                if "hierarchy" in outcomes[concept_id]:
                    outcome_context["hierarchy"] = outcomes[concept_id]["hierarchy"]
                else:
                    outcome_context["hierarchy"] = format_concept_neighbors(
                        db, concept_id, parent_levels=parent_levels, show_siblings=False, show_children=False
                    )
                    
            variables = {
                "drug_infos": drug_infos,
                "outcome_name": outcome_info["name"],
                "hierarchy": outcome_context.get("hierarchy", "(not provided)"),
                "cem_verdict": format_cem_verdict(outcomes[concept_id]["cem_result"]),
                "cem_evidence": format_cem_evidence(cem_rows.get(concept_id)),
                "first_judgment": format_first_judgment(outcomes[concept_id].get("llm_validate_result")),
            }
            
            result = None
            for attempt in range(review_retries):
                try:
                    response = llm2.invoke(
                        review_prompt,
                        variables,
                        task="review",
                        item=f"{i}/{len(target_items)}",
                        concept=outcome_info["name"],
                        attempt=f"{attempt+1}/{review_retries}",
                    )
                    result = parse_json_dict(response)
                    if is_valid_result(result):
                        break
                    log("warn", "suggestor.review", level="WARN", concept=outcome_info["name"],
                        attempt=f"{attempt+1}/{review_retries}", error="missing result/reason keys")
                except Exception as e:
                    log("warn", "suggestor.review", level="WARN", concept=outcome_info["name"],
                        attempt=f"{attempt+1}/{review_retries}", error=e)
                    result = None
    
            if not is_valid_result(result):
                log("error", "suggestor.review", level="ERROR", concept=outcome_info["name"],
                    reason="max retries exceeded")
                result = {"result": None, "reason": "validation_failed_after_retries"}
    
            outcomes[concept_id]["re_validate_result"] = result
            outcomes[concept_id]["final_result"] = outcomes[concept_id]["re_validate_result"]["result"]

        excluded, included, failed = count_results(target_items.values(), "final_result")
        log("ok", "suggestor.review", n=len(target_items), included=included, excluded=excluded,
            failed=failed, time=secs(time.perf_counter() - t0))
        return outcomes
        
    def adjudicator(state: AgentState):
        outcomes = state["outcomes"]
        ingredient_concepts = state["ingredient_concepts"]
        updated_outcomes = review_discordant(outcomes, ingredient_concepts, state.get("cem_result_detail") or {})
        
        return {
            "outcomes": updated_outcomes
        }
        
    return get_candidate_concepts, add_cem_result, validate, adjudicator

    
def build_graph(get_candidate_concepts, add_cem_result, validate, adjudicator):
    builder = StateGraph(AgentState)
    builder.add_node("get_candidate_concepts", get_candidate_concepts)
    builder.add_node("add_cem_result", add_cem_result)
    builder.add_node("validate", validate)
    builder.add_node("adjudicator", adjudicator)

    builder.add_edge(START, "get_candidate_concepts")
    builder.add_edge("get_candidate_concepts", "add_cem_result")
    builder.add_edge("add_cem_result", "validate")
    builder.add_edge("validate", "adjudicator")
    builder.add_edge("adjudicator", END)

    return builder


def build_suggestor(llm, llm2, db, cfg):
    nodes = build_nodes(llm, llm2, db, cfg)
    graph = build_graph(*nodes)
    return graph.compile()