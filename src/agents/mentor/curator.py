from typing import TypedDict, Dict, Any, List
from langgraph.graph import START, END, StateGraph
from core.prompt import get_prompt
from core.utils import parse_json_dict
from core.config.cdm import config as cdm_config
from core.utils import log, secs
from core import openfda
import json
import time
import tiktoken
_enc = tiktoken.get_encoding("cl100k_base")


CONCEPT_TABLE = cdm_config["concept_table"]


def count_tokens(obj):
    if not isinstance(obj, str):
        obj = json.dumps(obj, ensure_ascii=False)
    return len(_enc.encode(obj))


class AgentState(TypedDict):
    ingredient_concept_ids: List
    ingredient_concepts: Dict[int, Any]
    is_all_known_ingredient: bool


CURATOR_MODES = ("summary", "raw", "name_only")


def build_nodes(llm, db, cfg):
    summarize_prompt = get_prompt("curator", cfg["curator"]["prompt"])
    mode = cfg["curator"]["mode"]

    def check_valid_ingredient_concept_id(ingredient_concept_id):
        sql = f"""
        SELECT COUNT(*) AS cnt
        FROM {CONCEPT_TABLE} c
        WHERE c.concept_id = {int(ingredient_concept_id)}
            AND c.concept_class_id = 'Ingredient'
            AND c.standard_concept = 'S'
        """
        success, rows, error = db.query(sql, concept_id=ingredient_concept_id)
        if not success:
            log("error", "curator.validate", level="ERROR", concept_id=ingredient_concept_id, error=error)
            return False
        return bool(rows) and rows[0]["cnt"] > 0

    def get_ingredient_name(ingredient_concept_id):
        sql = f"""
        SELECT c.concept_name
        FROM {CONCEPT_TABLE} c
        WHERE c.concept_id = {int(ingredient_concept_id)}
        """
        success, rows, error = db.query(sql, concept_id=ingredient_concept_id)
        if not success:
            log("error", "curator.validate", level="ERROR", concept_id=ingredient_concept_id, error=error)
            return ""
        return rows[0]["concept_name"] if rows else ""

    def collect_ingredient_name(state: AgentState):
        ingredient_concept_ids = state["ingredient_concept_ids"]
        is_all_known_ingredient = True
        ingredient_concepts = {}
        t0 = time.perf_counter()

        for ingredient_concept_id in ingredient_concept_ids:
            is_valid = check_valid_ingredient_concept_id(ingredient_concept_id)

            if not is_valid:
                is_all_known_ingredient = False
                log("error", "curator.validate", level="ERROR", concept_id=ingredient_concept_id,
                    reason="invalid or unknown ingredient concept id")
                break

            name = get_ingredient_name(ingredient_concept_id)
            ingredient_concepts[ingredient_concept_id] = {"ingredient_name": name}

        log("ok" if is_all_known_ingredient else "error", "curator.validate",
            level="INFO" if is_all_known_ingredient else "ERROR",
            n=len(ingredient_concept_ids), valid=len(ingredient_concepts), time=secs(time.perf_counter() - t0))
        
        return {
            "ingredient_concepts": ingredient_concepts,
            "is_all_known_ingredient": is_all_known_ingredient,
        }

    def collect_ingredient_info(state: AgentState):
        ingredient_concept_ids = state["ingredient_concept_ids"]
        ingredient_concepts = state["ingredient_concepts"]

        is_all_known_ingredient = True

        names = [ingredient_concepts[cid]["ingredient_name"] for cid in ingredient_concept_ids]
        results = openfda.collect(names, **cfg["openfda"])

        for ingredient_concept_id, ingredient_name, result in zip(ingredient_concept_ids, names, results):
            found = result["found"]
            if not found:
                is_all_known_ingredient = False
                log("error", "curator.collect_info", level="ERROR", ingredient=ingredient_name, reason="not found in openfda")
                break

            drug_info = result["drug_info"]

            indications_raw = drug_info["indications"]
            mechanism_of_action_raw = drug_info["mechanism_of_action"]
            contraindications_raw = drug_info["contraindications"]
            adverse_effects_raw = drug_info["adverse_effects"]

            indications = indications_raw[0]["text"] if indications_raw else ""
            mechanism_of_action = mechanism_of_action_raw[0]["text"] if mechanism_of_action_raw else ""
            contraindications = contraindications_raw[0]["text"] if contraindications_raw else ""

            adverse_effects = ""
            for section in ["boxed_warning", "adverse_reactions", "warnings_and_precautions", "warnings"]:
                section_data = adverse_effects_raw.get(section, [])
                if len(section_data) != 0:
                    adverse_effects += "\n# " + section + "\n"
                    adverse_effects += section_data[0]["text"]

            ingredient_concepts[ingredient_concept_id] = {
                "ingredient_name": ingredient_name,
                "indications": indications,
                "mechanism_of_action": mechanism_of_action,
                "contraindications": contraindications,
                "adverse_effects": adverse_effects,
            }

        return {
            "ingredient_concepts": ingredient_concepts,
            "is_all_known_ingredient": is_all_known_ingredient,
        }

    def summarize(state: AgentState):
        ingredient_concept_ids = state["ingredient_concept_ids"]
        ingredient_concepts = state["ingredient_concepts"]

        # ablation: hand the downstream prompts the raw labels or the names alone, without the LLM summary
        if mode != "summary":
            for ingredient_concept_id in ingredient_concept_ids:
                ingredient_info = ingredient_concepts[ingredient_concept_id]
                log("skip", "curator.summarize", ingredient=ingredient_info["ingredient_name"], mode=mode,
                    tokens=count_tokens(ingredient_info))
            return {"ingredient_concepts": ingredient_concepts}

        for ingredient_concept_id in ingredient_concept_ids:
            ingredient_info = ingredient_concepts[ingredient_concept_id]
            ingredient_name = ingredient_info["ingredient_name"]
            t0 = time.perf_counter()
            before = count_tokens(ingredient_info)
            response = llm.invoke(summarize_prompt, {"drug_info": ingredient_info}, ingredient=ingredient_name)
            summarized = parse_json_dict(response)
            after = count_tokens(summarized)
            reduction = (1 - after / before) * 100 if before else 0.0
            ingredient_concepts[ingredient_concept_id] = summarized
            log("ok", "curator.summarize", ingredient=ingredient_name, tokens_before=before, tokens_after=after,
                reduction=f"{reduction:.1f}", time=secs(time.perf_counter() - t0))

        return {
            "ingredient_concepts": ingredient_concepts,
        }

    return collect_ingredient_name, collect_ingredient_info, summarize


def route_if_known(state: AgentState):
    if not state.get("is_all_known_ingredient", False):
        return "end"
    return "continue"


def build_graph(collect_ingredient_name, collect_ingredient_info, summarize, mode="summary"):
    # name_only skips the openFDA lookup: the names go straight to summarize, which passes them through
    after_name = "summarize" if mode == "name_only" else "collect_info"

    builder = StateGraph(AgentState)

    builder.add_node("collect_name", collect_ingredient_name)
    builder.add_node("collect_info", collect_ingredient_info)
    builder.add_node("summarize", summarize)

    builder.add_edge(START, "collect_name")
    builder.add_conditional_edges(
        "collect_name",
        route_if_known,
        {
            "continue": after_name,
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "collect_info",
        route_if_known,
        {
            "continue": "summarize",
            "end": END,
        },
    )
    builder.add_edge("summarize", END)
    return builder


def build_curator(llm, db, cfg):
    if cfg["curator"]["mode"] not in CURATOR_MODES:
        raise ValueError(f"unknown curator mode: {cfg['curator']['mode']} (allowed: {', '.join(CURATOR_MODES)})")
    collect_ingredient_name, collect_ingredient_info, summarize = build_nodes(llm, db, cfg)
    graph = build_graph(collect_ingredient_name, collect_ingredient_info, summarize, cfg["curator"]["mode"])
    return graph.compile()