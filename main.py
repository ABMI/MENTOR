import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))
from dotenv import load_dotenv
load_dotenv() 
from agents.mentor.agent import build_mentor, log_run_config
from agents.mentor import agent as mentor_agent
from core.database import Database
from core.config.database import config as database_config
from core.config.experiment import build_config
from core.utils import span, tee_stdout
from core.concept_set import write_nco_files
from datetime import datetime
import csv
import json
import time
import traceback

CEM_LABELS = {0: "evidence found", 1: "no evidence", 2: "not covered"}


def build_initial_state(ingredient_concept_ids, search):
    state: mentor_agent.AgentState = {
        "ingredient_concept_ids": ingredient_concept_ids,
        "minimum_min_patients": search["min_floor"],
        "init_min_patients": search["min_patients"],
        "min_patients": search["min_patients"],
        "step": search["step"],
    }
    return state


def main(ingredient_concept_ids, overrides=None, run_dir=".", cfg=None):
    """overrides: changes on top of EXPERIMENT; run_dir: where per-run outputs (cem_result.csv) go."""
    cfg = cfg or build_config(overrides)
    cfg["cem"]["save_path"] = str(Path(run_dir) / "cem_result.csv")
    log_run_config(cfg, overrides)
    db = Database(database_config)
    mentor = build_mentor(db, cfg)
    state = build_initial_state(ingredient_concept_ids, cfg["search"])
    final_state = mentor.invoke(state)
    db.close()
    
    return final_state


def save_nco_list(final_state, path):
    """Final NCOs (final_result == 1) as a CSV, largest patient count first."""
    rows = []
    for concept_id, info in final_state.get("outcomes", {}).items():
        if info.get("final_result") != 1:
            continue
        first = info.get("llm_validate_result") or {}
        review = info.get("re_validate_result")
        rows.append({
            "concept_id": concept_id,
            "concept_name": info["name"],
            "patient_count": info["patient_count"],
            "cem": CEM_LABELS.get(int(info.get("cem_result", 2))),
            "decided_by": "review" if review else "cem+llm agree",
            "reason": (review or first).get("reason", ""),
        })
    rows.sort(key=lambda r: -r["patient_count"])
    fields = ["concept_id", "concept_name", "patient_count", "cem", "decided_by", "reason"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_experiment(ingredient_concept_ids, name, overrides=None, cfg=None,
                   batch_id=None, exp=(1, 1), log_root="logs"):
    """One full run -> logs/<batch_id>_<name>/ with run.log, final_state.json, nco_list.csv, cem_result.csv,
    concept_set.json and negative_control_outcomes.csv.

    Exceptions are written to run.log and reported as status="error" instead of raised,
    so a queue of experiments keeps going.
    """
    cfg = cfg or build_config(overrides)
    batch_id = batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(log_root) / f"{batch_id}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    final_state, status, t0 = None, "error", time.perf_counter()

    with tee_stdout(run_dir / "run.log"):
        try:
            with span("run", llm=True, batch=batch_id, exp=f"{exp[0]}/{exp[1]}", name=name,
                      ingredients=ingredient_concept_ids) as end:
                final_state = main(ingredient_concept_ids, overrides, run_dir, cfg)
                status = final_state.get("verdict", "invalid_ingredient")
                end.update(
                    status=status,
                    iters=(cfg["search"]["min_patients"] - final_state["min_patients"]) // cfg["search"]["step"],
                    nco_count=sum(1 for v in final_state.get("outcomes", {}).values() if v.get("final_result") == 1),
                )
        except Exception:
            for line in traceback.format_exc().splitlines():
                print(f"# {line}")

    nco_count = None
    if final_state is not None:
        with open(run_dir / "final_state.json", "w", encoding="utf-8") as f:
            json.dump(final_state, f, ensure_ascii=False, indent=2)
        nco_count = len(save_nco_list(final_state, run_dir / "nco_list.csv"))
        db = Database(database_config)
        try:
            write_nco_files(final_state, run_dir, db)   # concept_set.json (ATLAS) + negative_control_outcomes.csv (Strategus)
        finally:
            db.close()

    return {
        "name": name,
        "status": status,
        "nco_count": nco_count,
        "elapsed": time.perf_counter() - t0,
        "run_dir": run_dir,
    }
