"""Final NCOs as files that OHDSI tools import directly (rule-based, no LLM).

concept_set.json                ATLAS concept set expression (Concept Sets > Import > Concept Set Expression,
                                or the negative control concept set of an ATLAS Estimation design)
negative_control_outcomes.csv   CohortGenerator / Strategus negativeControlOutcomeCohortSet
                                (cohortId, cohortName, outcomeConceptId)

Patient counts in the Concept Pool include descendant concepts, so the outputs do too:
includeDescendants = true in the concept set, and generateNegativeControlOutcomeCohorts(...)
should be run with detectOnDescendants = TRUE.
"""
import csv
import json
from pathlib import Path

from core.config.cdm import config as cdm_config
from core.utils import log

CONCEPT_TABLE = cdm_config["concept_table"]
CONCEPT_FIELDS = ["concept_id", "concept_name", "domain_id", "vocabulary_id", "concept_class_id",
                  "standard_concept", "concept_code", "invalid_reason"]
STANDARD_CAPTION = {"S": "Standard", "C": "Classification"}
INVALID_CAPTION = {"D": "Invalid", "U": "Invalid"}


def final_ncos(final_state):
    """Final NCOs (final_result == 1), largest patient count first: [(concept_id, name, patient_count)]."""
    rows = [(int(cid), info["name"], int(info.get("patient_count", 0)))
            for cid, info in (final_state.get("outcomes") or {}).items() if info.get("final_result") == 1]
    return sorted(rows, key=lambda r: (-r[2], r[0]))


def fetch_concepts(db, concept_ids):
    """concept_id -> concept row from the CDM concept table ({} if the query fails)."""
    if not concept_ids:
        return {}
    sql = (f"SELECT {', '.join(CONCEPT_FIELDS)} FROM {CONCEPT_TABLE} "
           f"WHERE concept_id IN ({', '.join(str(int(c)) for c in concept_ids)})")
    ok, rows, error = db.query(sql, name="concept_set.fetch")
    if not ok:
        log("warn", "concept_set", level="WARN", reason="concept lookup failed; writing IDs and names only", error=error)
        return {}
    return {int(r["concept_id"]): r for r in rows}


def concept_set_expression(ncos, concepts):
    """ATLAS concept set expression: one item per NCO, descendants included."""
    items = []
    for cid, name, _ in ncos:
        c = concepts.get(cid, {})
        standard = c.get("standard_concept") or ""
        invalid = c.get("invalid_reason") or "V"
        items.append({
            "concept": {
                "CONCEPT_ID": cid,
                "CONCEPT_NAME": c.get("concept_name", name),
                "STANDARD_CONCEPT": standard,
                "STANDARD_CONCEPT_CAPTION": STANDARD_CAPTION.get(standard, "Non-Standard"),
                "INVALID_REASON": invalid,
                "INVALID_REASON_CAPTION": INVALID_CAPTION.get(invalid, "Valid"),
                "CONCEPT_CODE": c.get("concept_code", ""),
                "DOMAIN_ID": c.get("domain_id", "Condition"),
                "VOCABULARY_ID": c.get("vocabulary_id", ""),
                "CONCEPT_CLASS_ID": c.get("concept_class_id", ""),
            },
            "isExcluded": False,
            "includeDescendants": True,
            "includeMapped": False,
        })
    return {"items": items}


def negative_control_rows(ncos):
    """negativeControlOutcomeCohortSet rows; cohortId = outcomeConceptId (unique and traceable; renumber if it
    collides with other cohort IDs in a study)."""
    return [{"cohortId": cid, "cohortName": name, "outcomeConceptId": cid} for cid, name, _ in ncos]


def write_nco_files(final_state, run_dir, db=None):
    """Write concept_set.json and negative_control_outcomes.csv to run_dir; returns the number of NCOs."""
    run_dir = Path(run_dir)
    ncos = final_ncos(final_state)
    concepts = fetch_concepts(db, [c for c, _, _ in ncos]) if db is not None else {}
    with open(run_dir / "concept_set.json", "w", encoding="utf-8") as f:
        json.dump(concept_set_expression(ncos, concepts), f, ensure_ascii=False, indent=2)
    with open(run_dir / "negative_control_outcomes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cohortId", "cohortName", "outcomeConceptId"])
        writer.writeheader()
        writer.writerows(negative_control_rows(ncos))
    log("ok", "concept_set", ncos=len(ncos), with_metadata=bool(concepts), path=str(run_dir))
    return len(ncos)
