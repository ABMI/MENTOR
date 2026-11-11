import os
import csv
from core.config.cdm import config as cdm_config
from core.utils import log, secs, span
import time


CONCEPT_TABLE = cdm_config["concept_table"]
CONCEPT_ANCESTOR_TABLE = cdm_config["concept_ancestor_table"]
CONDITION_OCCURRENCE_TABLE = cdm_config["condition_occurrence_table"]
EXCLUDE_CONCEPTS_PATH = cdm_config["exclude_concepts_path"]


def load_excluded_concept_ids(path):
    if not path or not os.path.exists(path):
        log("warn", "pool.load", level="WARN", reason="exclude list not found", path=path)
        return set()
    ids = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = str(row.get("concept_id", "")).strip()
            if cid:
                ids.add(cid)
    return ids


def build_generate_sql(min_patients: int):
    return f"""
SELECT
    c.concept_id                 AS concept_id,
    c.concept_name               AS concept_name,
    COUNT(DISTINCT co.person_id) AS patient_count
FROM {CONCEPT_TABLE} c
JOIN {CONCEPT_ANCESTOR_TABLE} ca
  ON c.concept_id = ca.ancestor_concept_id
JOIN {CONDITION_OCCURRENCE_TABLE} co
  ON ca.descendant_concept_id = co.condition_concept_id
WHERE c.domain_id = 'Condition'
  AND c.standard_concept = 'S'
  AND c.invalid_reason IS NULL
GROUP BY
    c.concept_id,
    c.concept_name
HAVING
    COUNT(DISTINCT co.person_id) >= {min_patients}
ORDER BY
    patient_count DESC;
"""


def generate_person_count_with_descendants(db, min_patients, save_path):
    with span("pool.generate", min_patients=min_patients) as end:
        outcomes = _generate(db, min_patients, save_path)
        end.update(loaded=len(outcomes), path=save_path)
    return outcomes


def _generate(db, min_patients, save_path):
    outcomes = {}

    excluded_ids = load_excluded_concept_ids(EXCLUDE_CONCEPTS_PATH)  

    sql = build_generate_sql(min_patients)
    success, rows, error = db.query(sql)
    if not success:
        log("error", "pool.generate", level="ERROR", error=error)
        return outcomes

    n_excluded = 0
    for row in rows:
        concept_id = int(row["concept_id"])
        if str(concept_id) in excluded_ids:  
            n_excluded += 1
            continue
        outcomes[concept_id] = {
            "name": row["concept_name"],
            "patient_count": int(row["patient_count"]),
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_id", "concept_name", "patient_count"])
        for concept_id, info in outcomes.items():
            writer.writerow([concept_id, info["name"], info["patient_count"]])
    log("ok", "pool.query", candidates=len(rows), excluded=n_excluded, loaded=len(outcomes))
    return outcomes


def load_person_count_with_descendants(db, min_patients, path):
    outcomes = {}

    if not os.path.exists(path):
        log("warn", "pool.retrieve", level="WARN", reason="file not found", path=path)
        if db is None:
            log("error", "pool.retrieve", level="ERROR", reason="db required to generate")
            return outcomes
        generate_person_count_with_descendants(db, 1, path)
        if not os.path.exists(path):
            return outcomes

    t0 = time.perf_counter()
    excluded_ids = load_excluded_concept_ids(EXCLUDE_CONCEPTS_PATH)

    n_candidates = n_excluded = 0
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            patient_count = int(row["patient_count"])
            if patient_count < min_patients:
                continue
            n_candidates += 1
            concept_id = int(row["concept_id"])
            if str(concept_id) in excluded_ids:
                n_excluded += 1
                continue
            outcomes[concept_id] = {
                "name": row["concept_name"],
                "patient_count": patient_count,
            }

    log("ok", "pool.retrieve",
        min_patients=min_patients,
        candidates=n_candidates,
        excluded=n_excluded,
        loaded=len(outcomes),
        time=secs(time.perf_counter() - t0))
    return outcomes