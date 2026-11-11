from dotenv import load_dotenv
from pathlib import Path
import os
load_dotenv() 

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


config = {
    "concept_table": os.getenv("CONCEPT_TABLE"),
    "concept_ancestor_table": os.getenv("CONCEPT_ANCESTOR_TABLE"),
    "condition_occurrence_table": os.getenv("CONDITION_OCCURRENCE_TABLE"),
    "person_count_with_descendants_path": str(DATA_DIR / "person_count_with_descendants.csv"),   # built from the CDM on first run
    "exclude_concepts_path": str(DATA_DIR / "excluded_concepts_broad_preg.csv"),
}