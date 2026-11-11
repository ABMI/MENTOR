from core.config.cdm import config as cdm_config


CONCEPT_TABLE = cdm_config["concept_table"]
CONCEPT_ANCESTOR_TABLE = cdm_config["concept_ancestor_table"]


def run_query(db, sql: str, name: str) -> list:
    success, rows, error = db.query(sql, name=name)
    return rows if success else []   # db.query already logged the error


def build_parents_sql(concept_id: int, levels: int = 1):
    return f"""
SELECT
    c.concept_id,
    c.concept_name,
    ca.min_levels_of_separation AS level
FROM {CONCEPT_ANCESTOR_TABLE} ca
JOIN {CONCEPT_TABLE} c
  ON c.concept_id = ca.ancestor_concept_id
WHERE ca.descendant_concept_id = {concept_id}
  AND ca.min_levels_of_separation BETWEEN 1 AND {levels}
ORDER BY level, c.concept_name;
"""


def build_children_sql(concept_id: int, levels: int = 1):
    return f"""
SELECT
    c.concept_id,
    c.concept_name,
    ca.min_levels_of_separation AS level
FROM {CONCEPT_ANCESTOR_TABLE} ca
JOIN {CONCEPT_TABLE} c
  ON c.concept_id = ca.descendant_concept_id
WHERE ca.ancestor_concept_id = {concept_id}
  AND ca.min_levels_of_separation BETWEEN 1 AND {levels}
ORDER BY level, c.concept_name;
"""


def build_siblings_sql(concept_id: int):
    return f"""
SELECT DISTINCT
    c.concept_id,
    c.concept_name,
    1 AS level
FROM {CONCEPT_ANCESTOR_TABLE} p
JOIN {CONCEPT_ANCESTOR_TABLE} s
  ON s.ancestor_concept_id = p.ancestor_concept_id
 AND s.min_levels_of_separation = 1
JOIN {CONCEPT_TABLE} c
  ON c.concept_id = s.descendant_concept_id
WHERE p.descendant_concept_id = {concept_id}
  AND p.min_levels_of_separation = 1
  AND s.descendant_concept_id <> {concept_id}
ORDER BY c.concept_name;
"""


def format_concept_neighbors(
    db,
    concept_id: int,
    parent_levels: int = 1,
    child_levels: int = 1,
    show_parents: bool = True,
    show_siblings: bool = True,
    show_children: bool = True,
) -> str:
    concept_id = int(concept_id)
    sections = []
    if show_parents:
        sections.append(("parents", f"Parent concepts (up to {parent_levels} levels, nearest first)",
                         build_parents_sql(concept_id, parent_levels)))
    if show_siblings:
        sections.append(("siblings", "Sibling concepts (share a direct parent)", build_siblings_sql(concept_id)))
    if show_children:
        sections.append(("children", f"Child concepts (up to {child_levels} levels, nearest first)",
                         build_children_sql(concept_id, child_levels)))

    blocks = []
    for key, title, sql in sections:
        rows = run_query(db, sql, f"hierarchy.{key}")
        items = [f"- {r['concept_name']} (level {r['level']})" for r in rows] or ["- (none)"]
        blocks.append("\n".join([f"{title}:"] + items))
    return "\n\n".join(blocks)
