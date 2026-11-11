from langchain_core.prompts import PromptTemplate
from core.prompt.common import NCO_DEFINITION, DRUG_INFO_HEADER, output_format


DIAGNOSE_V2 = PromptTemplate.from_template(
"""You are an expert in pharmacoepidemiology.

[Task]
Judge whether the current set of negative control outcomes (NCOs) for the given drug(s) is sufficient for the study, or whether the search should continue with a lower patient-count threshold to find more candidates.

""" + NCO_DEFINITION + """

[Target]
Target number of NCOs: {target_nco_count}

""" + DRUG_INFO_HEADER + """
{drug_infos}

[NCO set]
Raw number of NCOs: {nco_count}
One outcome name per line.
{nco_list}

[How the set was built]
Each outcome above was validated individually against the definition, and every outcome that passed is listed. The set has NOT been optimized as a whole:
- Hierarchically related concepts are all kept. If a parent concept and its child concepts are all valid, they are listed separately (e.g. a general condition and several of its specific subtypes).
- Near-synonymous or heavily overlapping concepts may also be listed separately.
Such concepts largely capture the same patients and the same bias, so they count as roughly one NCO, not several. The raw number therefore overstates the number of independent NCOs.

[How to judge]
1. Effective count: estimate how many independent NCOs remain after collapsing parent/child and near-synonymous concepts into one. Compare this effective count, not the raw number, with the target.
2. Diversity: the set should span many organ systems, disease types, and clinical domains. A set dominated by one body region or disease category cannot detect bias broadly, even if it is large.
3. Do NOT re-judge whether individual outcomes are valid NCOs; the drug information is context for judging diversity only.

[Decision]
- result = 1 : sufficient. The effective count is at or above the target, and diversity is adequate.
- result = 0 : insufficient. The effective count is below the target, or the set is concentrated in a few categories; continue the search.

""" + output_format(
    "State the raw and estimated effective counts relative to the target, and which categories are well covered "
    "and which are missing or overrepresented."
) + "\n"
)


# variant name -> template. Pick one with EXPERIMENT["diagnostician"]["prompt"].
# Variables available:
#   {drug_infos}        summarized drug labels
#   {target_nco_count}  EXPERIMENT["diagnostician"]["target_nco_count"]
#   {nco_count}         raw number of current NCOs
#   {nco_list}          one "- name" line per NCO
PROMPTS = {
    "v2": DIAGNOSE_V2,
}
