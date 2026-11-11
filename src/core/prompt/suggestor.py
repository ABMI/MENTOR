from langchain_core.prompts import PromptTemplate
from core.prompt.common import (
    NCO_DEFINITION, DRUG_INFO_HEADER, REFERENCE_RULES, NCO_DECISION, output_format,
)


VALIDATE_V2 = PromptTemplate.from_template(
"""You are an expert in pharmacoepidemiology.

[Task]
Judge whether the outcome below is a valid negative control outcome (NCO) for the given drug(s).

""" + NCO_DEFINITION + """

""" + DRUG_INFO_HEADER + """
{drug_infos}

[Outcome]
{outcome_name}

[Concept hierarchy]
{hierarchy}

[How to reason]
""" + REFERENCE_RULES + """
- Reason from the biological mechanism linking (or not linking) the drug and the outcome, using the drug information only as supporting context.

""" + NCO_DECISION + """

""" + output_format(
    "If result = 1, explain why there is no causal relationship, grounded in the drug's mechanism and the "
    "outcome's pathophysiology (not merely its absence from the drug information). If result = 0, name the "
    "violated condition and the specific relationship (indication, contraindication, adverse effect, or mechanistic pathway)."
) + "\n"
)


# variant name -> template. Pick one with EXPERIMENT["suggestor"]["prompt"].
# Variables available:
#   {drug_infos}    summarized drug labels
#   {outcome_name}  outcome concept name
#   {hierarchy}     parent concepts, or "(not provided)" when use_hierarchy is off
PROMPTS = {
    "v2": VALIDATE_V2,
}
