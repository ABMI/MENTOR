from langchain_core.prompts import PromptTemplate
from core.prompt.common import (
    NCO_DEFINITION, DRUG_INFO_HEADER, REFERENCE_RULES, NCO_DECISION, output_format,
)


REVIEW_V2 = PromptTemplate.from_template(
"""You are an expert in pharmacoepidemiology acting as an adjudicator.

[Task]
Two sources disagree on whether the outcome below is a valid negative control outcome (NCO) for the given drug(s):
- CEM (Common Evidence Model): an automated screen based on existing evidence of drug-outcome associations from publications, product labels, and spontaneous reports.
- A first-pass LLM judgment based on pharmacological reasoning (it may be unavailable if that step failed).
Resolve the disagreement and make the final decision.

""" + NCO_DEFINITION + """

""" + DRUG_INFO_HEADER + """
{drug_infos}

[Outcome]
{outcome_name}

[Concept hierarchy]
{hierarchy}

[CEM verdict]
{cem_verdict}

[CEM evidence]
{cem_evidence}

[First-pass LLM judgment]
{first_judgment}

[How to read the CEM evidence]
Each number is a count of evidence items linking the drug and the outcome, at four matching levels:
- exact: the outcome concept itself; descendant: narrower (more specific) concepts; parent: the direct parent concept; ancestor: broader (more general) concepts.
Sources:
- Publications: drug-outcome relationships found in the published medical literature. A publication can describe any relationship, including the drug being studied as a treatment for the outcome, so it signals an association but not its direction.
- Product labels: the outcome is listed as an adverse reaction in official drug labels.
- Spontaneous reports: the drug-outcome pair shows a statistical signal in spontaneous adverse event reports. These signals are prone to reporting bias and confounding by indication: complications of the disease the drug treats can be flagged without being caused by the drug.
How CEM decides:
- ANY evidence from any source at any matching level excludes the outcome from the NCO list; the closer all counts are to 0, the better the candidate.
- This rule is deliberately strict. Evidence that exists only at the parent/ancestor level comes from broader concepts and may not apply to this specific outcome.
- Zero evidence does NOT prove there is no relationship; CEM coverage is incomplete.
Flags: indication_or_contraindication=1 means CEM matched an indication/contraindication; too_broad=1 means the concept is too general to be a useful NCO; drug_induced=1 means the concept itself implies drug causation; pregnancy_related=1 marks pregnancy outcomes; not_prevalent=1 means the outcome is rare; optimized_out=1 means CEM dropped it as redundant with a related concept.

[How to decide]
""" + REFERENCE_RULES + """
- Then apply the following rules:
1. Identify which source is likely wrong and why. Do not side with either source by default.
2. An exact or descendant product label count means a labeled adverse reaction (condition 2 is violated) unless the label match is clearly a different condition.
3. Other exact/descendant evidence (publications, spontaneous reports) combined with a plausible pathway (indication, adverse effect, or mechanism) means the outcome is NOT a valid NCO.
4. If the only evidence is parent/ancestor-level, or is explainable by confounding by indication, and you find no mechanistic link, the outcome can be a valid NCO.
5. If you remain uncertain after weighing everything, choose result = 0. A false NCO biases calibration more than a missing one.

""" + NCO_DECISION + """

""" + output_format(
    "State which source you sided with, the decisive evidence or mechanism, and why the other source was wrong."
) + "\n"
)


# variant name -> template. Pick one with EXPERIMENT["adjudicator"]["prompt"].
# Variables available:
#   {drug_infos}      summarized drug labels
#   {outcome_name}    outcome concept name
#   {hierarchy}       parent concepts, or "(not provided)" when use_hierarchy is off
#   {cem_verdict}     CEM decision in words (candidate / not candidate)
#   {cem_evidence}    evidence counts per source and matching level, CEM flags, as text
#   {first_judgment}  the suggestor's first-pass result and reason
PROMPTS = {
    "v2": REVIEW_V2,
}
