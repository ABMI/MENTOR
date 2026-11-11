"""Text shared by every role's prompt, so definitions and output rules cannot drift apart.

Each prompt is laid out in the same order:
  role line -> [Task] -> [Definition] -> inputs -> [How to ...] -> [Decision] -> [Output format]
Inputs that are the same for every call in a run (drug information) come before the per-call ones
(outcome, CEM evidence), so vLLM prefix caching can reuse them.
"""

NCO_DEFINITION = """[Definition: negative control outcome (NCO)]
A negative control outcome is an outcome (disease, symptom, or event) that has NO causal relationship with any of the given drugs. NCOs are used to detect residual confounding and systematic bias in observational studies. An outcome is a valid NCO ONLY IF it meets ALL of the following for EVERY given drug:
1. It is NOT an indication of the drug (a condition the drug treats or prevents).
2. It is NOT a contraindication or a known adverse effect of the drug.
3. It has NO causal relationship through the drug's mechanism of action, metabolism, or pharmacokinetic/pharmacodynamic pathways, i.e. there is no biological basis for the drug to cause or prevent it.
If the outcome violates ANY of the three conditions, it is NOT a valid NCO."""

DRUG_INFO_HEADER = "[Drug information (reference only, not exhaustive)]"

REFERENCE_RULES = """- The drug information is REFERENCE ONLY. It is not an exhaustive or authoritative list.
- Do NOT conclude that there is no causal relationship simply because the outcome is absent from the drug information. Absence from the list is NOT evidence of no relationship.
- Base your judgment on your own expert knowledge of the drug's mechanism of action, metabolism, pharmacokinetics/pharmacodynamics, and the pathophysiology of the outcome."""

NCO_DECISION = """[Decision]
- result = 1 : the outcome IS a valid NCO (it meets all three conditions).
- result = 0 : the outcome is NOT a valid NCO (it violates at least one condition)."""

JSON_ONLY = "- Do NOT output any text outside the JSON object (no preamble, reasoning, or code fences)."

REASON_LENGTH = "2-4 complete sentences"


def output_format(reason_rule):
    """[Output format] block for the {"result", "reason"} prompts. Braces are escaped for PromptTemplate."""
    return f"""[Output format]
- Output ONLY a single JSON object in exactly this format: {{{{"result": 0, "reason": ""}}}}
- "reason": {REASON_LENGTH}. {reason_rule}
{JSON_ONLY}"""
