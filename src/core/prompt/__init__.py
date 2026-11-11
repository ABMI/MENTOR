import hashlib
from core.prompt import curator, suggestor, adjudicator, diagnostician

_MODULES = {
    "curator": curator,
    "suggestor": suggestor,
    "adjudicator": adjudicator,
    "diagnostician": diagnostician,
}

# variables the calling code passes to each role's prompt; a variant may use fewer
VARIABLES = {
    "curator": {"drug_info"},
    "suggestor": {"drug_infos", "outcome_name", "hierarchy"},
    "adjudicator": {"drug_infos", "outcome_name", "hierarchy",
                    "cem_verdict", "cem_evidence", "first_judgment"},
    "diagnostician": {"drug_infos", "target_nco_count", "nco_count", "nco_list"},
}


def get_prompt(role, name):
    prompts = _MODULES[role].PROMPTS
    if name not in prompts:
        raise KeyError(f"unknown {role} prompt: {name} (allowed: {', '.join(prompts)})")
    prompt = prompts[name]
    unknown = set(prompt.input_variables) - VARIABLES[role]
    if unknown:
        raise ValueError(f"{role} prompt {name} uses unknown variables: {sorted(unknown)} "
                         f"(allowed: {sorted(VARIABLES[role])})")
    return prompt


def prompt_hash(prompt):
    return hashlib.sha1(prompt.template.encode()).hexdigest()[:8]
