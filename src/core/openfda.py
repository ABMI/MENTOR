from __future__ import annotations
import hashlib
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
import requests
from core.utils import log, secs, span


API = "https://api.fda.gov/drug/label.json"
API_KEY = os.environ.get("OPENFDA_API_KEY")
MAX_SKIP = 25_000
PAGE = 100
ADVERSE_FIELDS = [
    "boxed_warning",
    "adverse_reactions",
    "warnings_and_precautions",
    "warnings",
]
FALLBACK_FIELDS = ["clinical_pharmacology"]


class OpenFDAClient:
    def __init__(self, api_key: str | None = API_KEY, min_interval: float = 0.3,
                 verbose: bool = False):
        self.api_key = api_key
        self.min_interval = min_interval
        self.verbose = verbose         
        self._last = 0.0
        self.session = requests.Session()
        self.request_count = 0
        self.request_time = 0.0        

    def _throttle(self) -> None:
        gap = time.time() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.time()

    def get(self, params: dict) -> dict | None:
        if self.api_key:
            params = {**params, "api_key": self.api_key}

        for attempt in range(4):
            self._throttle()
            t0 = time.perf_counter()
            try:
                r = self.session.get(API, params=params, timeout=60)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            elapsed = time.perf_counter() - t0
            self.request_count += 1
            self.request_time += elapsed

            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                wait = 60 if not self.api_key else 5
                log("warn", "openfda", level="WARN", reason="rate limited (429)", wait=secs(wait))
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

        raise RuntimeError("Max retries exceeded")


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def _collect_texts(labels: list[dict], field: str) -> list[dict]:
    by_hash: dict[str, str] = {}
    counts: Counter = Counter()

    for lab in labels:
        val = lab.get(field)
        chunks = val if isinstance(val, list) else ([val] if isinstance(val, str) else [])
        seen_here = set()
        for c in chunks:
            if not isinstance(c, str) or not c.strip():
                continue
            h = hashlib.md5(_norm(c).encode()).hexdigest()
            by_hash.setdefault(h, c.strip())
            if h not in seen_here:
                counts[h] += 1
                seen_here.add(h)

    return [
        {"text": by_hash[h], "label_count": n}
        for h, n in counts.most_common()
    ]


def build_query(ingredient: str) -> str:
    t = ingredient.replace('"', '\\"')
    return f'(openfda.substance_name:"{t}" OR openfda.generic_name:"{t}")'


def is_single_ingredient(label: dict, ingredient: str) -> bool:
    """True if the label's product has exactly one active substance and it is this ingredient (salt forms allowed,
    e.g. "SITAGLIPTIN PHOSPHATE"). Falls back to generic_name when substance_name is missing; unknown -> False."""
    word = re.compile(rf"\b{re.escape(ingredient.upper())}\b")
    meta = label.get("openfda", {})
    substances = {s.strip().upper() for s in meta.get("substance_name", []) if s.strip()}
    if substances:
        return len(substances) == 1 and bool(word.search(next(iter(substances))))
    generic = " ".join(meta.get("generic_name", [])).upper()
    return bool(generic) and bool(word.search(generic)) and not re.search(r"\bAND\b|/|,", generic)


def fetch(client: OpenFDAClient, ingredient: str, max_labels: int = 1000, single_ingredient: bool = True) -> dict:
    t_start = time.perf_counter()
    req0, http0 = client.request_count, client.request_time
    q = build_query(ingredient)
    now = datetime.now(timezone.utc).isoformat()

    first = client.get({"search": q, "limit": 1})

    if first is None:
        log("warn", "openfda.label", level="WARN", ingredient=ingredient, reason="not found",
            requests=client.request_count - req0, http=secs(client.request_time - http0),
            time=secs(time.perf_counter() - t_start))
        return {
            "query": ingredient,
            "found": False,
            "not_marketed_us": True,
            "label_count": 0,
            "drug_info": None,
            "retrieved_at": now,
        }

    total = int(first["meta"]["results"]["total"])
    target = min(total, max_labels, MAX_SKIP)

    labels: list[dict] = []
    while len(labels) < target:
        page = client.get({"search": q, "limit": min(PAGE, target - len(labels)),
                           "skip": len(labels)})
        if not page or not page.get("results"):
            break
        labels.extend(page["results"])

    # Combination products (e.g. lisinopril/hydrochlorothiazide, empagliflozin/metformin) also match the query, and
    # their wording can win the most-frequent-version vote below. Use single-ingredient labels when there are any.
    n_fetched = len(labels)
    n_single = None
    if single_ingredient:
        single = [lab for lab in labels if is_single_ingredient(lab, ingredient)]
        n_single = len(single)
        if single:
            labels = single
        else:
            log("warn", "openfda.label", level="WARN", ingredient=ingredient,
                reason="no single-ingredient label; using all labels including combination products")

    drug_info = {
        "drug_name": ingredient,
        "indications": _collect_texts(labels, "indications_and_usage"),
        "mechanism_of_action": _collect_texts(labels, "mechanism_of_action"),
        "contraindications": _collect_texts(labels, "contraindications"),
        "adverse_effects": {
            f: _collect_texts(labels, f) for f in ADVERSE_FIELDS
        },
        "_fallback": {
            f: _collect_texts(labels, f) for f in FALLBACK_FIELDS
        },
    }

    elapsed = time.perf_counter() - t_start
    log("ok", "openfda.label",
        ingredient=ingredient,
        fetched=n_fetched,
        single_ingredient=n_single,
        used=len(labels),
        requests=client.request_count - req0,
        indications=int(bool(drug_info["indications"])),
        moa=int(bool(drug_info["mechanism_of_action"])),
        contra=int(bool(drug_info["contraindications"])),
        adverse=[f for f in ADVERSE_FIELDS if drug_info["adverse_effects"][f]],
        http=secs(client.request_time - http0),
        time=secs(elapsed))

    return {
        "query": ingredient,
        "found": True,
        "not_marketed_us": False,
        "label_count": total,
        "labels_fetched": n_fetched,
        "labels_used": len(labels),
        "single_ingredient_labels": n_single,
        "drug_info": drug_info,
        "retrieved_at": now,
        "elapsed_sec": round(elapsed, 3),
    }


def collect(ingredients: list[str], max_labels: int = 1000, min_interval: float = 0.3,
            single_ingredient: bool = True) -> list[dict]:
    client = OpenFDAClient(min_interval=min_interval)
    with span("openfda", n_ingredients=len(ingredients)) as end:
        if not API_KEY:
            log("warn", "openfda", level="WARN", reason="OPENFDA_API_KEY not set")
        out = [fetch(client, ing, max_labels=max_labels, single_ingredient=single_ingredient) for ing in ingredients]
        end.update(found=sum(r["found"] for r in out),
                   requests=client.request_count,
                   http=secs(client.request_time))
    return out