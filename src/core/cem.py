import csv
import time
from datetime import datetime
from core.utils import log, secs, span
import requests


def get_negative_controls(
    concept_ids,
    base_url="https://atlas-demo.ohdsi.org/WebAPI",
    source_key="CEM",
    name_prefix="NC",
    domain="Drug",
    outcome="condition",
    token=None,
    max_wait=300,
    poll_interval=1,
    log_every=10,
    request_timeout=180,
    retry_interval=1800,
    max_rounds=None,
    save_path="cem_result.csv",
):
    """Create a concept set, request CEM negative controls, and poll for the result.

    A round that yields no rows within max_wait seconds (or fails on a request) is abandoned;
    after retry_interval seconds a new round starts again from creating the concept set.
    max_rounds=None keeps retrying until a round succeeds.
    """
    with span("cem", ingredients=list(concept_ids)) as end:
        session = requests.Session()
        session.headers.update({"Accept": "application/json"})
        if token:
            session.headers["Authorization"] = token

        def req(method, path, **kw):
            r = session.request(method, f"{base_url}/{path.lstrip('/')}", timeout=request_timeout, **kw)
            r.raise_for_status()
            return r.json() if r.text.strip() else None

        def run_round(round_no):
            name = f"{name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            t0 = time.perf_counter()
            cs_id = req("POST", "/conceptset/", json={"name": name, "description": ""})["id"]
            log("ok", "cem.create_set", round=round_no, name=name, set_id=cs_id, time=secs(time.perf_counter() - t0))

            t0 = time.perf_counter()
            req("PUT", f"/conceptset/{cs_id}/items", json=[
                {"conceptSetId": cs_id, "conceptId": int(c),
                 "isExcluded": 0, "includeDescendants": 1, "includeMapped": 0}
                for c in concept_ids
            ])
            log("ok", "cem.add_concepts", set_id=cs_id, count=len(concept_ids), time=secs(time.perf_counter() - t0))

            t0 = time.perf_counter()
            req("POST", f"/evidence/{source_key}/negativecontrols", json={
                "jobName": f"nc_{cs_id}",
                "conceptSetId": cs_id,
                "conceptSetName": name,
                "sourceKey": source_key,
                "conceptsOfInterest": [str(c) for c in concept_ids],
                "conceptDomainId": domain,
                "outcomeOfInterest": outcome,
                "csToInclude": 0, "csToExclude": 0,
                "csToIncludeSQL": "", "csToExcludeSQL": "",
            })
            log("ok", "cem.request", set_id=cs_id, time=secs(time.perf_counter() - t0))

            t0 = time.perf_counter()
            attempt = 0
            poll_errors = 0
            next_log = log_every
            while True:
                attempt += 1
                try:
                    rows = req("GET", f"/evidence/{source_key}/negativecontrols/{cs_id}") or []
                except requests.RequestException:
                    rows = []          # a failed poll is just another empty poll until max_wait
                    poll_errors += 1
                elapsed = time.perf_counter() - t0
                if rows:
                    break
                if elapsed >= max_wait:
                    raise TimeoutError(f"no results after {elapsed:.1f}s ({attempt} polls, {poll_errors} failed)")
                if elapsed >= next_log:
                    log("wait", "cem.poll", attempts=attempt, errors=poll_errors, elapsed=secs(elapsed))
                    next_log += log_every
                time.sleep(poll_interval)

            log("ok", "cem.poll", attempts=attempt, errors=poll_errors, rows=len(rows), time=secs(elapsed))
            return cs_id, list(rows)

        round_no = 0
        while True:
            round_no += 1
            try:
                cs_id, rows = run_round(round_no)
                break
            except (TimeoutError, requests.RequestException) as e:
                if max_rounds is not None and round_no >= max_rounds:
                    log("error", "cem.round", level="ERROR", round=f"{round_no}/{max_rounds}",
                        error=f"{type(e).__name__}: {e}")
                    raise
                log("warn", "cem.round", level="WARN", round=round_no, error=f"{type(e).__name__}: {e}",
                    retry_in=secs(retry_interval))
                time.sleep(retry_interval)

        if rows:
            fieldnames = list(rows[0].keys())
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        result = {
            int(row["conceptId"]): (row["negativeControl"] == 1 or row["optimizedOut"] == 1)
            for row in rows
        }

        result_detail = {
            int(row["conceptId"]): row
            for row in rows
        }

        end.update(set_id=cs_id, rounds=round_no, total=len(result), recommended=sum(result.values()), path=save_path)

    return result, result_detail
