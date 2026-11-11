import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime


def parse_json_dict(text):
    clean_text = text.strip().replace("```json", "").replace("```", "")
    try:
        data = json.loads(clean_text)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Failed to parse JSON as dict: {text!r}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}: {text!r}")
    return data


# Log line: HH:MM:SS.mmm LEVEL iter [scope] status k=v ...
#   start / end : a phase. Every end has a matching start with the same scope,
#                 and end.time is the wall time from start to end.
#   ok          : one step finished. time covers that step only; its child
#                 lines (db / llm) are printed just before it.
#   wait / skip / warn / error : progress, cache hit, recoverable issue, failure.
# Lines starting with "#" are separators and can be skipped by parsers.


class Tee:
    """Writes stdout to both the console and a file."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


@contextmanager
def tee_stdout(path):
    """with tee_stdout("run.log"): ... -> everything printed inside also goes to the file."""
    tee = Tee(path)
    sys.stdout = tee
    try:
        yield
    finally:
        tee.close()


_spans = {}
_iter = None


def secs(x):
    return f"{x:.3f}s"


def _fmt(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple, set)):
        return "[" + ",".join(_fmt(x) for x in v) + "]"
    s = str(v)
    if s == "" or any(c.isspace() or c in '"=[],' for c in s):
        s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "")
        return f'"{s}"'
    return s


def log(text="", stage="", level="INFO", **kv):
    if stage == "llm" and text == "ok":
        for sp in _spans.values():
            if sp["llm"] is not None:
                sp["llm"]["llm_calls"] += 1
                sp["llm"]["prompt"] += int(kv.get("prompt", 0))
                sp["llm"]["completion"] += int(kv.get("completion", 0))

    now = datetime.now()
    ts = now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"
    it = f"i{_iter}" if _iter is not None else "--"
    tag = (f"[{stage}]" if stage else "").ljust(20)
    extra = " ".join(f"{k}={_fmt(v)}" for k, v in kv.items())
    print(f"{ts} {level:<5} {it:<3} {tag} {text:<6} {extra}".rstrip())


def span_start(scope, llm=False, n=None, **kv):
    """Open a phase. llm=True adds llm_calls/tokens to its end line; n sets the iter column."""
    global _iter
    if n is not None:
        print()
        print(f"# {'─' * 31} {scope} {n} {'─' * 31}")
        _iter = n
    _spans[scope] = {
        "t0": time.perf_counter(),
        "llm": {"llm_calls": 0, "prompt": 0, "completion": 0} if llm else None,
    }
    log("start", scope, **kv)


def span_end(scope, level="INFO", **kv):
    global _iter
    if scope == "run":
        # close phases left open by an exception so every start still has an end
        for open_scope in [k for k in reversed(list(_spans)) if k != "run"]:
            span_end(open_scope, level="ERROR", error="aborted")
    sp = _spans.pop(scope)
    if sp["llm"] is not None:
        st = sp["llm"]
        kv.update(llm_calls=st["llm_calls"], prompt=st["prompt"], completion=st["completion"],
                  tokens=st["prompt"] + st["completion"])
    kv["time"] = secs(time.perf_counter() - sp["t0"])
    log("end", scope, level, **kv)
    if scope == "iter":
        _iter = None
    if scope == "run":
        _iter = None
        _spans.clear()


@contextmanager
def span(scope, llm=False, n=None, **kv):
    """with span(...) as end: ...; end.update(...) -> fields for the end line."""
    span_start(scope, llm=llm, n=n, **kv)
    end = {}
    try:
        yield end
    except Exception as e:
        span_end(scope, level="ERROR", **end, error=f"{type(e).__name__}: {e}")
        raise
    span_end(scope, **end)

    
# def get_descendant_concept_ids(db, concept_ids, batch_size: int = 1000):
#     if not concept_ids:
#         return []

#     safe_ids = [int(cid) for cid in concept_ids]
#     result: set[int] = set() 

#     for i in range(0, len(safe_ids), batch_size):
#         chunk = safe_ids[i : i + batch_size]
#         id_list = ", ".join(str(cid) for cid in chunk)

#         sql = f"""
#         SELECT DISTINCT c.concept_id
#         FROM [MIMIC3_CDM].[dbo].[concept_ancestor] ca
#         JOIN [MIMIC3_CDM].[dbo].[concept] c
#             ON ca.descendant_concept_id = c.concept_id
#         WHERE ca.ancestor_concept_id IN ({id_list})
#             AND c.standard_concept = 'S'
#             AND c.invalid_reason = 'n'
#         """

#         success, rows, error = db.query(sql)
#         if not success:
#             print(f"# Error (batch {i // batch_size}): {error}")
#             continue

#         for r in rows:
#             result.add(r["concept_id"]) 

#     ids = list(result)
#     return ids