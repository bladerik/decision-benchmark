#!/usr/bin/env python3
"""Benchmark OpenRouter decision models on a labelled sample. Standard library only.

  bench.py models                       live decision models, prices, open weights, your credit
  bench.py plan   QUESTIONS ITEMS [opts] check inputs, pick models, estimate cost (no requests)
  bench.py run    QUESTIONS ITEMS --max-usd 0.50 [opts]   run, then write results + report
  bench.py report RESULTS.json [--out report.md]          rebuild the report, no requests

File formats: references/formats.md. Key: env var OPENROUTER_API_KEY.
"""
import argparse, base64, csv, hashlib, json, math, os, re, statistics, struct, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://openrouter.ai/api"
MODELS_URL = API + "/v1/models?output_modalities=decisions"
IMAGE_TOKENS_GUESS = 1600  # rough upper estimate per image, for cost estimates
THRESHOLDS = (0.8, 0.9)
MIN_KEPT = 20  # a suggested threshold must keep at least this many answers
ATTEMPTS = 4
RETRY_CODES = (408, 429, 500, 502, 503, 504)
SAVE_EVERY_S = 1.0  # checkpoint at most this often while running; always on exit
SAFETY = 2.0  # budget reservation = SAFETY x the rough estimate of one request
BOX_JEV = (1, 14)  # first Jev version that answers box questions

# Provider limits from https://openrouter.ai/docs/guides/community/multimodal-decisions#limits
# (the model list does not expose them). Matched by id prefix after stripping "~" and ":suffix".
IMAGE_LIMITS = [
    ("cloudflare/clef", {"max_images": 4}),
    ("liquid/", {"max_images": 8, "max_body": 4_500_000}),
    ("morph/", {"max_images": 16}),
    ("inception/", {"max_images": 16}),
    ("baseten/", {"max_images": 16}),
    ("scaledown/", {"max_images": 1}),
    ("openai/gpt-6-luna-decisions", {"max_images": 128}),
    ("typesafe/jev", {"min_pixels": 78_400, "max_pixels": 2_458_624, "max_file": 16 * 1024 * 1024}),
]

_now = time.monotonic
_sleep = time.sleep


# ---------- HTTP ----------

def key():
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not k:
        sys.exit("OPENROUTER_API_KEY is not set. Export it in the terminal that starts the agent, then retry.")
    if any(ch.isspace() or ord(ch) < 32 for ch in k):
        sys.exit("OPENROUTER_API_KEY contains a space or line break. Copy the key again and re-export it, then retry.")
    return k


def redact(text):
    """Remove the API key (and anything shaped like one) from text that gets printed or saved."""
    text = str(text)
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    for part in [k] + k.split():
        if len(part) >= 8:
            text = text.replace(part, "[redacted]")
    text = re.sub(r"sk-or-[A-Za-z0-9_\-]+", "[redacted]", text)
    return re.sub(r"(Bearer\s+)[^\s'\"]+", r"\1[redacted]", text)


def http(method, url, body=None, auth=True, timeout=180):
    headers = {"Content-Type": "application/json", "X-Title": "decision-benchmark"}
    if auth:
        headers["Authorization"] = "Bearer " + key()
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def post_with_retry(url, body):
    """Returns (response, error, seconds). Retries 408/429/5xx with backoff; seconds cover every attempt and wait."""
    err, t0 = None, _now()
    for attempt in range(ATTEMPTS):
        try:
            return http("POST", url, body), None, _now() - t0
        except urllib.error.HTTPError as e:
            try:
                detail = e.read()[:300].decode(errors="replace")
            except Exception:
                detail = ""
            err = redact(f"HTTP {e.code}: {detail}")[:300]
            if e.code not in RETRY_CODES:
                break
        except Exception as e:  # network, timeout, bad JSON
            err = redact(repr(e))[:300]
        if attempt < ATTEMPTS - 1:
            _sleep(2 * 2 ** attempt)
    return None, err, _now() - t0


def live_models():
    return http("GET", MODELS_URL, auth=False)["data"]


def key_note():
    """One line about the key's own spending limit (GET /api/v1/key), so a runaway loop cannot drain the account."""
    try:
        d = http("GET", API + "/v1/key")["data"]
    except Exception:
        return "Key limit: unknown (could not read it). Check it at https://openrouter.ai/settings/keys."
    if d.get("limit") is None:
        return ("Warning: this key has no credit limit. Create a key for this benchmark with a limit of a few dollars "
                "that resets daily at https://openrouter.ai/settings/keys.")
    reset = f" (resets {d['limit_reset']})" if d.get("limit_reset") else ""
    left = d.get("limit_remaining")
    return f"Key limit: ${float(d['limit']):.2f}{reset}" + ("" if left is None else f", ${float(left):.2f} left.")


def credit():
    try:
        d = http("GET", API + "/v1/credits")["data"]
        return d["total_credits"] - d["total_usage"]
    except Exception:
        return None


# ---------- inputs ----------

def image_size(path):
    with open(path, "rb") as f:
        head = f.read(26)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", head[16:24])
        if head[:2] != b"\xff\xd8":
            return None
        f.seek(2)
        while True:
            b = f.read(1)
            while b and b != b"\xff":
                b = f.read(1)
            while b == b"\xff":
                b = f.read(1)
            if not b:
                return None
            if b[0] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                f.read(3)
                h, w = struct.unpack(">HH", f.read(4))
                return w, h
            f.seek(struct.unpack(">H", f.read(2))[0] - 2, 1)


def load_items(path, questions):
    base = os.path.dirname(os.path.abspath(path))
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        items = []
        for r in rows:
            it = {"id": (r.get("id") or "").strip() or str(len(items) + 1), "text": r.get("text") or "",
                  "images": [p for p in (r.get("image") or "").split("|") if p.strip()],
                  "expected": {q: r[q] for q in questions if (r.get(q) or "").strip() != ""}}
            items.append(it)
    else:
        items = []
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                imgs = r.get("images") or ([r["image"]] if r.get("image") else [])
                items.append({"id": str(r.get("id", n)).strip(), "text": r.get("text", ""), "state": r.get("state"),
                              "images": imgs, "expected": r.get("expected", {})})
    for it in items:
        it["images"] = [p if os.path.isabs(p) else os.path.join(base, p.strip()) for p in it["images"]]
    return items


def _number(x):
    if isinstance(x, bool) or not isinstance(x, (int, float, str)):
        raise ValueError(f"{x!r} is not a number")
    f = float(x)
    if not math.isfinite(f):
        raise ValueError(f"{x!r} is not a finite number")
    return f


def parse_expected(q, v):
    t = q["type"]
    if t == "noul":
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ("1", "true", "yes", "y", "ano", "áno", "a"):
            return True
        if s in ("0", "false", "no", "n", "nie", "ne"):
            return False
        raise ValueError(f"noul expected must be yes/no, got {v!r}")
    if t == "choice":
        if str(v) not in q["criteria"]:
            raise ValueError(f"choice expected {v!r} is not one of {list(q['criteria'])}")
        return str(v)
    if t == "score":
        f = _number(v)
        if not f.is_integer() or not 0 <= f < len(q["criteria"]):
            raise ValueError(f"score expected {v!r} must be a whole level index 0..{len(q['criteria']) - 1}")
        return int(f)
    if t == "box":
        b = json.loads(v) if isinstance(v, str) else v
        if isinstance(b, dict):
            b = [b["top"], b["bottom"], b["left"], b["right"]]
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            raise ValueError(f"box expected must be [top, bottom, left, right], got {v!r}")
        b = [_number(x) for x in b]
        if not all(0 <= x <= 1 for x in b) or not (b[0] < b[1] and b[2] < b[3]):
            raise ValueError(f"box expected {v!r} needs fractions 0..1 with top < bottom and left < right")
        return b
    raise ValueError(f"unknown question type {t!r}")


def validate(questions, items):
    problems, warnings = [], []
    for k, q in questions.items():
        t = q.get("type")
        if t not in ("noul", "choice", "score", "box"):
            problems.append(f"question {k}: type must be noul, choice, score or box")
        if t == "choice" and not isinstance(q.get("criteria"), dict):
            problems.append(f"question {k}: choice needs `criteria` as an object {{option: description}} (not `options`)")
        if t == "score" and not isinstance(q.get("criteria"), list):
            problems.append(f"question {k}: score needs `criteria` as a list of levels, lowest first")
    if problems:
        return problems, warnings
    seen = {}
    for it in items:
        seen[it["id"]] = seen.get(it["id"], 0) + 1
    for i, n in seen.items():
        if n > 1:
            problems.append(f"item {i}: duplicate id ({n} items); every item needs its own id")
    labelled = {k: 0 for k in questions}
    for it in items:
        for k, v in list(it["expected"].items()):
            if k not in questions:
                problems.append(f"item {it['id']}: expected answer for unknown question {k!r}")
                continue
            try:
                it["expected"][k] = parse_expected(questions[k], v)
                labelled[k] += 1
            except (ValueError, KeyError, TypeError) as e:
                problems.append(f"item {it['id']}, question {k}: {e}")
        for p in it["images"]:
            if not os.path.exists(p):
                problems.append(f"item {it['id']}: image not found: {p}")
                continue
            if not p.lower().endswith((".png", ".jpg", ".jpeg")):
                problems.append(f"item {it['id']}: {os.path.basename(p)} must be PNG or JPEG")
            wh = image_size(p)
            if wh and (min(wh) < 256 or max(wh) > 1568):
                problems.append(f"item {it['id']}: {os.path.basename(p)} is {wh[0]}x{wh[1]}; resize so each side is 256-1568 px")
            if os.path.getsize(p) > 300_000:
                warnings.append(f"item {it['id']}: {os.path.basename(p)} is {os.path.getsize(p) // 1000} KB; Cloudflare models reject images over ~300 KB, recompress (JPEG ~85) to include them")
        st = it.get("state")
        if st is not None and not isinstance(st, dict):
            problems.append(f"item {it['id']}: state must be a JSON object")
        elif st and it["text"] and "text" in st:
            warnings.append(f"item {it['id']}: state has its own `text` field; the item's `text` replaces it")
        if not it["text"] and not it["images"] and not st:
            problems.append(f"item {it['id']}: has no text, state or image")
    for k, n in labelled.items():
        if n == 0:
            warnings.append(f"question {k}: no item has an expected answer, so it is asked but not scored")
        elif n < 30:
            warnings.append(f"question {k}: only {n} labelled items; accuracy differences of a few points are noise")
    return problems, warnings


def data_url(path):
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def build_state(it):
    """Plain object when there are no images (state fields plus `text`); a list of text + image parts when there are."""
    st0 = it.get("state")
    if not it["images"]:
        if st0 is None:
            return {"text": it["text"]}
        return {**st0, "text": it["text"]} if it["text"] else st0
    st = []
    if st0 is not None:
        st.append(json.dumps(st0, ensure_ascii=False))
    if it["text"]:
        st.append("text: " + it["text"])
    for n, p in enumerate(it["images"], 1):
        st.append(f"Image {n} (`image{n if len(it['images']) > 1 else ''}`):")
        st.append({"type": "image_url", "image_url": {"url": data_url(p)}})
    return st


def fingerprint(questions, items):
    """Hash of the questions and every item's content (images by file hash); labels are left out."""
    h = hashlib.sha256()
    h.update(json.dumps(questions, ensure_ascii=False).encode())  # key order matters: option order changes answers
    for it in items:
        imgs = []
        for p in it["images"]:
            with open(p, "rb") as f:
                imgs.append(hashlib.sha256(f.read()).hexdigest())
        h.update(json.dumps([it["id"], it["text"], it.get("state"), imgs], ensure_ascii=False).encode())
    return h.hexdigest()


# ---------- model selection and cost ----------

def base_id(mid):
    return mid.lstrip("~").split(":", 1)[0]


def answers_box(model, live=None):
    """Jev 1.14 and newer answer box questions; an alias counts when the live list resolves it to such a Jev."""
    if live and model in live:
        target = (live[model].get("alias_target") or {}).get("slug")
        if target:
            model = target
    m = re.fullmatch(r"typesafe/jev-(\d+)\.(\d+)", base_id(model))
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= BOX_JEV


def model_questions(mid, questions, live=None, llm=False):
    """The questions this model gets: box only for Jev 1.14+, never for the LLM baseline."""
    box_ok = not llm and answers_box(mid, live)
    return {k: q for k, q in questions.items() if q["type"] != "box" or box_ok}


def limits_for(mid):
    b = base_id(mid)
    return next((lim for prefix, lim in IMAGE_LIMITS if b.startswith(prefix)), {})


def check_limits(mid, items):
    """Reason this model cannot take some item's images, or None."""
    lim = limits_for(mid)
    for it in items:
        if lim.get("max_images") and len(it["images"]) > lim["max_images"]:
            return f"takes at most {lim['max_images']} images, item {it['id']} has {len(it['images'])}"
        body = 0
        for p in it["images"]:
            wh = image_size(p)
            if wh and lim.get("min_pixels") and wh[0] * wh[1] < lim["min_pixels"]:
                return f"needs at least {lim['min_pixels']:,} pixels per image, {os.path.basename(p)} has {wh[0] * wh[1]:,}"
            if wh and lim.get("max_pixels") and wh[0] * wh[1] > lim["max_pixels"]:
                return f"takes at most {lim['max_pixels']:,} pixels per image, {os.path.basename(p)} has {wh[0] * wh[1]:,}"
            size = os.path.getsize(p)
            if lim.get("max_file") and size > lim["max_file"]:
                return f"takes images up to {lim['max_file'] // 2 ** 20} MiB, {os.path.basename(p)} is larger"
            body += size * 4 // 3
        if lim.get("max_body") and body > lim["max_body"]:
            return f"takes requests up to {lim['max_body'] / 1e6:.1f} MB, item {it['id']} has ~{body / 1e6:.1f} MB of images"
    return None


def pick_models(questions, items, only=None, exclude=(), include_aliases=False):
    live = {m["id"]: m for m in live_models()}
    needs_image = any(it["images"] for it in items)
    chosen, skipped = [], []
    ids = only or list(live)
    for mid in ids:
        m = live.get(mid)
        if m is None:
            skipped.append((mid, "not in the live decision-model list"))
            continue
        if not only and (mid.startswith("~") and not include_aliases):
            skipped.append((mid, "alias of a listed version"))
            continue
        if mid.endswith(":free") and mid[:-5] in live and not only:
            skipped.append((mid, "free twin of a listed model"))
            continue
        if mid in exclude:
            skipped.append((mid, "excluded"))
            continue
        if needs_image and "image" not in m["architecture"]["input_modalities"]:
            skipped.append((mid, "text only, your items have images"))
            continue
        if not model_questions(mid, questions, live):
            skipped.append((mid, "no question it can answer (box needs Jev 1.14+)"))
            continue
        why = check_limits(mid, items) if needs_image else None
        if why:
            skipped.append((mid, why))
            continue
        chosen.append(m)
    return chosen, skipped, live


def est_tokens(questions, it):
    text = len(json.dumps(questions, ensure_ascii=False)) + len(it["text"]) + len(json.dumps(it.get("state") or ""))
    return text / 4 + IMAGE_TOKENS_GUESS * len(it["images"])


def baseline_max_tokens(questions):
    return 256 + 64 * len(questions)


def rough_cost(model, questions, it, llm=False):
    """Rough cost of one request from list prices (no safety margin)."""
    p = model.get("pricing") or {}
    f = lambda k: max(0.0, float(p.get(k) or 0))  # -1 means variable/unknown; check_baseline rejects it
    usd = est_tokens(questions, it) * (1.3 if llm else 1) * f("prompt") + f("request") + f("image") * len(it["images"])
    if llm:
        usd += baseline_max_tokens(questions) * f("completion")
    return usd


def request_estimate(model, questions, it, llm=False):
    """Conservative per-request amount reserved against --max-usd before the request starts."""
    return SAFETY * rough_cost(model, questions, it, llm)


def estimate(models, baseline, questions, items, passes, live=None):
    rows = [(m["id"], passes * sum(rough_cost(m, model_questions(m["id"], questions, live), it) for it in items)) for m in models]
    if baseline:
        bq = model_questions(baseline["id"], questions, llm=True)
        rows.append((baseline["id"] + " (LLM)", passes * sum(rough_cost(baseline, bq, it, llm=True) for it in items)))
    return rows


# ---------- calls ----------

def _desc(q):
    d = q.get("instructions")
    d = d if isinstance(d, str) else (json.dumps(d, ensure_ascii=False) if d is not None else "")
    if q["type"] == "noul" and isinstance(q.get("criteria"), dict):
        c = q["criteria"]
        if c.get("true"):
            d += f" True when: {c['true']}"
        if c.get("false"):
            d += f" False when: {c['false']}"
    return d.strip()


def llm_schema(questions):
    props = {}
    for k, q in questions.items():
        desc = _desc(q)
        if q["type"] == "noul":
            props[k] = {"type": "boolean", "description": desc}
        elif q["type"] == "choice":
            props[k] = {"type": "string", "enum": list(q["criteria"]),
                        "description": desc + " Options: " + json.dumps(q["criteria"], ensure_ascii=False)}
        elif q["type"] == "score":
            props[k] = {"type": "integer", "enum": list(range(len(q["criteria"]))),
                        "description": desc + " Levels (index: meaning): " + json.dumps(dict(enumerate(q["criteria"])), ensure_ascii=False)}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _cost(r):
    c = (r.get("usage") or {}).get("cost") if isinstance(r, dict) else None
    return float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) else None


def call_llm(model, questions, it):
    """Returns (answers, cost, error, seconds, responded)."""
    st = build_state(it)
    st = st if isinstance(st, list) else [json.dumps(st, ensure_ascii=False)]
    images = [x for x in st if isinstance(x, dict)]
    text = "\n".join(x for x in st if isinstance(x, str))
    parts = [{"type": "text", "text": "Answer every field strictly and factually from this content.\n" + text}] + images
    body = {"model": model, "messages": [{"role": "user", "content": parts}], "usage": {"include": True},
            "max_tokens": baseline_max_tokens(questions),
            "response_format": {"type": "json_schema", "json_schema": {"name": "decisions", "strict": True, "schema": llm_schema(questions)}}}
    r, err, secs = post_with_retry(API + "/v1/chat/completions", body)
    if err:
        return None, None, err, secs, False
    try:
        raw = json.loads(r["choices"][0]["message"]["content"])
        if not isinstance(raw, dict):
            raise ValueError("not a JSON object")
    except Exception as e:
        return None, _cost(r), redact(f"unparseable LLM output: {e!r}")[:300], secs, True
    ans = {}
    for k, q in questions.items():
        if k not in raw:
            continue
        if q["type"] == "noul" and isinstance(raw[k], bool):
            ans[k] = {"type": "noul", "noul": 1.0 if raw[k] else 0.0}
        elif q["type"] == "choice":
            ans[k] = {"type": "choice", "choice": raw[k]}
        elif q["type"] == "score":
            ans[k] = {"type": "score", "score": raw[k]}
    return ans, _cost(r), None, secs, True


def call_decision(model, questions, it):
    """Returns (answers, cost, error, seconds, responded). `questions` is already this model's set."""
    r, err, secs = post_with_retry(API + "/alpha/decisions", {"model": model, "state": build_state(it), "questions": questions})
    if err:
        return None, None, err, secs, False
    if not isinstance(r, dict):
        return None, None, "response is not a JSON object", secs, True
    return r.get("answers"), _cost(r), None, secs, True


# ---------- scoring ----------

def iou(a, b):
    t, bo, l, r = a
    t2, b2, l2, r2 = b
    inter = max(0, min(bo, b2) - max(t, t2)) * max(0, min(r, r2) - max(l, l2))
    union = (bo - t) * (r - l) + (b2 - t2) * (r2 - l2) - inter
    return inter / union if union > 0 else 0


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _modal_level(probs):
    if isinstance(probs, dict):
        try:
            pairs = [(int(k), v) for k, v in probs.items()]
        except (TypeError, ValueError):
            return None
    elif isinstance(probs, list):
        pairs = list(enumerate(probs))
    else:
        return None
    pairs = [(k, v) for k, v in pairs if _num(v)]
    return max(pairs, key=lambda kv: kv[1]) if pairs else None


def decide(a, qtype=None):
    """Returns (decision, confidence) for one answer; (None, None) for a missing, malformed or refused answer."""
    if not isinstance(a, dict) or a.get("refused") or a.get("refusal"):
        return None, None
    t = a.get("type", qtype)
    if qtype and t != qtype:
        return None, None
    conf = a.get("confidence") if _num(a.get("confidence")) else None
    if t == "noul":
        v = a.get("noul")
        if not _num(v) or not 0 <= v <= 1:
            return None, None
        return v >= 0.5, conf if conf is not None else max(v, 1 - v)
    if t == "choice":
        c = a.get("choice")
        if not isinstance(c, str):
            return None, None
        p = (a.get("probabilities") or {}).get(c) if isinstance(a.get("probabilities"), dict) else None
        return c, conf if conf is not None else (p if _num(p) else None)
    if t == "score":
        modal = _modal_level(a.get("probabilities"))
        if modal:
            return modal[0], conf if conf is not None else modal[1]
        s = a.get("score")
        if not _num(s):
            return None, None
        return int(math.floor(s + 0.5)), conf
    if t == "box":
        b = a.get("box")
        if not isinstance(b, dict) or not all(_num(b.get(k)) for k in ("top", "bottom", "left", "right")):
            return None, None
        return (b["top"], b["bottom"], b["left"], b["right"]), conf
    return None, None


def valid_decision(d, q):
    """A decision the question allows: a listed option, a level on the scale, a box inside the image."""
    if q["type"] == "choice":
        return d in (q.get("criteria") or {})
    if q["type"] == "score":
        return isinstance(d, int) and 0 <= d < len(q.get("criteria") or [])
    if q["type"] == "box":
        return all(0 <= x <= 1 for x in d) and d[0] < d[1] and d[2] < d[3]
    return True


def correct(decision, expected):
    if decision is None:
        return False
    if isinstance(expected, list):  # box: overlap of at least half counts as right
        return isinstance(decision, tuple) and iou(decision, expected) >= 0.5
    return decision == expected


def same(a, b):
    if isinstance(a, tuple) and isinstance(b, tuple):
        return iou(a, b) >= 0.5
    return a == b


def charge(c):
    """What a call cost: the reported cost, else the estimate counted for it, else nothing known."""
    if c.get("charge") is not None:
        return c["charge"]
    return c.get("cost") or 0


def final_calls(calls):
    """One call per (model, item, pass): a successful one wins, otherwise the latest attempt."""
    last = {}
    for c in calls:
        k = (c["model"], c["item"], c["pass"])
        if k not in last or not c.get("error") or last[k].get("error"):
            last[k] = c
    return last


def asked(res, mid):
    """Question keys this model was asked (stored by `run`; derived for results.json from older versions)."""
    meta = res["models"].get(mid, {})
    if meta.get("questions") is not None:
        return [k for k in meta["questions"] if k in res["questions"]]
    return list(model_questions(mid, res["questions"], llm=bool(meta.get("llm"))))


def summarise(res):
    questions, items = res["questions"], {it["id"]: it for it in res["items"]}
    finals = final_calls(res["calls"])
    by_model = {mid: [] for mid in res["models"]}
    for (mid, _, _), c in finals.items():
        by_model.setdefault(mid, []).append(c)
    asks = {mid: asked(res, mid) for mid in by_model}
    common = common_questions(res)
    rows = []
    for mid, calls in by_model.items():
        meta = res["models"].get(mid, {})
        is_llm = bool(meta.get("llm"))
        ok_calls = [c for c in calls if not c.get("error")]
        judged, decisions, perq, invalid = [], {}, {}, []
        for c in calls:
            it = items.get(c["item"])
            if it is None:
                continue
            ans = c.get("answers") if not c.get("error") and isinstance(c.get("answers"), dict) else {}
            for k in asks[mid]:
                d, conf = decide(ans.get(k), questions[k]["type"])
                if d is not None and not valid_decision(d, questions[k]):
                    d, conf = None, None
                if d is None and not c.get("error"):
                    invalid.append(f"{c['item']} / {k}" + (f" (pass {c['pass']})" if res.get("passes", 1) > 1 else "")
                                   + (": missing" if k not in ans else ": invalid"))
                if d is not None:
                    decisions[(c["item"], k, c["pass"])] = d
                exp = it["expected"].get(k)
                if exp is None:
                    continue
                ok = correct(d, exp)
                perq.setdefault(k, []).append(ok)
                if k in common:
                    judged.append((ok, None if is_llm or d is None else conf))
        pairs = [(v, decisions[(i, k, 2)]) for (i, k, p), v in decisions.items() if p == 1 and (i, k, 2) in decisions]
        costs = [c["cost"] for c in ok_calls if _num(c.get("cost"))]
        all_calls = [c for c in res["calls"] if c["model"] == mid]
        row = {"model": mid, "llm": is_llm, "n": len(judged), "calls": len(calls),
               "planned": meta.get("planned", len(calls)), "questions": asks[mid],
               "errors": len(calls) - len(ok_calls), "error_sample": next((c["error"] for c in calls if c.get("error")), None),
               "acc": sum(o for o, _ in judged) / len(judged) if judged else None,
               "median_s": statistics.median(c["secs"] for c in ok_calls) if ok_calls else None,
               "usd_per_1k": 1000 * statistics.mean(costs) if costs else None,
               "stable": sum(same(a, b) for a, b in pairs) / len(pairs) if pairs else None,
               "hf": meta.get("hugging_face_id"), "perq": {k: (sum(v), len(v)) for k, v in perq.items()},
               "spent": sum(charge(c) for c in all_calls),
               "estimated": sum(1 for c in all_calls if c.get("cost_estimated")), "invalid": invalid}
        has_conf = any(conf is not None for _, conf in judged)
        for t in THRESHOLDS:
            confident = [o for o, conf in judged if conf is not None and conf >= t]
            row[f"acc@{t}"] = (sum(confident) / len(confident) if confident else None) if has_conf else None
            row[f"cov@{t}"] = (len(confident) / len(judged) if judged else None) if has_conf else None
        row["auto"] = auto_threshold(judged, res.get("target", 0.95))
        rows.append(row)
    rows.sort(key=lambda r: (-(r["acc"] or 0), r["usd_per_1k"] or 9e9))
    return rows


def common_questions(res):
    """Questions every model was asked: the main ranking compares models on these only."""
    mids = set(res["models"]) | {c["model"] for c in res["calls"]}
    return [k for k in res["questions"] if all(k in asked(res, m) for m in mids)]


def auto_threshold(judged, target):
    """Lowest observed confidence whose kept answers reach the target accuracy with at least MIN_KEPT answers.
    `judged` holds (correct, confidence) for every labelled answer asked, None confidence = not covered.
    Returns (threshold, coverage of all judged answers, kept answers) or None."""
    with_conf = [(o, c) for o, c in judged if c is not None]
    for t in sorted({c for _, c in with_conf}):
        kept = [o for o, c in with_conf if c >= t]
        if len(kept) < MIN_KEPT:
            break
        if sum(kept) / len(kept) >= target:
            return t, len(kept) / len(judged), len(kept)
    return None


# ---------- report ----------

def pct(x):
    return "–" if x is None else f"{100 * x:.0f} %"


def hard_items(res):
    items = {it["id"]: it["expected"] for it in res["items"]}
    tally = {}
    for c in final_calls(res["calls"]).values():
        if c["item"] not in items or c.get("error"):
            continue
        ans = c.get("answers") if isinstance(c.get("answers"), dict) else {}
        for k in asked(res, c["model"]):
            exp = items[c["item"]].get(k)
            if exp is None:
                continue
            w, n, _ = tally.get((c["item"], k), (0, 0, exp))
            tally[(c["item"], k)] = (w + (not correct(decide(ans.get(k), res["questions"][k]["type"])[0], exp)), n + 1, exp)
    return sorted(((key, v) for key, v in tally.items() if v[1] and v[0] / v[1] >= 0.5), key=lambda x: -x[1][0] / x[1][1])


def num(x, fmt):
    return "–" if x is None else fmt.format(x)


def write_report(res, out):
    rows, common = summarise(res), common_questions(res)
    target = res.get("target", 0.95)
    done, planned = sum(r["calls"] for r in rows), sum(r["planned"] for r in rows)
    stopped = res.get("stopped")
    spent = sum(r["spent"] for r in rows) + res.get("spent_dropped", 0)
    L = ["# Decision model benchmark", ""]
    if stopped or done < planned:
        why = {"cap": "stopped at the --max-usd cap", "interrupted": "interrupted"}.get(stopped, "not every planned request ran")
        L += [f"> **Partial run: {why}. {done}/{planned} planned requests completed.** Accuracy below covers only the "
              "completed requests, and models may have stopped at different points. Rerun the same command (with a higher "
              "--max-usd if the cap stopped it) to finish before choosing a model.", ""]
    L += [f"{len(res['items'])} items × {len(res['questions'])} questions × {res['passes']} passes, run {res['started']}. "
          f"Total spent: ${spent:.4f}.", ""]
    estimated = sum(r["estimated"] for r in rows)
    if estimated:
        L += [f"{estimated} responses had no usage.cost; each is counted at its pre-request estimate in the totals.", ""]
    L += ["| Model | Accuracy | Acc ≥0.9 (coverage) | Acc ≥0.8 (coverage) | Median s | $ / 1 000 items | Same answer both passes | Open weights | Failed calls | Requests done |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        name = r["model"] + (" (LLM baseline)" if r["llm"] else "")
        ow = "–" if r["llm"] else (f"yes ([{r['hf']}](https://huggingface.co/{r['hf']}))" if r["hf"] else "no")
        L.append(f"| {name} | {pct(r['acc'])} | {pct(r['acc@0.9'])} ({pct(r['cov@0.9'])}) | {pct(r['acc@0.8'])} ({pct(r['cov@0.8'])}) | "
                 f"{num(r['median_s'], '{:.1f}')} | {num(r['usd_per_1k'], '${:.3f}')} | "
                 f"{pct(r['stable'])} | {ow} | {r['errors']}/{r['calls']} | {r['calls']}/{r['planned']} |")
    others = [k for k in res["questions"] if k not in common]
    L += ["", "Accuracy counts every labelled (item, question, pass) a model was asked; failed requests and missing or "
          "malformed answers count as wrong. "
          f"The main columns use only the questions every model was asked ({', '.join(common) or 'none'})"
          + (f"; {', '.join(others)} appear only in the per-question table" if others else "") + ". "
          "\"Acc ≥0.9 (coverage)\": accuracy on answers with confidence ≥ 0.9, and the share of all labelled answers "
          "that clear it; the rest would go to a fallback. One item = one request answering all its questions. "
          "LLM baselines return no confidence, so their threshold columns are empty.", "",
          f"## Threshold for ≥ {target * 100:.0f} % accuracy", "",
          f"Lowest confidence at which the kept answers reach the target, with at least {MIN_KEPT} kept answers. "
          "Tuned on this sample, so treat it as an estimate.", ""]
    for r in rows:
        if r["llm"] or r["acc"] is None:
            continue
        a = r["auto"]
        L.append(f"- {r['model']}: " + (f"confidence ≥ {float(a[0])!r} keeps {pct(a[1])} of answers ({a[2]} answers)" if a
                                        else f"no threshold reaches it with at least {MIN_KEPT} answers on this sample"))
    qs = list(res["questions"])
    L += ["", "## Accuracy per question", "", "Correct share of labelled answers per question (correct/asked). "
          "A question far below the rest needs better wording or a fallback of its own.", "",
          "| Model | " + " | ".join(qs) + " |", "|---|" + "---|" * len(qs)]
    for r in rows:
        cells = []
        for q in qs:
            if q not in r["questions"]:
                cells.append("not asked")
            elif q in r["perq"]:
                c, n = r["perq"][q]
                cells.append(f"{pct(c / n)} ({c}/{n})")
            else:
                cells.append("–")
        L.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    hard = hard_items(res)
    if hard:
        L += ["", "## Items most models got wrong", "",
              "Either the expected answer is wrong or the question can be read another way. Check these first.", ""]
        L += [f"- {i} / {k}: expected {json.dumps(res_exp)}, wrong in {w} of {n} model answers" for (i, k), (w, n, res_exp) in hard]
    errs = [r for r in rows if r["error_sample"] or r["invalid"]]
    if errs:
        L += ["", "## Errors", ""]
        for r in errs:
            if r["error_sample"]:
                L.append(f"- {r['model']}: {r['errors']} failed requests, e.g. {redact(r['error_sample'])}")
            if r["invalid"]:
                more = f" and {len(r['invalid']) - 5} more" if len(r["invalid"]) > 5 else ""
                L.append(f"- {r['model']}: {len(r['invalid'])} answers missing or invalid in otherwise successful requests "
                         f"(counted as wrong): {', '.join(r['invalid'][:5])}{more}")
    text = "\n".join(L) + "\n"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return text.rstrip("\n")


# ---------- commands ----------

def cmd_models(a):
    ms = live_models()
    print(f"{len(ms)} decision models live at {MODELS_URL}\n")
    print(f"{'model':45} {'$ / M input':>11}  {'inputs':18} open weights")
    for m in sorted(ms, key=lambda m: m["id"]):
        print(f"{m['id']:45} {float(m['pricing'].get('prompt') or 0) * 1e6:11.3f}  {'+'.join(m['architecture']['input_modalities']):18} {m.get('hugging_face_id') or 'no'}")
    if os.environ.get("OPENROUTER_API_KEY"):
        c = credit()
        print("\nOPENROUTER_API_KEY is set. Remaining credit: " + ("unknown" if c is None else f"${c:.2f}"))
        print(key_note())
    else:
        print("\nOPENROUTER_API_KEY is not set.")


def _price(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def check_baseline(m, items):
    bad = [k for k, v in (m.get("pricing") or {}).items() if _price(v) is not None and _price(v) < 0]
    if bad:
        return (f"baseline {m['id']} has no fixed price ({', '.join(bad)} is listed as variable), so --max-usd cannot "
                "budget it; pick a chat model with a listed price")
    params = set(m.get("supported_parameters") or [])
    if not params & {"structured_outputs", "response_format"}:
        return f"baseline {m['id']} does not list structured outputs (response_format); pick a chat model that supports them"
    if any(it["images"] for it in items) and "image" not in (m.get("architecture") or {}).get("input_modalities", []):
        return f"baseline {m['id']} takes no image input, and your items have images; pick a vision chat model"
    return None


def prepare(a):
    with open(a.questions, encoding="utf-8") as f:
        questions = json.load(f)
    items = load_items(a.items, questions)
    problems, warnings = validate(questions, items)
    for w in warnings:
        print("warning:", w)
    if problems:
        print("\n".join("problem: " + p for p in problems))
        sys.exit("Fix the problems above, then rerun.")
    only = [m.strip() for m in a.models.split(",")] if a.models else None
    models, skipped, live = pick_models(questions, items, only, set((a.exclude or "").split(",")), a.include_aliases)
    baseline = None
    if a.baseline:
        all_models = {m["id"]: m for m in http("GET", API + "/v1/models", auth=False)["data"]}
        if a.baseline not in all_models:
            sys.exit(f"baseline {a.baseline} not found on OpenRouter")
        baseline = all_models[a.baseline]
        why = check_baseline(baseline, items)
        if why:
            sys.exit(why)
        if not model_questions(baseline["id"], questions, llm=True):
            sys.exit("the baseline gets no questions (it cannot answer box); drop --baseline")
    if any(q["type"] == "box" for q in questions.values()):
        print("note: only Jev 1.14+ answers box questions; the other models get the remaining questions, "
              "and the main ranking uses only the questions every model gets")
    return questions, items, models, skipped, baseline, live


def print_plan(questions, items, models, skipped, baseline, passes, live=None):
    print(f"\n{len(items)} items, {len(questions)} questions, {passes} passes")
    for mid, why in skipped:
        print(f"  skip {mid}: {why}")
    rows = estimate(models, baseline, questions, items, passes, live)
    for mid, usd in rows:
        print(f"  run  {mid:45} ~${usd:.5f}")
    total = sum(u for _, u in rows)
    print(f"\n{len(rows)} models, {len(items) * passes * len(rows)} requests, estimated ${total:.3f} "
          f"(rough: images are guessed at {IMAGE_TOKENS_GUESS} tokens each; the run reserves {SAFETY:g}x this per request "
          f"against --max-usd and stops before it would pass the cap)")
    return total


def cmd_plan(a):
    q, items, models, skipped, baseline, live = prepare(a)
    print_plan(q, items, models, skipped, baseline, a.passes, live)
    if os.environ.get("OPENROUTER_API_KEY"):
        c = credit()
        print("Remaining credit: " + ("unknown" if c is None else f"${c:.2f}"))
        print(key_note())


def save_atomic(res, out):
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, out)


class Budget:
    """--max-usd ledger. A request starts only after its conservative estimate is reserved under the lock."""

    def __init__(self, cap, spent=0.0):
        self.cap, self.spent, self.reserved = cap, spent, 0.0
        self.lock = threading.Lock()

    def reserve(self, est):
        with self.lock:
            if self.spent + self.reserved + est > self.cap:
                return False
            self.reserved += est
            return True

    def settle(self, est, cost, responded):
        """Releases the reservation and books the charge: the reported cost, else the estimate when a response
        arrived without one, else 0 (no response, nothing billed). Returns (charge, estimated)."""
        with self.lock:
            self.reserved -= est
            estimated = cost is None and responded
            amount = cost if cost is not None else (est if responded else 0.0)
            self.spent += amount
            return amount, estimated


def cmd_run(a):
    key()  # a missing or broken key stops here, before any request
    questions, items, models, skipped, baseline, live = prepare(a)
    print_plan(questions, items, models, skipped, baseline, a.passes, live)
    print(key_note())
    out = a.out or "results.json"
    fp = fingerprint(questions, items)
    by_id = {m["id"]: m for m in models}
    asks = {m["id"]: model_questions(m["id"], questions, live) for m in models}
    if baseline:
        by_id[baseline["id"]] = baseline
        asks[baseline["id"]] = model_questions(baseline["id"], questions, llm=True)
    res = {"started": time.strftime("%Y-%m-%d %H:%M"), "passes": a.passes, "target": a.target, "fingerprint": fp,
           "questions": questions, "items": [{"id": it["id"], "expected": it["expected"]} for it in items], "calls": [],
           "models": {m["id"]: {"hugging_face_id": m.get("hugging_face_id"), "questions": list(asks[m["id"]]),
                                "planned": len(items) * a.passes} for m in models},
           "stopped": None, "spent_dropped": 0.0}
    if baseline:
        res["models"][baseline["id"]] = {"llm": True, "questions": list(asks[baseline["id"]]), "planned": len(items) * a.passes}
    keys = {(m, it["id"], p) for m in by_id for it in items for p in range(1, a.passes + 1)}
    done = set()
    if os.path.exists(out):  # resume: keep finished calls from an earlier run of the same inputs
        with open(out, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("fingerprint") != fp:
            sys.exit(f"{out} was made from different questions or items (or by an older bench.py), so its answers "
                     "cannot be reused. Pass a new --out, e.g. --out results-v2.json.")
        res["started"] = old.get("started", res["started"])
        res["spent_dropped"] = old.get("spent_dropped", 0.0)
        for c in old.get("calls", []):
            if (c["model"], c["item"], c["pass"]) in keys:
                res["calls"].append(c)
            else:
                res["spent_dropped"] += charge(c)
        done = {k for k, c in final_calls(res["calls"]).items() if not c.get("error")}
        print(f"resuming: {len(done)} successful calls already in {out}")
    jobs = [(m, it, p) for p in range(1, a.passes + 1) for it in items for m in by_id if (m, it["id"], p) not in done]
    budget = Budget(a.max_usd, sum(charge(c) for c in res["calls"]) + res["spent_dropped"])
    lock, stop, refused, last_save = threading.Lock(), threading.Event(), [], [None]

    def work(job):
        mid, it, p = job
        if stop.is_set():
            return
        is_llm = baseline is not None and mid == baseline["id"]
        est = request_estimate(by_id[mid], asks[mid], it, is_llm)
        if not budget.reserve(est):
            with lock:
                res["stopped"] = res["stopped"] or "cap"
                refused.append(est)
            stop.set()
            return
        try:
            ans, cost, err, secs, responded = (call_llm if is_llm else call_decision)(mid, asks[mid], it)
        except Exception as e:
            ans, cost, err, secs, responded = None, None, redact(f"internal error: {e!r}")[:300], 0.0, False
        amount, estimated = budget.settle(est, cost, responded)
        with lock:
            res["calls"].append({"model": mid, "item": it["id"], "pass": p, "llm": is_llm, "secs": round(secs, 2),
                                 "cost": cost, "charge": amount, "cost_estimated": estimated, "answers": ans, "error": err})
            if last_save[0] is None or _now() - last_save[0] >= SAVE_EVERY_S:
                last_save[0] = _now()
                try:
                    save_atomic(res, out)
                except OSError as e:
                    print(f"warning: could not save {out} ({e}); continuing, the next save retries", file=sys.stderr, flush=True)
            n = len(res["calls"])
            if n % 20 == 0:
                print(f"  {n} calls, ${budget.spent:.4f} spent", flush=True)

    report = os.path.splitext(out)[0] + ".md"
    ex = ThreadPoolExecutor(max(1, a.workers))
    interrupted = False
    try:
        futures = [ex.submit(work, j) for j in jobs]
        for f in as_completed(futures):
            f.result()
    except KeyboardInterrupt:
        interrupted = True
        stop.set()
        res["stopped"] = "interrupted"
        print("\nInterrupted: waiting for requests already sent, so their answers and costs are saved...", flush=True)
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        with lock:
            save_atomic(res, out)
    with lock:
        text = write_report(res, report)
    print("\n" + text)
    if interrupted:
        print(f"\nInterrupted. Finished calls are saved in {out}. "
              "Rerun the same command to continue.")
        sys.exit(130)
    if res["stopped"] == "cap":
        print(f"\nStopped at the --max-usd cap (${a.max_usd}, ${budget.spent:.4f} spent so far): the next request would have "
              f"reserved ~${max(refused or [0]):.4f}. Rerun with a higher cap to finish (finished calls are kept).")
    print(f"\nWrote {out} and {report}")


def cmd_report(a):
    with open(a.results, encoding="utf-8") as f:
        res = json.load(f)
    if a.target:
        res["target"] = a.target
    out = a.out or os.path.splitext(a.results)[0] + ".md"
    print(write_report(res, out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models", help="list live decision models and your credit")
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("questions", help="questions.json (Decisions API `questions` object)")
        p.add_argument("items", help="items.jsonl or items.csv with expected answers")
        p.add_argument("--models", help="comma-separated model ids (default: every live decision model that fits)")
        p.add_argument("--exclude", help="comma-separated model ids to leave out")
        p.add_argument("--include-aliases", action="store_true", help="also run ~alias ids such as ~typesafe/jev-latest")
        p.add_argument("--baseline", help="normal LLM to compare, e.g. a cheap vision chat model id")
        p.add_argument("--passes", type=int, default=2, help="repeat runs to measure stability (default 2)")
        p.add_argument("--target", type=float, default=0.95, help="accuracy target for the threshold suggestion")
        if name == "run":
            p.add_argument("--max-usd", type=float, required=True,
                           help="spending cap; each request reserves a conservative estimate first, none starts past the cap")
            p.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
            p.add_argument("--out", help="results file (default results.json; report goes next to it as .md)")
    p = sub.add_parser("report")
    p.add_argument("results")
    p.add_argument("--out")
    p.add_argument("--target", type=float)
    a = ap.parse_args(argv)
    {"models": cmd_models, "plan": cmd_plan, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
