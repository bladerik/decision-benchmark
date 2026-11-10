#!/usr/bin/env python3
"""Benchmark OpenRouter decision models on a labelled sample. Standard library only.

  bench.py models                       live decision models, prices, open weights, your credit
  bench.py plan   QUESTIONS ITEMS [opts] check inputs, pick models, estimate cost (no requests)
  bench.py run    QUESTIONS ITEMS --max-usd 0.50 [opts]   run, then write results + report
  bench.py report RESULTS.json [--out report.md]          rebuild the report, no requests

File formats: references/formats.md. Key: env var OPENROUTER_API_KEY.
"""
import argparse, base64, csv, json, os, statistics, struct, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://openrouter.ai/api"
MODELS_URL = API + "/v1/models?output_modalities=decisions"
IMAGE_TOKENS_GUESS = 1600  # rough upper estimate per image, only for the pre-run cost estimate
THRESHOLDS = (0.8, 0.9)


# ---------- HTTP ----------

def key():
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not k:
        sys.exit("OPENROUTER_API_KEY is not set. Export it in the terminal that starts the agent, then retry.")
    return k


def http(method, url, body=None, auth=True, timeout=180):
    headers = {"Content-Type": "application/json", "X-Title": "decision-benchmark"}
    if auth:
        headers["Authorization"] = "Bearer " + key()
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def post_with_retry(url, body):
    """Returns (response, error, seconds). Retries 429 and 5xx with backoff."""
    err = None
    for attempt in range(4):
        t0 = time.time()
        try:
            return http("POST", url, body), None, time.time() - t0
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
            if e.code not in (408, 429, 500, 502, 503, 504):
                break
        except Exception as e:  # network, timeout, bad JSON
            err = repr(e)[:300]
        time.sleep(2 * 2 ** attempt)
    return None, err, time.time() - t0


def live_models():
    return http("GET", MODELS_URL, auth=False)["data"]


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
            it = {"id": r.get("id") or str(len(items) + 1), "text": r.get("text") or "",
                  "images": [p for p in (r.get("image") or "").split("|") if p.strip()],
                  "expected": {q: r[q] for q in questions if (r.get(q) or "").strip() != ""}}
            items.append(it)
    else:
        items = []
        for n, line in enumerate(open(path, encoding="utf-8"), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            imgs = r.get("images") or ([r["image"]] if r.get("image") else [])
            items.append({"id": str(r.get("id", n)), "text": r.get("text", ""), "state": r.get("state"),
                          "images": imgs, "expected": r.get("expected", {})})
    for it in items:
        it["images"] = [p if os.path.isabs(p) else os.path.join(base, p.strip()) for p in it["images"]]
    return items


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
        i = int(v)
        if not 0 <= i < len(q["criteria"]):
            raise ValueError(f"score expected {v!r} must be a level index 0..{len(q['criteria']) - 1}")
        return i
    if t == "box":
        b = json.loads(v) if isinstance(v, str) else v
        if isinstance(b, dict):
            b = [b["top"], b["bottom"], b["left"], b["right"]]
        return [float(x) for x in b]
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
        if not it["text"] and not it["images"] and not it.get("state"):
            problems.append(f"item {it['id']}: has no text, state or image")
    for k, n in labelled.items():
        if n == 0:
            warnings.append(f"question {k}: no item has an expected answer, so it is asked but not scored")
        elif n < 30:
            warnings.append(f"question {k}: only {n} labelled items; accuracy differences of a few points are noise")
    return problems, warnings


def data_url(path):
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def build_state(it):
    """Plain object when there are no images; a list of text + image parts when there are."""
    if not it["images"]:
        return it["state"] if it.get("state") is not None else {"text": it["text"]}
    st = []
    if it.get("state") is not None:
        st.append(json.dumps(it["state"], ensure_ascii=False))
    if it["text"]:
        st.append("text: " + it["text"])
    for n, p in enumerate(it["images"], 1):
        st.append(f"Image {n} (`image{n if len(it['images']) > 1 else ''}`):")
        st.append({"type": "image_url", "image_url": {"url": data_url(p)}})
    return st


# ---------- model selection and cost ----------

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
        chosen.append(m)
    return chosen, skipped, live


def est_tokens(questions, it):
    text = len(json.dumps(questions, ensure_ascii=False)) + len(it["text"]) + len(json.dumps(it.get("state") or ""))
    return text / 4 + IMAGE_TOKENS_GUESS * len(it["images"])


def estimate(models, baseline, questions, items, passes):
    toks = sum(est_tokens(questions, it) for it in items) * passes
    rows = [(m["id"], toks * float(m["pricing"].get("prompt") or 0)) for m in models]
    if baseline:
        p = baseline["pricing"]
        rows.append((baseline["id"] + " (LLM)", toks * 1.3 * float(p.get("prompt") or 0) + len(items) * passes * 300 * float(p.get("completion") or 0)))
    return rows


# ---------- calls ----------

def llm_schema(questions):
    props = {}
    for k, q in questions.items():
        desc = q.get("instructions") if isinstance(q.get("instructions"), str) else json.dumps(q.get("instructions"))
        if q["type"] == "noul":
            props[k] = {"type": "boolean", "description": desc or ""}
        elif q["type"] == "choice":
            props[k] = {"type": "string", "enum": list(q["criteria"]),
                        "description": (desc or "") + " Options: " + json.dumps(q["criteria"], ensure_ascii=False)}
        elif q["type"] == "score":
            props[k] = {"type": "integer", "enum": list(range(len(q["criteria"]))),
                        "description": (desc or "") + " Levels (index: meaning): " + json.dumps(dict(enumerate(q["criteria"])), ensure_ascii=False)}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def call_llm(model, questions, it):
    st = build_state(it)
    st = st if isinstance(st, list) else [json.dumps(st, ensure_ascii=False)]
    images = [x for x in st if isinstance(x, dict)]
    text = "\n".join(x for x in st if isinstance(x, str))
    parts = [{"type": "text", "text": "Answer every field strictly and factually from this content.\n" + text}] + images
    body = {"model": model, "messages": [{"role": "user", "content": parts}], "usage": {"include": True},
            "response_format": {"type": "json_schema", "json_schema": {"name": "decisions", "strict": True, "schema": llm_schema(questions)}}}
    r, err, secs = post_with_retry(API + "/v1/chat/completions", body)
    if err:
        return None, None, err, secs
    try:
        raw = json.loads(r["choices"][0]["message"]["content"])
    except Exception as e:
        return None, (r.get("usage") or {}).get("cost"), f"unparseable LLM output: {e!r}"[:300], secs
    ans = {}
    for k, q in questions.items():
        if k not in raw:
            continue
        if q["type"] == "noul":
            ans[k] = {"type": "noul", "noul": 1.0 if raw[k] else 0.0}
        elif q["type"] == "choice":
            ans[k] = {"type": "choice", "choice": raw[k]}
        elif q["type"] == "score":
            ans[k] = {"type": "score", "score": raw[k]}
    return ans, (r.get("usage") or {}).get("cost"), None, secs


def answers_box(model):
    return model.lstrip("~").startswith("typesafe/jev") and model != "typesafe/jev-1.13"


def call_decision(model, questions, it):
    if not answers_box(model):  # only Jev 1.14+ answers box; ask the other models the rest
        questions = {k: q for k, q in questions.items() if q["type"] != "box"}
    r, err, secs = post_with_retry(API + "/alpha/decisions", {"model": model, "state": build_state(it), "questions": questions})
    if err:
        return None, None, err, secs
    return r.get("answers", {}), (r.get("usage") or {}).get("cost"), None, secs


# ---------- scoring ----------

def iou(a, b):
    t, bo, l, r = a
    t2, b2, l2, r2 = b
    inter = max(0, min(bo, b2) - max(t, t2)) * max(0, min(r, r2) - max(l, l2))
    union = (bo - t) * (r - l) + (b2 - t2) * (r2 - l2) - inter
    return inter / union if union > 0 else 0


def decide(a):
    """Returns (decision, confidence) for one answer; decision None for a refusal."""
    t = a.get("type")
    if t == "noul":
        return a["noul"] >= 0.5, a.get("confidence", max(a["noul"], 1 - a["noul"]))
    if t == "choice":
        return a["choice"], a.get("confidence", (a.get("probabilities") or {}).get(a["choice"]))
    if t == "score":
        return int(round(a["score"])), a.get("confidence")
    if t == "box":
        b = a["box"]
        return (b["top"], b["bottom"], b["left"], b["right"]), a.get("confidence")
    return None, None


def correct(decision, expected):
    if isinstance(expected, list):  # box: overlap of at least half counts as right
        return decision is not None and iou(decision, expected) >= 0.5
    return decision == expected


def same(a, b):
    if isinstance(a, tuple) and isinstance(b, tuple):
        return iou(a, b) >= 0.5
    return a == b


def summarise(res):
    questions, items = res["questions"], {it["id"]: it for it in res["items"]}
    by_model = {}
    for c in res["calls"]:
        by_model.setdefault(c["model"], []).append(c)
    rows = []
    for mid, calls in by_model.items():
        ok_calls = [c for c in calls if not c.get("error")]
        judged, decisions, perq = [], {}, {}
        for c in ok_calls:
            for k, a in (c["answers"] or {}).items():
                if k not in questions:
                    continue
                d, conf = decide(a)
                decisions[(c["item"], k, c["pass"])] = d
                exp = items[c["item"]]["expected"].get(k)
                if exp is None:
                    continue
                ok = correct(d, exp)
                judged.append((ok, None if c.get("llm") else conf))
                perq.setdefault(k, []).append(ok)
        pairs = [(v, decisions.get((i, k, 2))) for (i, k, p), v in decisions.items() if p == 1 and (i, k, 2) in decisions]
        meta = res["models"].get(mid, {})
        costs = [c["cost"] for c in ok_calls if c.get("cost") is not None]
        row = {"model": mid, "llm": bool(meta.get("llm")), "n": len(judged), "calls": len(calls),
               "errors": len(calls) - len(ok_calls), "error_sample": next((c["error"] for c in calls if c.get("error")), None),
               "acc": sum(o for o, _ in judged) / len(judged) if judged else None,
               "median_s": statistics.median(c["secs"] for c in ok_calls) if ok_calls else None,
               "usd_per_1k": 1000 * statistics.mean(costs) if costs else None,
               "stable": sum(same(a, b) for a, b in pairs) / len(pairs) if pairs else None,
               "hf": meta.get("hugging_face_id"), "perq": {k: (sum(v), len(v)) for k, v in perq.items()},
               "spent": sum(costs)}
        for t in THRESHOLDS:
            confident = [o for o, conf in judged if conf is not None and conf >= t]
            has_conf = any(conf is not None for _, conf in judged)
            row[f"acc@{t}"] = (sum(confident) / len(confident) if confident else None) if has_conf else None
            row[f"cov@{t}"] = (len(confident) / len(judged) if judged else None) if has_conf else None
        row["auto"] = auto_threshold(judged, res.get("target", 0.95))
        rows.append(row)
    rows.sort(key=lambda r: (-(r["acc"] or 0), r["usd_per_1k"] or 9e9))
    return rows


def auto_threshold(judged, target):
    """Lowest confidence threshold whose answers reach the target accuracy, with its coverage."""
    with_conf = [(o, c) for o, c in judged if c is not None]
    if not with_conf:
        return None
    for t in [x / 100 for x in range(50, 100)]:
        kept = [o for o, c in with_conf if c >= t]
        if len(kept) >= 5 and sum(kept) / len(kept) >= target:
            return t, len(kept) / len(judged)
    return None


# ---------- report ----------

def pct(x):
    return "–" if x is None else f"{100 * x:.0f} %"


def hard_items(res):
    items = {it["id"]: it["expected"] for it in res["items"]}
    tally = {}
    for c in res["calls"]:
        for k, a in (c.get("answers") or {}).items():
            exp = items.get(c["item"], {}).get(k)
            if exp is None:
                continue
            w, n, _ = tally.get((c["item"], k), (0, 0, exp))
            tally[(c["item"], k)] = (w + (not correct(decide(a)[0], exp)), n + 1, exp)
    return sorted(((key, v) for key, v in tally.items() if v[1] and v[0] / v[1] >= 0.5), key=lambda x: -x[1][0] / x[1][1])


def num(x, fmt):
    return "–" if x is None else fmt.format(x)


def write_report(res, out):
    rows = summarise(res)
    target = res.get("target", 0.95)
    L = [f"# Decision model benchmark", "",
         f"{len(res['items'])} items × {len(res['questions'])} questions × {res['passes']} passes, run {res['started']}. "
         f"Total spent: ${sum(r['spent'] for r in rows):.4f}.", "",
         "| Model | Accuracy | Acc ≥0.9 (coverage) | Acc ≥0.8 (coverage) | Median s | $ / 1 000 items | Same answer both passes | Open weights | Failed calls |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        name = r["model"] + (" (LLM baseline)" if r["llm"] else "")
        ow = "–" if r["llm"] else (f"yes ([{r['hf']}](https://huggingface.co/{r['hf']}))" if r["hf"] else "no")
        L.append(f"| {name} | {pct(r['acc'])} | {pct(r['acc@0.9'])} ({pct(r['cov@0.9'])}) | {pct(r['acc@0.8'])} ({pct(r['cov@0.8'])}) | "
                 f"{num(r['median_s'], '{:.1f}')} | {num(r['usd_per_1k'], '${:.3f}')} | "
                 f"{pct(r['stable'])} | {ow} | {r['errors']}/{r['calls']} |")
    L += ["", "Accuracy counts every labelled (item, question) answer over all passes. \"Acc ≥0.9 (coverage)\": accuracy on answers "
          "with confidence ≥ 0.9, and the share of answers that clear it; the rest would go to a fallback. One item = one request answering all questions. "
          "LLM baselines return no confidence, so their threshold columns are empty.", "",
          f"## Threshold for ≥ {target * 100:.0f} % accuracy", ""]
    for r in rows:
        if r["llm"] or r["acc"] is None:
            continue
        a = r["auto"]
        L.append(f"- {r['model']}: " + (f"confidence ≥ {a[0]:.2f} keeps {pct(a[1])} of answers" if a else "no threshold reaches it on this sample"))
    qs = list(res["questions"])
    L += ["", "## Accuracy per question", "", "| Model | " + " | ".join(qs) + " |", "|---|" + "---|" * len(qs)]
    for r in rows:
        L.append(f"| {r['model']} | " + " | ".join(f"{r['perq'][q][0]}/{r['perq'][q][1]}" if q in r["perq"] else "–" for q in qs) + " |")
    hard = hard_items(res)
    if hard:
        L += ["", "## Items most models got wrong", "",
              "Either the expected answer is wrong or the question can be read another way. Check these first.", ""]
        L += [f"- {i} / {k}: expected {json.dumps(res_exp)}, wrong in {w} of {n} model answers" for (i, k), (w, n, res_exp) in hard]
    errs = [r for r in rows if r["error_sample"]]
    if errs:
        L += ["", "## Errors", ""] + [f"- {r['model']}: {r['error_sample']}" for r in errs]
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    return "\n".join(L)


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
    else:
        print("\nOPENROUTER_API_KEY is not set.")


def prepare(a):
    questions = json.load(open(a.questions, encoding="utf-8"))
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
    if any(q["type"] == "box" for q in questions.values()):
        print("note: only Jev 1.14+ answers box questions; the other models get the remaining questions")
    return questions, items, models, skipped, baseline


def print_plan(questions, items, models, skipped, baseline, passes):
    print(f"\n{len(items)} items, {len(questions)} questions, {passes} passes")
    for mid, why in skipped:
        print(f"  skip {mid}: {why}")
    rows = estimate(models, baseline, questions, items, passes)
    for mid, usd in rows:
        print(f"  run  {mid:45} ~${usd:.5f}")
    total = sum(u for _, u in rows)
    print(f"\n{len(rows)} models, {len(items) * passes * len(rows)} requests, estimated ${total:.3f} "
          f"(rough: images are guessed at {IMAGE_TOKENS_GUESS} tokens each; the run stops at --max-usd)")
    return total


def cmd_plan(a):
    q, items, models, skipped, baseline = prepare(a)
    print_plan(q, items, models, skipped, baseline, a.passes)
    if os.environ.get("OPENROUTER_API_KEY"):
        c = credit()
        print("Remaining credit: " + ("unknown" if c is None else f"${c:.2f}"))


def cmd_run(a):
    questions, items, models, skipped, baseline = prepare(a)
    print_plan(questions, items, models, skipped, baseline, a.passes)
    out = a.out or "results.json"
    res = {"started": time.strftime("%Y-%m-%d %H:%M"), "passes": a.passes, "target": a.target, "questions": questions,
           "items": [{"id": it["id"], "expected": it["expected"]} for it in items], "calls": [],
           "models": {m["id"]: {"hugging_face_id": m.get("hugging_face_id")} for m in models}}
    done = {}
    if os.path.exists(out):  # resume: keep successful calls from an earlier, interrupted run
        old = json.load(open(out, encoding="utf-8"))
        done = {(c["model"], c["item"], c["pass"]): c for c in old.get("calls", []) if not c.get("error")}
        res["calls"] = list(done.values())
        print(f"resuming: {len(done)} successful calls already in {out}")
    if baseline:
        res["models"][baseline["id"]] = {"llm": True}
    jobs = [(m, it, p) for p in range(1, a.passes + 1) for it in items for m in [x["id"] for x in models] + ([baseline["id"]] if baseline else [])
            if (m, it["id"], p) not in done]
    spent = [sum(c.get("cost") or 0 for c in res["calls"])]
    lock = threading.Lock()

    def work(job):
        mid, it, p = job
        with lock:
            if spent[0] >= a.max_usd:
                return
        is_llm = baseline is not None and mid == baseline["id"]
        ans, cost, err, secs = (call_llm if is_llm else call_decision)(mid, questions, it)
        with lock:
            spent[0] += cost or 0
            res["calls"].append({"model": mid, "item": it["id"], "pass": p, "llm": is_llm, "secs": round(secs, 2),
                                 "cost": cost, "answers": ans, "error": err})
            n = len(res["calls"])
            if n % 20 == 0:
                print(f"  {n} calls, ${spent[0]:.4f} spent", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if spent[0] >= a.max_usd:
        print(f"\nStopped at the --max-usd cap (${a.max_usd}); rerun with a higher cap to finish (finished calls are kept).")
    report = os.path.splitext(out)[0] + ".md"
    print("\n" + write_report(res, report))
    print(f"\nWrote {out} and {report}")


def cmd_report(a):
    res = json.load(open(a.results, encoding="utf-8"))
    if a.target:
        res["target"] = a.target
    out = a.out or os.path.splitext(a.results)[0] + ".md"
    print(write_report(res, out))


def main():
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
            p.add_argument("--max-usd", type=float, required=True, help="hard spending cap; no new requests after it")
            p.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
            p.add_argument("--out", help="results file (default results.json; report goes next to it as .md)")
    p = sub.add_parser("report")
    p.add_argument("results")
    p.add_argument("--out")
    p.add_argument("--target", type=float)
    a = ap.parse_args()
    {"models": cmd_models, "plan": cmd_plan, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
