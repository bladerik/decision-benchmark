import time
import json
"""Offline tests for bench.py. Run from this folder: python3 -m unittest"""
import contextlib, io, json, os, struct, tempfile, threading, time, unittest, urllib.error
from unittest import mock

import bench

KEY = "sk-or-v1-testkey0123456789"


def png(path, w, h, extra=b""):
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", w, h) + b"\x08\x02\x00\x00\x00" + extra)


def dm(mid, prompt="0.000001", inputs=("text", "image"), alias=None, hf=None):
    m = {"id": mid, "pricing": {"prompt": prompt, "completion": "0"}, "hugging_face_id": hf,
         "architecture": {"input_modalities": list(inputs)}}
    if alias:
        m["alias_target"] = {"slug": alias}
    return m


def llm(mid, params=("response_format", "structured_outputs", "max_tokens"), inputs=("text", "image")):
    return {"id": mid, "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            "supported_parameters": list(params), "architecture": {"input_modalities": list(inputs)}}


class FakeAPI:
    """Stands in for bench.http: routes by URL, records every request."""

    def __init__(self, decision_models, all_models=(), on_decision=None, on_chat=None):
        self.decision_models, self.all_models = list(decision_models), list(all_models)
        self.on_decision, self.on_chat = on_decision, on_chat
        self.posts, self.lock = [], threading.Lock()

    def __call__(self, method, url, body=None, auth=True, timeout=180):
        if method == "GET" and url == bench.MODELS_URL:
            return {"data": self.decision_models}
        if method == "GET" and url == bench.API + "/v1/models":
            return {"data": self.all_models}
        if method == "GET" and url.endswith("/v1/credits"):
            return {"data": {"total_credits": 10, "total_usage": 1}}
        with self.lock:
            self.posts.append((url, body))
        if url.endswith("/alpha/decisions"):
            return self.on_decision(body)
        if url.endswith("/chat/completions"):
            return self.on_chat(body)
        raise AssertionError("unexpected request " + url)


def yes_to_all(body, cost=0.001):
    ans = {}
    for k, q in body["questions"].items():
        if q["type"] == "noul":
            ans[k] = {"type": "noul", "noul": 0.97}
    return {"answers": ans, "usage": {"cost": cost}}


class Workspace(unittest.TestCase):
    """Temp folder with questions + items, the key set, HTTP mocked, no real sleeps."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": KEY})
        p.start()
        self.addCleanup(p.stop)
        s = mock.patch.object(bench, "_sleep", lambda s: None)
        s.start()
        self.addCleanup(s.stop)

    def path(self, name):
        return os.path.join(self.dir, name)

    def write(self, name, questions=None, items=None, text=None):
        p = self.path(name)
        with open(p, "w", encoding="utf-8") as f:
            if questions is not None:
                json.dump(questions, f)
            elif items is not None:
                f.write("\n".join(json.dumps(i) for i in items) + "\n")
            else:
                f.write(text)
        return p

    def files(self, n=2, questions=None):
        q = questions or {"spam": {"type": "noul", "instructions": "`text` is spam"}}
        items = [{"id": f"i{k}", "text": f"message {k}", "expected": {"spam": True}} for k in range(n)]
        return self.write("questions.json", questions=q), self.write("items.jsonl", items=items)

    def main(self, api, *argv):
        out = io.StringIO()
        with mock.patch.object(bench, "http", api), contextlib.redirect_stdout(out):
            try:
                bench.main(list(argv))
                code = 0
            except SystemExit as e:
                code = e.code
        return code, out.getvalue()

    def results(self, name="results.json"):
        return json.loads(self.read(name))

    def read(self, name="results.md"):
        with open(self.path(name), encoding="utf-8") as f:
            return f.read()


# ---------- Sol #1: the key never leaks ----------

class KeyTests(Workspace):
    def test_key_with_line_break_is_rejected_without_echo(self):
        for bad in ("sk-or-v1-abc\ndef", "sk-or-v1-abc def"):
            with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": bad}):
                with self.assertRaises(SystemExit) as cm:
                    bench.key()
            self.assertNotIn("abc", str(cm.exception.code))

    def test_captured_errors_redact_the_key(self):
        def boom(*a, **k):
            raise ValueError(f"Invalid header value b'Bearer {KEY}'")
        with mock.patch.object(bench, "http", boom):
            r, err, secs = bench.post_with_retry(bench.API + "/alpha/decisions", {})
        self.assertIsNone(r)
        self.assertNotIn("testkey0123456789", err)
        self.assertIn("redacted", err)
        self.assertNotIn("zzz999", bench.redact("echo sk-or-v1-zzz999 and Bearer zzz999"))


# ---------- Sol #18, Opus #15: retries ----------

class RetryTests(Workspace):
    def test_latency_covers_the_whole_retry_operation(self):
        clock, n = [0.0], [0]

        def http(*a, **k):
            n[0] += 1
            clock[0] += 1
            if n[0] == 1:
                raise urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO(b"busy"))
            return {"ok": 1}

        def sleep(s):
            clock[0] += s
        with mock.patch.object(bench, "http", http), mock.patch.object(bench, "_now", lambda: clock[0]), \
                mock.patch.object(bench, "_sleep", sleep):
            r, err, secs = bench.post_with_retry("u", {})
        self.assertEqual(r, {"ok": 1})
        self.assertEqual(secs, 4.0)  # 1 s failed attempt + 2 s backoff + 1 s success

    def test_no_sleep_after_the_last_attempt(self):
        sleeps = []

        def http(*a, **k):
            raise urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO(b"busy"))
        with mock.patch.object(bench, "http", http), mock.patch.object(bench, "_sleep", sleeps.append):
            r, err, secs = bench.post_with_retry("u", {})
        self.assertEqual(len(sleeps), bench.ATTEMPTS - 1)
        self.assertIn("503", err)

        sleeps.clear()

        def http400(*a, **k):
            raise urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b"bad"))
        with mock.patch.object(bench, "http", http400), mock.patch.object(bench, "_sleep", sleeps.append):
            bench.post_with_retry("u", {})
        self.assertEqual(sleeps, [])


# ---------- both: checkpoints, interrupts, worker exceptions ----------

class CheckpointTests(Workspace):
    def test_results_are_saved_after_every_call(self):
        q, items = self.files(3)
        seen = []

        def on_decision(body):
            out = self.path("results.json")
            seen.append(len(self.results()["calls"]) if os.path.exists(out) else 0)
            return yes_to_all(body)
        api = FakeAPI([dm("a/m")], on_decision=on_decision)
        with mock.patch.object(bench, "SAVE_EVERY_S", 0):
            code, _ = self.main(api, "run", q, items, "--models", "a/m", "--passes", "1", "--max-usd", "1",
                                "--workers", "1", "--out", self.path("results.json"))
        self.assertEqual(code, 0)
        self.assertEqual(seen, [0, 1, 2])
        self.assertFalse(os.path.exists(self.path("results.json.tmp")))

    def test_interrupt_keeps_finished_calls_and_marks_the_run(self):
        q, items = self.files(3)
        n = [0]

        def on_decision(body):
            n[0] += 1
            if n[0] > 1:
                raise KeyboardInterrupt
            return yes_to_all(body)
        api = FakeAPI([dm("a/m")], on_decision=on_decision)
        code, out = self.main(api, "run", q, items, "--models", "a/m", "--passes", "1", "--max-usd", "1",
                              "--workers", "1", "--out", self.path("results.json"))
        self.assertNotEqual(code, 0)
        res = self.results()
        self.assertEqual(res["stopped"], "interrupted")
        self.assertEqual([c["error"] for c in res["calls"]].count(None), 1)
        md = self.read()
        self.assertIn("Partial run", "\n".join(md.splitlines()[:6]))

    def test_interrupt_waits_for_requests_in_flight_before_the_final_save(self):
        q, items = self.files(3)

        def on_decision(body):
            text = json.dumps(body)
            if "message 2" in text:
                time.sleep(0.05)
                raise KeyboardInterrupt
            if "message 1" in text:
                time.sleep(0.3)
            return yes_to_all(body)
        api = FakeAPI([dm("a/m")], on_decision=on_decision)
        with mock.patch.object(bench, "SAVE_EVERY_S", 100):
            code, _ = self.main(api, "run", q, items, "--models", "a/m", "--passes", "1", "--max-usd", "1",
                                "--workers", "3", "--out", self.path("results.json"))
        self.assertNotEqual(code, 0)
        ok = sorted(c["item"] for c in self.results()["calls"] if not c["error"])
        self.assertEqual(ok, ["i0", "i1"])

    def test_exception_inside_a_worker_becomes_a_call_error(self):
        q, items = self.files(2)
        real = bench.build_state

        def build_state(it):
            if it["id"] == "i1":
                raise RuntimeError("broken item")
            return real(it)
        api = FakeAPI([dm("a/m")], on_decision=yes_to_all)
        with mock.patch.object(bench, "build_state", build_state):
            code, _ = self.main(api, "run", q, items, "--models", "a/m", "--passes", "1", "--max-usd", "1",
                                "--out", self.path("results.json"))
        self.assertEqual(code, 0)
        errs = [c for c in self.results()["calls"] if c["error"]]
        self.assertEqual(len(errs), 1)
        self.assertIn("broken item", errs[0]["error"])


# ---------- both: full denominator, partial runs ----------

def make_res(questions, items, calls, models, passes=1, **extra):
    r = {"started": "t", "passes": passes, "target": 0.95, "questions": questions, "items": items,
         "calls": calls, "models": models}
    r.update(extra)
    return r


def call(mid, item, answers, p=1, error=None, cost=0.001, llm=False):
    return {"model": mid, "item": item, "pass": p, "llm": llm, "secs": 0.1, "cost": cost, "answers": answers, "error": error}


Q1 = {"q1": {"type": "noul", "instructions": "x"}}
YES = {"type": "noul", "noul": 0.97}


class DenominatorTests(unittest.TestCase):
    def test_missing_malformed_and_errored_answers_count_as_wrong(self):
        items = [{"id": i, "expected": {"q1": True}} for i in "abcde"]
        calls = [call("m", "a", {"q1": YES}),
                 call("m", "b", {}),  # answer missing
                 call("m", "c", {"q1": {"type": "noul", "noul": "yes"}}),  # malformed
                 call("m", "d", "garbage"),  # not an object
                 call("m", "e", None, error="HTTP 500")]
        res = make_res(Q1, items, calls, {"m": {"questions": ["q1"], "planned": 5}})
        row = bench.summarise(res)[0]
        self.assertAlmostEqual(row["acc"], 1 / 5)
        self.assertAlmostEqual(row["cov@0.9"], 1 / 5)
        self.assertEqual(row["errors"], 1)

    def test_partial_run_is_marked_at_the_top_with_planned_vs_completed(self):
        items = [{"id": i, "expected": {"q1": True}} for i in "abcd"]
        res = make_res(Q1, items, [call("m", "a", {"q1": YES})], {"m": {"questions": ["q1"], "planned": 4}}, stopped="cap")
        with tempfile.TemporaryDirectory() as d:
            md = bench.write_report(res, os.path.join(d, "r.md"))
        top = "\n".join(md.splitlines()[:6])
        self.assertIn("Partial run", top)
        self.assertIn("cap", top)
        self.assertIn("1/4", md)


# ---------- both: --max-usd ----------

class BudgetTests(Workspace):
    def test_concurrent_requests_cannot_overshoot_the_cap(self):
        q, items = self.files(4)

        def on_decision(body):
            time.sleep(0.05)
            return yes_to_all(body, cost=0.4)
        api = FakeAPI([dm("a/m")], on_decision=on_decision)
        with mock.patch.object(bench, "request_estimate", lambda *a, **k: 0.3):
            code, out = self.main(api, "run", q, items, "--models", "a/m", "--passes", "1", "--max-usd", "0.5",
                                  "--workers", "4", "--out", self.path("results.json"))
        self.assertEqual(len(api.posts), 1)
        res = self.results()
        self.assertEqual(res["stopped"], "cap")
        self.assertIn("Partial run", self.read())

    def test_missing_cost_is_counted_at_the_estimate_and_reported(self):
        q, items = self.files(1)
        api = FakeAPI([dm("a/m")], on_decision=lambda b: {"answers": yes_to_all(b)["answers"]})
        with mock.patch.object(bench, "request_estimate", lambda *a, **k: 0.3):
            self.main(api, "run", q, items, "--models", "a/m", "--passes", "1", "--max-usd", "5",
                      "--out", self.path("results.json"))
        c = self.results()["calls"][0]
        self.assertTrue(c["cost_estimated"])
        self.assertAlmostEqual(c["charge"], 0.3)
        md = self.read()
        self.assertIn("$0.3000", md)
        self.assertIn("no usage.cost", md)

    def test_baseline_output_tokens_are_bounded(self):
        q, items = self.files(1)
        bodies = []

        def on_chat(body):
            bodies.append(body)
            return {"choices": [{"message": {"content": json.dumps({"spam": True})}}], "usage": {"cost": 0.001}}
        api = FakeAPI([dm("a/m")], [llm("x/chat")], on_decision=yes_to_all, on_chat=on_chat)
        self.main(api, "run", q, items, "--models", "a/m", "--baseline", "x/chat", "--passes", "1", "--max-usd", "5",
                  "--out", self.path("results.json"))
        self.assertIsInstance(bodies[0]["max_tokens"], int)
        self.assertLessEqual(bodies[0]["max_tokens"], 4096)

    def test_failed_calls_count_in_spent_and_in_the_resumed_ledger(self):
        q, items = self.files(1)
        api = FakeAPI([dm("a/m")], [llm("x/chat")], on_decision=yes_to_all,
                      on_chat=lambda b: {"choices": [{"message": {"content": "not json"}}], "usage": {"cost": 0.1}})
        with mock.patch.object(bench, "request_estimate", lambda *a, **k: 0.001):
            self.main(api, "run", q, items, "--models", "a/m", "--baseline", "x/chat", "--passes", "1",
                      "--max-usd", "5", "--out", self.path("results.json"))
        md = self.read()
        self.assertIn("Total spent: $0.1010", md)
        # rerun: the failed baseline call is retried, but the earlier $0.10 still counts against the cap
        api2 = FakeAPI([dm("a/m")], [llm("x/chat")], on_decision=yes_to_all,
                       on_chat=lambda b: {"choices": [{"message": {"content": "not json"}}], "usage": {"cost": 0.1}})
        with mock.patch.object(bench, "request_estimate", lambda *a, **k: 0.05):
            code, out = self.main(api2, "run", q, items, "--models", "a/m", "--baseline", "x/chat", "--passes", "1",
                                  "--max-usd", "0.12", "--out", self.path("results.json"))
        self.assertEqual(api2.posts, [])
        self.assertIn("Total spent: $0.1010", self.read())


# ---------- both: inputs ----------

class InputTests(Workspace):
    def test_text_and_state_are_both_sent(self):
        st = bench.build_state({"text": "customer message", "state": {"subject": "Invoice"}, "images": []})
        self.assertEqual(st, {"subject": "Invoice", "text": "customer message"})

    def test_duplicate_ids_are_rejected(self):
        q = {"spam": {"type": "noul"}}
        p = self.write("items.jsonl", items=[{"id": "1", "text": "a"}, {"id": " 1", "text": "b"}])
        problems, _ = bench.validate(q, bench.load_items(p, q))
        self.assertTrue(any("duplicate" in x for x in problems), problems)
        c = self.write("items.csv", text="id,text,spam\n2,a,yes\n,b,no\n")
        problems, _ = bench.validate(q, bench.load_items(c, q))
        self.assertTrue(any("duplicate" in x for x in problems), problems)

    def test_bad_box_labels_are_rejected(self):
        q = {"dmg": {"type": "box", "instructions": "damage in `image`"}}
        bad = [[0.1, 0.5, 0.2], [0.5, 0.1, 0.2, 0.3], [0.1, 0.5, 0.6, 0.2], [0, 1, 0, 1.5], [0.1, float("nan"), 0.2, 0.3],
               ["a", 0.5, 0.2, 0.3]]
        for b in bad:
            items = [{"id": "x", "text": "t", "images": [], "expected": {"dmg": b}}]
            problems, _ = bench.validate(q, items)
            self.assertTrue(problems, b)
        items = [{"id": "x", "text": "t", "images": [], "expected": {"dmg": [0.1, 0.5, 0.2, 0.6]}}]
        self.assertEqual(bench.validate(q, items)[0], [])

    def test_non_integral_score_labels_are_rejected(self):
        q = {"u": {"type": "score", "criteria": ["low", "mid", "high"]}}
        for bad in (1.5, "1.5", True):
            problems, _ = bench.validate(q, [{"id": "x", "text": "t", "images": [], "expected": {"u": bad}}])
            self.assertTrue(problems, bad)
        for good, want in ((2, 2), ("2", 2), (2.0, 2)):
            items = [{"id": "x", "text": "t", "images": [], "expected": {"u": good}}]
            self.assertEqual(bench.validate(q, items)[0], [])
            self.assertEqual(items[0]["expected"]["u"], want)


# ---------- both: resume ----------

class ResumeTests(Workspace):
    def run_once(self, q, items, *extra):
        api = FakeAPI([dm("a/m"), dm("b/m")], on_decision=yes_to_all)
        code, out = self.main(api, "run", q, items, "--passes", "1", "--max-usd", "5",
                              "--out", self.path("results.json"), *extra)
        return api, code, out

    def test_resume_refuses_changed_items(self):
        q, items = self.files(2)
        self.run_once(q, items, "--models", "a/m")
        self.write("items.jsonl", items=[{"id": "i0", "text": "changed", "expected": {"spam": True}},
                                         {"id": "i1", "text": "message 1", "expected": {"spam": True}}])
        api, code, out = self.run_once(q, items, "--models", "a/m")
        self.assertNotEqual(code, 0)
        self.assertIn("--out", str(code) + out)
        self.assertEqual(api.posts, [])

    def test_resume_refuses_a_changed_image_file(self):
        png(self.path("p.png"), 400, 400, b"one")
        q = self.write("questions.json", questions={"spam": {"type": "noul", "instructions": "`image` is spam"}})
        items = self.write("items.jsonl", items=[{"id": "p", "image": "p.png", "expected": {"spam": True}}])
        self.run_once(q, items, "--models", "a/m")
        png(self.path("p.png"), 400, 400, b"two")
        api, code, out = self.run_once(q, items, "--models", "a/m")
        self.assertNotEqual(code, 0)
        self.assertEqual(api.posts, [])

    def test_resume_drops_calls_outside_the_current_job_set(self):
        q, items = self.files(2)
        self.run_once(q, items, "--models", "a/m,b/m")
        api, code, out = self.run_once(q, items, "--models", "a/m")
        self.assertEqual(code, 0)
        self.assertEqual(api.posts, [])
        self.assertEqual({c["model"] for c in self.results()["calls"]}, {"a/m"})
        self.assertNotIn("b/m", self.read().split("Total spent")[1])


# ---------- both: threshold and per-question accuracy ----------

class ThresholdTests(unittest.TestCase):
    def test_threshold_at_full_confidence_is_found(self):
        judged = [(True, 1.0)] * 25 + [(False, 0.995)] * 25
        self.assertEqual(bench.auto_threshold(judged, 0.95), (1.0, 0.5, 25))

    def test_threshold_below_one_half_is_found(self):
        self.assertEqual(bench.auto_threshold([(True, 0.4)] * 30, 0.95), (0.4, 1.0, 30))

    def test_threshold_needs_twenty_kept_answers(self):
        self.assertIsNone(bench.auto_threshold([(True, 0.99)] * 19, 0.95))

    def test_threshold_coverage_uses_the_full_denominator(self):
        judged = [(True, 0.99)] * 25 + [(False, None)] * 25
        self.assertEqual(bench.auto_threshold(judged, 0.95), (0.99, 0.5, 25))

    def test_report_prints_kept_count_and_per_question_accuracy(self):
        q = {"q1": {"type": "noul"}, "q2": {"type": "noul"}}
        items = [{"id": str(i), "expected": {"q1": True, "q2": False}} for i in range(25)]
        calls = [call("m", str(i), {"q1": YES, "q2": YES}) for i in range(25)]
        res = make_res(q, items, calls, {"m": {"questions": ["q1", "q2"], "planned": 25}})
        with tempfile.TemporaryDirectory() as d:
            md = bench.write_report(res, os.path.join(d, "r.md"))
        per_q = md.split("## Accuracy per question")[1]
        self.assertIn("100 %", per_q)
        self.assertIn("0 %", per_q)
        self.assertIn("no threshold", md.split("## Threshold")[1].split("##")[0])
        res["calls"] = [call("m", str(i), {"q1": YES, "q2": {"type": "noul", "noul": 0.6}}) for i in range(25)]
        md = bench.write_report(res, os.devnull)
        self.assertIn("25 answers", md.split("## Threshold")[1].split("##")[0])


# ---------- both: decide() ----------

class DecideTests(unittest.TestCase):
    def test_missing_null_and_refused_answers(self):
        for a in (None, {}, "x", {"type": "noul"}, {"type": "noul", "noul": None}, {"type": "choice", "choice": None},
                  {"type": "score"}, {"type": "box", "box": None}, {"type": "box", "box": {"top": 1}},
                  {"type": "noul", "noul": 0.9, "refused": True}, {"refusal": "cannot answer"}):
            self.assertEqual(bench.decide(a), (None, None), a)

    def test_score_uses_the_modal_level(self):
        self.assertEqual(bench.decide({"type": "score", "score": 0.8, "probabilities": {"0": 0.8, "4": 0.2}}), (0, 0.8))
        self.assertEqual(bench.decide({"type": "score", "score": 0.8, "probabilities": [0.8, 0, 0, 0, 0.2]}), (0, 0.8))
        self.assertEqual(bench.decide({"type": "score", "score": 1.7}), (2, None))


# ---------- box support, model selection, plan checks ----------

class ModelTests(Workspace):
    def test_box_support_parses_the_jev_version(self):
        self.assertTrue(bench.answers_box("typesafe/jev-1.14"))
        self.assertTrue(bench.answers_box("typesafe/jev-1.14:nitro"))
        self.assertTrue(bench.answers_box("typesafe/jev-2.0"))
        self.assertFalse(bench.answers_box("typesafe/jev-1.2"))
        self.assertFalse(bench.answers_box("typesafe/jev-1.13"))
        self.assertFalse(bench.answers_box("typesafe/jevx-1.20"))
        self.assertFalse(bench.answers_box("~typesafe/jev-latest"))
        live = {"~typesafe/jev-latest": dm("~typesafe/jev-latest", alias="typesafe/jev-1.14")}
        self.assertTrue(bench.answers_box("~typesafe/jev-latest", live))

    def test_main_ranking_uses_questions_every_model_was_asked(self):
        q = {"q1": {"type": "noul"}, "b": {"type": "box"}}
        items = [{"id": "a", "expected": {"q1": True, "b": [0.1, 0.5, 0.1, 0.5]}}]
        wrong_box = {"type": "box", "box": {"top": 0.6, "bottom": 0.9, "left": 0.6, "right": 0.9}}
        res = make_res(q, items, [call("typesafe/jev-1.14", "a", {"q1": YES, "b": wrong_box}), call("o/m", "a", {"q1": YES})],
                       {"typesafe/jev-1.14": {"questions": ["q1", "b"], "planned": 1}, "o/m": {"questions": ["q1"], "planned": 1}})
        rows = {r["model"]: r for r in bench.summarise(res)}
        self.assertEqual(rows["typesafe/jev-1.14"]["acc"], 1.0)
        self.assertEqual(rows["o/m"]["acc"], 1.0)
        md = bench.write_report(res, os.devnull)
        self.assertIn("0 %", md.split("## Accuracy per question")[1])
        self.assertIn("not asked", md.split("## Accuracy per question")[1])

    def test_model_with_no_answerable_question_is_skipped(self):
        q = {"b": {"type": "box", "instructions": "damage in `image`"}}
        png(self.path("p.png"), 400, 400)
        qs = self.write("questions.json", questions=q)
        items = self.write("items.jsonl", items=[{"id": "p", "image": "p.png"}])
        api = FakeAPI([dm("o/m"), dm("typesafe/jev-1.14")])
        code, out = self.main(api, "plan", qs, items)
        self.assertIn("skip o/m", out)
        self.assertIn("run  typesafe/jev-1.14", out)

    def test_baseline_gets_noul_criteria(self):
        s = bench.llm_schema({"x": {"type": "noul", "instructions": "I", "criteria": {"true": "T yes", "false": "F no"}}})
        self.assertIn("T yes", s["properties"]["x"]["description"])
        self.assertIn("F no", s["properties"]["x"]["description"])

    def test_plan_checks_provider_image_limits(self):
        for n in range(5):
            png(self.path(f"p{n}.png"), 400, 400)
        png(self.path("small.png"), 256, 256)  # 65,536 px, below Jev's 78,400
        qs = self.write("questions.json", questions={"s": {"type": "noul", "instructions": "`image1` is blurry"}})
        items = self.write("items.jsonl", items=[{"id": "a", "images": [f"p{n}.png" for n in range(5)]},
                                                 {"id": "b", "image": "small.png"}])
        api = FakeAPI([dm("cloudflare/clef"), dm("typesafe/jev-1.14"), dm("o/m")])
        code, out = self.main(api, "plan", qs, items)
        self.assertIn("skip cloudflare/clef", out)
        self.assertIn("skip typesafe/jev-1.14", out)
        self.assertIn("run  o/m", out)

    def test_plan_checks_the_baseline(self):
        png(self.path("p.png"), 400, 400)
        qs = self.write("questions.json", questions={"s": {"type": "noul", "instructions": "`image` is blurry"}})
        items = self.write("items.jsonl", items=[{"id": "a", "image": "p.png"}])
        api = FakeAPI([dm("o/m")], [llm("x/nojson", params=("max_tokens",))])
        code, out = self.main(api, "plan", qs, items, "--baseline", "x/nojson")
        self.assertNotEqual(code, 0)
        self.assertIn("structured outputs", str(code) + out)
        api = FakeAPI([dm("o/m")], [llm("x/textonly", inputs=("text",))])
        code, out = self.main(api, "plan", qs, items, "--baseline", "x/textonly")
        self.assertNotEqual(code, 0)
        self.assertIn("image", str(code) + out)


# ---------- round 2 ----------

class Round2ReportTests(unittest.TestCase):
    def test_threshold_is_printed_without_rounding_below_it(self):
        items = [{"id": str(i), "expected": {"q1": True}} for i in range(50)]
        calls = [call("m", str(i), {"q1": {"type": "noul", "noul": 0.99, "confidence": 0.9954}}) for i in range(25)]
        calls += [call("m", str(i), {"q1": {"type": "noul", "noul": 0.2, "confidence": 0.9951}}) for i in range(25, 50)]
        res = make_res(Q1, items, calls, {"m": {"questions": ["q1"], "planned": 50}})
        md = bench.write_report(res, os.devnull)
        self.assertIn("confidence ≥ 0.9954 keeps", md)

    def test_invalid_answers_are_noted_in_errors_and_never_covered(self):
        q = {"team": {"type": "choice", "criteria": {"billing": "b", "shipping": "s"}}, "u": {"type": "score", "criteria": ["a", "b"]}}
        items = [{"id": "a", "expected": {"team": "billing", "u": 1}}, {"id": "b", "expected": {"team": "billing", "u": 1}}]
        calls = [call("m", "a", {"team": {"type": "choice", "choice": "bogus", "confidence": 0.99},
                                 "u": {"type": "score", "score": 7, "confidence": 0.99}}),
                 call("m", "b", {"team": {"type": "choice", "choice": "billing", "confidence": 0.99}})]
        res = make_res(q, items, calls, {"m": {"questions": ["team", "u"], "planned": 2}})
        row = bench.summarise(res)[0]
        self.assertAlmostEqual(row["cov@0.9"], 1 / 4)
        errors = bench.write_report(res, os.devnull).split("## Errors")[1]
        self.assertIn("missing or invalid", errors)
        self.assertIn("a / team", errors)


class Round2InputTests(Workspace):
    def test_reordering_choice_criteria_changes_the_fingerprint(self):
        it = [{"id": "a", "text": "t", "state": None, "images": []}]
        q1 = {"team": {"type": "choice", "criteria": {"billing": "b", "other": "o"}}}
        q2 = {"team": {"type": "choice", "criteria": {"other": "o", "billing": "b"}}}
        self.assertNotEqual(bench.fingerprint(q1, it), bench.fingerprint(q2, it))

    def test_negative_price_baseline_is_rejected_and_never_reserves_negative(self):
        q, items = self.files(1)
        bad = llm("x/variable")
        bad["pricing"]["completion"] = "-1"
        code, out = self.main(FakeAPI([dm("a/m")], [bad]), "plan", q, items, "--baseline", "x/variable")
        self.assertNotEqual(code, 0)
        self.assertIn("price", str(code) + out)
        it = {"id": "a", "text": "t", "images": []}
        self.assertGreaterEqual(bench.rough_cost(dm("a/m", prompt="-1"), {}, it), 0)

    def test_run_checks_the_key_before_any_request(self):
        q, items = self.files(1)
        api = FakeAPI([dm("a/m")], on_decision=yes_to_all)
        for env in ({}, {"OPENROUTER_API_KEY": "sk-or-v1-abc\ndef"}):
            with mock.patch.dict(os.environ, env, clear=True):
                code, out = self.main(api, "run", q, items, "--models", "a/m", "--max-usd", "1",
                                      "--out", self.path("results.json"))
            self.assertNotEqual(code, 0)
            self.assertIn("OPENROUTER_API_KEY", str(code))
        self.assertEqual(api.posts, [])


class Round2SaveTests(Workspace):
    def test_checkpoints_are_compact_and_throttled(self):
        q, items = self.files(5)
        saves, real = [], bench.save_atomic

        def counting(res, out):
            saves.append(len(res["calls"]))
            real(res, out)
        api = FakeAPI([dm("a/m")], on_decision=yes_to_all)
        with mock.patch.object(bench, "save_atomic", counting), mock.patch.object(bench, "_now", lambda: 100.0):
            self.main(api, "run", q, items, "--models", "a/m", "--passes", "1", "--max-usd", "1",
                      "--workers", "1", "--out", self.path("results.json"))
        self.assertEqual(saves[0], 1)  # the first finished call is saved at once
        self.assertLess(len(saves), 5)  # then at most once per second, plus the final save
        self.assertEqual(saves[-1], 5)
        self.assertNotIn("\n ", self.read("results.json"))

    def test_failed_checkpoint_warns_and_the_run_continues(self):
        q, items = self.files(3)
        real, n = bench.save_atomic, [0]

        def flaky(res, out):
            n[0] += 1
            if n[0] == 1:
                raise OSError("disk full")
            real(res, out)
        api = FakeAPI([dm("a/m")], on_decision=yes_to_all)
        err = io.StringIO()
        with mock.patch.object(bench, "save_atomic", flaky), contextlib.redirect_stderr(err):
            code, _ = self.main(api, "run", q, items, "--models", "a/m", "--passes", "1", "--max-usd", "1",
                                "--workers", "1", "--out", self.path("results.json"))
        self.assertEqual(code, 0)
        self.assertIn("disk full", err.getvalue())
        self.assertEqual(len(self.results()["calls"]), 3)


if __name__ == "__main__":
    unittest.main()
