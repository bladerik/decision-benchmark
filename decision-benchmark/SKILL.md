---
name: decision-benchmark
description: Decision models (Jev, GPT Luna Decisions, Drex, Clef and the other OpenRouter models that answer with a typed choice and a probability instead of text) - decide whether one fits a task, benchmark every live one on the user's own labelled sample, and recommend a model, a confidence threshold and a fallback. Use when someone considers a decision model, wants to automate a yes/no check, classification or routing step, or asks which decision model suits their case.
---

# Decision benchmark

A decision model reads a state (text, JSON, images) and answers typed questions: a yes/no probability, one of fixed options, a level on a scale, or a box in an image. It returns no text, answers in under a second and costs fractions of a cent. It fits only some decisions. This skill finds out whether it fits the user's case and, if so, which model wins on their own data.

Ask one question at a time; keep the type names noul, choice, score and box untranslated. Create a folder `decision-benchmark-<task>/` in the current directory, keep the user's files there and run every command below from it. `<skill>` below is this skill's folder; where `python3` is missing (often on Windows), use `py` or `python`.

## 1. Find the need

Open with "What are you considering a decision model for?" Expect a vague answer. Ask about the process around it, and research their domain or tools when that helps, until you know: what comes in (text, image, both), which answer comes out, how many items a day, what happens next with the answer, what a wrong answer costs, and whether the items are sensitive (client data, contracts, health, finance, anything under NDA).

The benchmark sends the sample through OpenRouter to several model providers. For sensitive items, say so plainly and let the user choose: anonymise the sample with placeholders, or stop here.

Done when the user confirms your one-sentence restatement ("For each <input>, decide <answer from a fixed set>, then <next step>.") and has chosen how to handle sensitive data.

## 2. Fit verdict

A decision model is System 1: you look and immediately know. Anything that needs System 2, thinking, belongs to an LLM or an agent.

| System 1: decision model | System 2: LLM or agent |
|---|---|
| Spam or not | Counting, arithmetic |
| Which of 30 categories | Comparing dates, deadlines relative to today |
| Product colour from a fixed list | Multi-step reasoning, combining several facts |
| Does the message mention an appointment | Extracting or writing text (the date itself, a summary, a reply) |
| Is the photo blurry; where is the damage (box) | Answers with no fixed set |

Split the task into its decisions, mark each one, and give the verdict with the reason:

- **Fits:** continue.
- **Partly fits:** propose a pipeline where the decision model triages and an agent does the hard part only for the items that need it. Example: the decision model answers "does the message contain an appointment?"; only yes-items go to an agent that turns "z utorka na štvrtok" into 10. 11. → 12. 11. given today's date. Benchmark the decision-model part.
- **Does not fit:** warn, explain which part needs System 2, and suggest the better tool, usually an LLM with JSON output (structured outputs). Continue only when the user explicitly asks to benchmark anyway.

Done when the user has the verdict and has agreed on what to benchmark, or has stopped.

## 3. Design the questions

Write `questions.json`; the format with an example of each type is in [references/formats.md](references/formats.md). Types: **noul** (yes/no with a probability), **choice** (one of fixed options, with probabilities), **score** (a level on an ordered scale), **box** (locate a target in an image; Jev 1.14+ only).

Each rule comes from a real failure:

- **One reading only.** Name the input the question is about (`text`, `image`) and state the condition as an observable fact. Real case: "text max 4 words or none?" was read as "is there no text"; OpenAI's decision model was 100 % confident and wrong 12 of 12 times. Rephrased as "Count the words of all visible text. The total is 4 words or fewer (0 words also counts as yes)." it got 13 of 15.
- **One decision per question.** "Urgent and about billing?" becomes two questions.
- **Maths in code.** Ask for the fact ("the invoice shows a due date"), and let code count, compare and compute.
- **Option order.** Models lean toward the first option; put a catch-all or default option last and describe every option concretely.
- **Lean state.** Send only what the decision needs; large irrelevant context lowers accuracy and raises cost.

Show the questions to the user. Done when the user approves every one.

## 4. Labelled sample

Help the user gather 50 to 100 items whose right answer they already know: hand-sorted history, a folder per category, a spreadsheet column. Mix in borderline cases in roughly real proportions, with at least ~10 items for each answer. When labels are missing, propose labels and let the user confirm or correct each; the user's answer is the gold label. Images must be PNG or JPEG, 256 to 1568 px per side; Jev also needs at least 78,400 pixels per image (about 280×280), so aim for ~512 px or more on the short side; ideally under 300 KB (resize with whatever is installed: `sips -Z 1024` on macOS, ImageMagick, Pillow). Apply the data choice from step 1.

Write `items.jsonl` or `items.csv` ([references/formats.md](references/formats.md)). Done when `plan` (step 6) reports no problems.

## 5. Gateway and key

The script calls OpenRouter's Decisions API. The user needs an OpenRouter account with a few euros of credit (https://openrouter.ai/settings/credits; the benchmark itself usually costs cents) and a key from https://openrouter.ai/settings/keys. Another gateway works only if it serves the same Decisions API.

The key lives only in the environment variable `OPENROUTER_API_KEY`, which the user sets in their own terminal before starting the agent from it: `export OPENROUTER_API_KEY=...` (macOS, Linux) or `$env:OPENROUTER_API_KEY="..."` (PowerShell). Check with `python3 <skill>/scripts/bench.py models`: it lists the live decision models and shows whether the key is set and the remaining credit. The script needs network access and that variable in your shell; when it reports the key missing or cannot reach openrouter.ai although the user set it, ask the user to allow network access for this command or to pass the variable into your sandbox (in Codex: `shell_environment_policy`).

## 6. Run the benchmark

```bash
python3 <skill>/scripts/bench.py plan questions.json items.jsonl --baseline <llm-id>
python3 <skill>/scripts/bench.py run  questions.json items.jsonl --baseline <llm-id> --max-usd 0.50
```

`plan` validates the files, fetches the live decision-model list, keeps every model that accepts the inputs, and estimates the cost without sending anything. For `--baseline` pick one cheap current chat model with structured outputs (and image input when items have images) from https://openrouter.ai/models, so the user sees what a normal LLM does on the same questions. Show the plan and the estimate, and run after the user agrees. `run` makes 2 passes (stability), 4 requests at a time, saves `results.json` as it goes and on exit or interrupt, stops at the `--max-usd` cap, and continues from `results.json` when started again with the same files. The cap counts all spend in that results file, so finishing a capped run needs a higher `--max-usd`. When the questions or items change, it asks for a new `--out`. The cap reserves a cautious estimate per request, so a run may stop a little early; for a hard limit the user sets a credit limit on the key at https://openrouter.ai/settings/keys. `python3 <skill>/scripts/bench.py --help` lists the remaining options.

When the report lists "Items most models got wrong", check those labels and question wordings with the user; after a fix, rerun with `--out results-v2.json`.

Done when `results.md` reports a complete run (every planned request answered or failed) and every failing model fails for a reason you have named (the "Errors" section). A partial run (cap hit, interrupted) gets finished first, by rerunning with a higher cap or after the interruption.

## 7. Report and recommend

Show the main table from `results.md`: accuracy, accuracy and coverage at confidence ≥ 0.9 and ≥ 0.8, median latency, cost per 1 000 items, stability between passes, open weights. Then recommend, in a few sentences:

- **The model:** accuracy first, then cost and speed. A model that changes its answers between passes is a risk.
- **The threshold:** ask what accuracy the case needs and read it from the report's threshold section (`bench.py report results.json --target 0.98` recomputes it). Check each question's own accuracy: a question far below the rest needs better wording or a fallback of its own. The threshold is tuned on this sample, so call it an estimate, and suggest checking it on 20 to 30 fresh items before relying on it.
- **The fallback:** where items below the threshold go, a frontier LLM or agent, or a human, and what share of items that is. An item goes to fallback when any of its questions falls below the threshold, so with several questions per item the share is higher than 1 − coverage of single answers.
- **Local option:** when an open-weights model is close to the best, say it can run locally from its Hugging Face weights, so the data never leaves the user's machine. That setup is worth it now that the benchmark has shown the model works for their case.
