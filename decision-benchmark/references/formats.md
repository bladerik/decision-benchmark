# Input and output formats

## questions.json

The Decisions API `questions` object, sent as written. Keys are your question names.

```json
{
  "is_spam": {
    "type": "noul",
    "instructions": "The `text` is unsolicited advertising or a scam, not a message from a real customer about their order."
  },
  "team": {
    "type": "choice",
    "instructions": "Which team should handle the request in `text`?",
    "criteria": {
      "billing": "Invoices, payments, refunds.",
      "shipping": "Delivery times, tracking, lost parcels.",
      "product": "How the product works, defects, returns for a defect."
    }
  },
  "urgency": {
    "type": "score",
    "instructions": "How urgent is the request in `text`?",
    "criteria": ["Can wait a week", "Needs an answer today", "Customer is blocked right now"]
  },
  "damage": {
    "type": "box",
    "instructions": "the damaged area of the product in `image`"
  }
}
```

- `noul`: yes/no. Optional `criteria: {"true": "...", "false": "..."}` sharpens both sides (the LLM baseline gets them too).
- `choice`: `criteria` is an object option → description (not `options`).
- `score`: `criteria` is a list of levels, lowest first.
- `box`: a description of the target, not a command, no `criteria`. Only `typesafe/jev-1.14` and newer answer it (see Image limits below), and it always returns a box, so pair it with a noul "is the target there?".

## items.jsonl (one JSON object per line)

```json
{"id": "m01", "text": "Hi, my parcel has not arrived since Monday...", "expected": {"is_spam": false, "team": "shipping", "urgency": 1}}
{"id": "p07", "image": "images/p07.jpg", "expected": {"damage": [0.42, 0.80, 0.10, 0.55]}}
{"id": "x3", "state": {"subject": "Invoice 2291", "body": "..."}, "expected": {"team": "billing"}}
```

- `id` must be unique (after trimming spaces); `plan` rejects duplicates. Without an id, a JSONL item gets its line number and a CSV row its row number, which must not clash with explicit ids either.
- `text`, `state` (a JSON object, for structured input) and `image` / `images` (PNG or JPEG paths, relative to the items file) can be combined. Refer to them in questions as `text` and `image` (`image1`, `image2`… with several images).
  - Text and state without images are sent as one object: the state's fields plus `text`, i.e. `{"subject": "Invoice 2291", "body": "...", "text": "customer message"}`. When the state already has a `text` field, the item's `text` replaces it (`plan` warns).
  - With images, the state (as JSON), the text and the images are sent as a list of parts.
- `expected` holds the gold answers; leave out questions you have no answer for (they are asked but not scored).
  - noul: `true`/`false` (also yes/no, 1/0)
  - choice: the option key
  - score: the level index (0 = first level), a whole number; `1.5` is rejected
  - box: `[top, bottom, left, right]` as fractions 0 to 1 of the image height/width, with top < bottom and left < right; counts as right when it overlaps the model's box by at least half (IoU ≥ 0.5)

## items.csv

Columns `id`, `text`, `image` (several paths separated by `|`), plus one column per question name holding the expected answer. Empty cell = no gold answer.

```csv
id,text,image,is_spam,team
m01,"Hi, my parcel has not arrived since Monday...",,no,shipping
```

## Image limits per model

`plan` and `run` reject images outside 256 to 1568 px per side, and skip a model whose provider limit an item breaks (the reason is printed as `skip <model>: ...`). The limits come from [OpenRouter's multimodal decisions guide](https://openrouter.ai/docs/guides/community/multimodal-decisions#limits); the model list does not expose them, so check the guide when a provider changes.

| Models | Limit |
|---|---|
| Jev (`typesafe/jev-*`) | PNG or JPEG only; each side 256 to 1568 px; 78,400 to 2,458,624 pixels per image (a 256×256 image is too small); at most 16 MiB per file |
| Clef, Clef Flash, Clef Omni (`cloudflare/clef*`) | at most 4 images; keep each image under ~300 KB (`plan` warns); only about the first 2,000 tokens of text are read |
| Liquid d1, d1-ultra (`liquid/*`) | at most 8 images; request body at most 4.5 MB |
| Morph, Inception (Mercury Decide), Baseten (Elephant, Mammoth, Joey) | at most 16 images |
| GPT-6 Luna Decisions | at most 128 images |
| ScaleDown | one image (document) per request |

Box questions go only to Jev 1.14 and newer (the version is read from the id, e.g. `typesafe/jev-1.14:nitro`; `~typesafe/jev-latest` counts when the live list resolves it to such a version). Other models and the LLM baseline get the remaining questions; a model left with no question is skipped.

`--baseline` must be a chat model that lists structured outputs (`response_format`) and, when items have images, image input; `plan` stops otherwise. Its answer length is capped (`max_tokens` = 256 + 64 per question).

## Outputs

- `results.json`: every request with the raw answers, seconds (including retries and backoff), cost, and the charge counted against `--max-usd`. It is saved (compact JSON) after the first request, then at most once a second while running, and always when the run ends or is interrupted; a failed save prints a warning and the run continues. It also stores a fingerprint of the questions and items (text, state, image file contents; not the labels), each model's asked questions and planned request count, and `stopped` (`cap`, `interrupted` or null).
  - `run` with the same `--out` resumes: finished calls are kept, failed ones retried, calls for models or passes no longer requested are dropped (their cost stays in the total). Labels may change between runs.
  - When the questions or items changed, `run` refuses to resume and asks for a new `--out`, e.g. `--out results-v2.json`. Files from older bench.py versions have no fingerprint and are refused the same way.
- `--max-usd`: before each request the run reserves twice its rough estimate; a request starts only while spent + reserved + its reservation stays within the cap, so parallel requests cannot run past it on estimates. A baseline whose list price is variable (shown as -1) is rejected by `plan`. A response without `usage.cost` is counted at its reservation and the report says how many. Failed requests that were billed count too. The cap counts the cumulative spend recorded in that results file across reruns, so finishing a capped run needs a higher `--max-usd` than the amount already spent. The cap is still estimate-based; for a strict guarantee set a credit limit on the OpenRouter key.
- `results.md`: the report.
  - A partial run (cap hit, interrupted, or not every planned request done) is flagged at the top, with completed/planned requests; each model's row shows `Requests done`.
  - Accuracy counts every labelled (item, question, pass) a model was asked; failed requests and missing, refused or malformed answers count as wrong and as not covered. The main table uses only the questions every model was asked, so models are ranked on the same task; box questions (Jev only) and other questions some model skipped appear in the per-question table, with `not asked` for the rest.
  - Score answers are judged by the most likely level when the model returns level probabilities, else by the rounded score.
  - Threshold section: the lowest observed confidence (any value up to 1.0) whose kept answers reach the target accuracy, with at least 20 kept answers; coverage is the share of all labelled answers asked.
  - Then accuracy per question (correct/asked), items most models got wrong, and errors: failed requests, plus answers missing or invalid (an option not in `criteria`, a level off the scale, a malformed box) in otherwise successful requests. Those count as wrong and never as covered. The API key is redacted from every error.
  - `report results.json --target 0.98` rebuilds it without new requests.
