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

- `noul`: yes/no. Optional `criteria: {"true": "...", "false": "..."}` sharpens both sides.
- `choice`: `criteria` is an object option → description (not `options`).
- `score`: `criteria` is a list of levels, lowest first.
- `box`: a description of the target, not a command, no `criteria`. Only `typesafe/jev-1.14` and newer answer it, and it always returns a box, so pair it with a noul "is the target there?".

## items.jsonl (one JSON object per line)

```json
{"id": "m01", "text": "Hi, my parcel has not arrived since Monday...", "expected": {"is_spam": false, "team": "shipping", "urgency": 1}}
{"id": "p07", "image": "images/p07.jpg", "expected": {"damage": [0.42, 0.80, 0.10, 0.55]}}
{"id": "x3", "state": {"subject": "Invoice 2291", "body": "..."}, "expected": {"team": "billing"}}
```

- `text`, `state` (any JSON object, for structured input) and `image` / `images` (PNG or JPEG paths, relative to the items file) can be combined. Refer to them in questions as `text` and `image` (`image1`, `image2`… with several images).
- `expected` holds the gold answers; leave out questions you have no answer for (they are asked but not scored).
  - noul: `true`/`false` (also yes/no, 1/0)
  - choice: the option key
  - score: the level index (0 = first level)
  - box: `[top, bottom, left, right]` as fractions of the image height/width; counts as right when it overlaps the model's box by at least half (IoU ≥ 0.5)

## items.csv

Columns `id`, `text`, `image` (several paths separated by `|`), plus one column per question name holding the expected answer. Empty cell = no gold answer.

```csv
id,text,image,is_spam,team
m01,"Hi, my parcel has not arrived since Monday...",,no,shipping
```

## Outputs

- `results.json`: every request with the raw answers, seconds and cost. `run` resumes from it; use `--out results-v2.json` after changing questions or labels.
- `results.md`: the report (main table, threshold for the target accuracy, accuracy per question, items most models got wrong, errors). `report results.json --target 0.98` rebuilds it without new requests.
