# METC Human Evaluation — Span & Label Quality Annotation Guidelines

> **Editorial note for the public release.** This is the annotation protocol used for the
> paper's English and German human evaluation. Sections 1–6 are the instructions as given to
> annotators, except that references to internal source files now point to their public
> equivalents in this repository. Section 7 replaces the internal planning notes with the
> scope of the study as it was actually run. The paper reports Task 1 only, and only Task 1
> ratings are released, in `metc-data/human/` inside the `metc-data.zip` release asset.

## 1. Project Overview

We use an LLM-based judge to rate the quality of error-correction edits — for each
edit, how well its **span** (the text it touches) and its **error-type description**
(what kind of error it claims to fix) match the actual correction. This annotation
project collects human judgments on the same edits so we can:

1. Establish a human ground truth for span quality and label quality.
2. Compare the LLM judge against human judgment.
3. Identify where and why the judge diverges from humans, in order to improve it.

There are two tasks:

- **Task 1 — Your independent judgment.** You rate each edit on two axes (span
  quality, label quality). Your rating is the ground truth.
- **Task 2 — Reviewing the LLM judge.** After Task 1, you see the LLM judge's
  verdict and reasoning, and rate how acceptable they are.

Some annotators may do only Task 1.

---

## 2. What You Will See

For each item:

- **Language** — the language of the sentence.
- **Original sentence** — the sentence as written, before correction.
- **Corrected sentence** — the sentence after the correction is applied.
- **The edit** — the specific change being judged, shown as `'original' => 'corrected'`.
  Special forms:
  - `(insert) => 'text'` — text was inserted (nothing was there before).
  - `'text' => (delete)` — text was removed.
  - `(insertion; see corrected sentence above)` — an insertion whose exact target
    appears in the corrected sentence.
- **Error-type description** — one plain-English sentence stating what kind of error
  this edit claims to fix (e.g. *"Problem with a noun's number."*).

> **The description is deliberately phrased in plain language and does not reveal which
> system produced it.** Judge it on its own terms — *does this description fit this
> edit?* — not on which tool you think wrote it.

---

## 3. Workflow

1. Read the original sentence, corrected sentence, and the edit carefully.
2. Assign **span quality** (Good / Acceptable / Bad).
3. Assign **label quality** (Good / Acceptable / Bad).
4. Write a brief one-sentence reason for each axis, citing the span or the description.
5. Select your confidence.
6. Submit. *(Task 2 only:)* the LLM judge's verdict and reasoning are then revealed.
7. *(Task 2 only:)* rate the acceptability of the judge's verdict and the soundness /
   completeness of its reasoning.

> **Do not revise your Task-1 judgment after seeing the LLM output. Your independent
> judgment is the ground truth for this evaluation.**

---

## 4. Task 1 — Your Judgment

The two axes are **independent**: a clean span can carry a wrong description, and a
correct description can sit on a messy span. Rate them separately.

### 4.1 Span quality

Does the edit's span — the stretch of original text it touches — cleanly isolate the
underlying error?

- **Good** — the span exactly covers the error, no more, no less.
- **Acceptable** — the span covers the error but is imprecise: slightly too wide or too
  narrow, or semantically confusing (it isn't obvious what single error it isolates).
- **Bad** — the span includes unrelated text that makes it unclear what error the edit
  is fixing, or it misses the error.

### 4.2 Label quality

Is the error-type description the right description for this edit?

- **Good** — the description is the most precise correct interpretation of this edit.
- **Acceptable** — the description is a coarser super-type that still applies (correct,
  but a more specific description exists).
- **Bad** — the description is wrong, irrelevant, or contradicts what the edit does.

> Ratings are recorded on the three-point scale above. If a binary view is needed
> downstream, collapse to **Good vs. not-Good**.

### 4.3 Confidence

- **High** — the error and the correctness of the span / description are clear-cut.
- **Medium** — some ambiguity, but you can justify your judgment.
- **Low** — insufficient context or multiple plausible readings. Still choose a rating.

### 4.4 Unannotatable

If you cannot judge an item, mark it **unannotatable** and give a reason: too offensive,
hard to comprehend, corrupted text, too long / complex, or other.

### 4.5 Worked examples

All examples below are real items, with the descriptions shown exactly as annotators see
them (system-blind).

**Example A — both axes Good**
- Original: *People get certain disease because of genetic changes.*
- Corrected: *People get certain diseases because of genetic changes.*
- Edit: `'disease' => 'diseases'`
- Description: *"Problem with a noun's number."*
- **Span = Good** — the edit changes only the noun *disease*, exactly where the error is.
- **Label = Good** — singular→plural is precisely a noun-number error; this is the most
  precise description.

**Example B — same edit, coarser description → label Acceptable**
- Edit: `'disease' => 'diseases'`
- Description: *"Problem with grammatical agreement (number, person, or gender …)."*
- **Span = Good** — same clean span.
- **Label = Acceptable** — a number error is a defensible *broad* reading, but "noun
  number" would be more precise, so the description is coarser than ideal.

**Example C — label Bad (wrong description)**
- Original: *Genetic risk refers more to your chance of inheriting a disorder or disease.*
- Corrected: *Genetic risk refers to your chance of inheriting a disorder or disease.*
- Edit: `'more' => (delete)`
- Description: *"Problem with an article or determiner (missing, extra, or wrong)."*
- **Span = Good** — the edit cleanly deletes only the extra word *more*.
- **Label = Bad** — deleting *more* removes an unnecessary adverb / comparative, not an
  article or determiner; the description is simply wrong.

**Example D — span Acceptable (imprecise / confusing span)**
- Original: *Here I want to share Forest'view on this issue.*
- Corrected: *Here I want to share Forests'sview on this issue.*
- Edit: `"Forest'view" => "Forests'sview"`
- Description: *"Problem with a proper noun."*
- **Span = Acceptable** — the span captures the malformed token, but it conflates the
  name with the following word and an apostrophe / spacing issue, so it isn't a clean
  single-error span.
- **Label = Bad** — the change is mainly a possessive / spelling / spacing fix, not
  specifically a proper-noun choice.

**Example E — span Bad (includes unrelated text)**
- Original: *This knowledge maybe relavant to them, and vice versa.*
- Corrected: *This knowledge may be relevant to them or not.*
- Edit: `'relavant' => 'be relevant'`
- Description: *"A single correction that fixes two or more distinct kinds of errors at
  once …"*
- **Span = Bad** — *relavant* is only a misspelling, but the edit pulls in unrelated
  *be* from the preceding *maybe*, so the span no longer isolates a single error.
- **Label = Bad** — this conflates a spelling fix with a separate word-split / modal
  fix; it is not one edit fixing multiple error types within one span.

### 4.6 Quick reference — decision trees

**Span quality**
1. Does the span isolate exactly the error, nothing extra or missing? → **Good**
2. Does it cover the error but is slightly wide / narrow or confusing? → **Acceptable**
3. Does it include unrelated text, or miss the error? → **Bad**

**Label quality**
1. Does the description name the right *kind* of error for this edit?
   - No → **Bad**
   - Yes → step 2
2. Is it the *most precise* correct description, or only a coarser super-type?
   - Most precise → **Good**
   - Coarser but still valid → **Acceptable**

---

## 5. Task 2 — Reviewing the LLM Judge's Verdict

After you submit Task 1, you see the LLM judge's span / label verdicts and its
one-sentence reason for each. Rate two things.

### 5.1 Judgment acceptability

How acceptable is the judge's rating, according to these guidelines?

- **Critical error** — the judgment is embarrassingly or absurdly wrong.
- **Major flaw** — definitely incorrect and the mistake is significant (it overlooked
  something major), though not embarrassing.
- **Minor flaw** — a small but understandable mistake led to the wrong judgment
  ("accidental").
- **Acceptable** — different from what you'd have chosen, but defensible; you could be
  convinced, or the case is genuinely subjective / hard.
- **Ideal** — aligns with your own view.

### 5.2 Reasoning soundness

Judge the judge's *reasoning* — not whether it cited every piece of evidence. (If the
reasoning is missing some evidence but is otherwise sound, that belongs under
completeness below, and the soundness can still be Acceptable.)

- **Critical error** — the reasoning is embarrassingly bad.
- **Major flaw** — definitely incorrect, a major mistake in the reasoning.
- **Minor flaw** — a small, understandable mistake.
- **Acceptable** — different from how you'd think about it, but defensible.
- **Ideal** — the reasoning is flawless.

### 5.3 Reasoning completeness

- **Missing major evidence** — crucial evidence from the edit / sentence is absent.
- **Missing minor evidence** — some non-crucial evidence is left out.
- **Nothing missing** — all relevant evidence is discussed.

If evidence is missing, briefly say what should have been included.

---

## 6. Error-type descriptions (reference)

You judge the **plain-English description** shown with each edit, never a system code.
Descriptions are generated from each system's native error-type label and rewritten in a
uniform, end-user voice, so the same underlying category reads identically regardless of
which tool produced it — this is what keeps the comparison fair and system-blind. You do
not need to know the underlying taxonomy; only whether the description, as written, fits
the edit.

A few examples of how descriptions read:

- *"Problem with a noun's number."*
- *"Problem with an article or determiner."*
- *"A word is misspelled."*
- *"Problem with the formality of the wording (the sentence stays grammatically correct)."*
- *"A single correction that fixes two or more distinct kinds of errors at once."*

The released set of descriptions is in `evaluation/label_glosses.json`.

---

## 7. Scope of the study as run

- **English:** CoNLL-2014, a system-blind sample of 800 items (200 per system: reference,
  METC, ERRANT, jp_errant), each assigned independently to two of twelve professional
  annotators (1,600 assignments).
- **German:** MERLIN, the first 200 items of a deterministically shuffled pool with 200
  items per system (reference, METC, gERRANT, jp_errant), each assigned to two annotators
  (400 assignments).

See the paper's human-evaluation section for results and agreement figures.

> This guideline mirrors the rubric of the LLM judge it validates
> (`prompts/acceptability_judge.py`); keep the two in sync if either changes.
