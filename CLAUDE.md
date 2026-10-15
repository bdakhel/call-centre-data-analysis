# CLAUDE.md

Guidance for Claude (and any other contributor, human or AI) working in this repository.

## What this project is

`utility-cx-insights` is a **portfolio MVP** that simulates a customer experience (CX) /
contact centre analytics pipeline for a Canadian **natural gas and electricity utility**
operating in **Alberta**. It exists to demonstrate data engineering, NLP sentiment analysis,
and dashboarding skills (Python, pandas, transformers, Power BI) using a realistic-looking
but entirely fictional dataset.

This is a **portfolio / demo project, not a real company system**. There is no real utility
client, no real customer data, and no production deployment target.

## Non-negotiable rules

1. **All data is synthetic.** Every customer record, call transcript, ticket, name, phone
   number, account number, and complaint must be generated (e.g. via `Faker`, templates, or
   sampling) or clearly derived from public/open synthetic sources. Never ingest, copy, or
   reference real customer data, real call logs, or real PII of any kind.

2. **Never use real company names, logos, or branding.** Do not reference real Alberta or
   Canadian utilities (e.g. do not name actual gas/electric providers) anywhere in code,
   data, docs, notebooks, or the Power BI report. Use an obviously fictional utility name
   (e.g. "Prairie Sky Utilities", "Foothills Energy Co." — pick one and keep it consistent)
   and a generic/placeholder logo or no logo at all. Do not use any real corporate visual
   identity (colors/fonts copied from a real brand).

3. **Never use real people.** All customer names, agent names, and quotes must come from a
   synthetic generator (`Faker`) or be clearly invented. Do not use names of real public
   figures, coworkers, or anyone identifiable. No real email addresses or phone numbers.

4. **Reproducibility: use `random seed = 42` everywhere.** Any code path that uses
   randomness (data generation with Faker/numpy/random, train/test splits, model
   initialization, sampling, shuffling) must set and use seed `42` so results are
   reproducible across runs. Set it at the top of scripts/notebooks, e.g.:
   ```python
   import random
   import numpy as np
   random.seed(42)
   np.random.seed(42)
   ```
   and pass `random_state=42` to scikit-learn functions, `Faker.seed(42)`, etc.

5. **Code must be simple, commented, and runnable on a laptop CPU.**
   - Prefer straightforward, readable code over clever abstractions.
   - Add short comments explaining *why*, not just *what*, especially around data
     generation assumptions and modeling choices.
   - No GPU-only code paths. If using `transformers`/`torch`, default to small CPU-friendly
     models (e.g. distilled sentiment models) and keep batch sizes/dataset sizes modest so
     the whole pipeline runs on a typical laptop in a reasonable time.
   - Avoid heavyweight infrastructure (no databases, cloud services, or GPU clusters
     required to run this project end to end).

## Project structure

```
utility-cx-insights/
├── data/
│   ├── raw/          # synthetic raw data (generated, not real)
│   ├── processed/    # cleaned/feature-engineered data
│   └── labels/       # sentiment/intent labels for synthetic records
├── src/              # pipeline source code (data gen, cleaning, modeling, scoring)
├── notebooks/        # exploratory analysis / prototyping notebooks
├── powerbi/          # Power BI report files (.pbix) and supporting exports
├── docs/             # project documentation
├── tests/            # pytest unit tests
├── requirements.txt
└── CLAUDE.md
```

## Domain context (for realistic-but-fictional data)

- Service types: natural gas, electricity, dual-fuel accounts.
- Region: Alberta (e.g. Calgary, Edmonton, and smaller AB towns) — use plausible Alberta
  place names, but a fictional utility brand.
- Typical contact centre themes to simulate: billing disputes, high bill complaints,
  outage reports, move-in/move-out, payment plans, meter reading issues, rebate/efficiency
  program questions, winter heating cost concerns, storm/outage season spikes.
- Channels: phone call transcripts, chat transcripts, email/web ticket text.

## Working conventions

- Keep scripts idempotent: re-running data generation should produce the same synthetic
  dataset given seed 42.
- Document any assumption about data distributions (e.g. complaint category mix, seasonal
  call volume patterns) inline as a comment where it's implemented.
- Keep notebooks for exploration; promote reusable logic into `src/` modules.
- Tests in `tests/` should cover data generation invariants (e.g. no nulls in required
  fields, valid sentiment label ranges) and any non-trivial transformation logic.
