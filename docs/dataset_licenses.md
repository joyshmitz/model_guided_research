# Dataset / Licensing / Attribution Sanity Check

**Bead:** `model_guided_research-wgd`
**Date:** 2026-06-14
**Scope:** datasets and tokenizer assets used by the nanochat training path.
Doc only — no code changes. Flags attribution obligations; not legal advice.

## What data the code actually pulls

There is exactly **one external pretraining corpus** and it is downloaded on
demand:

| Item | Source | Where in code |
|---|---|---|
| Pretraining shards | `karpathy/fineweb-edu-100b-shuffle` on HuggingFace (`shard_00000..01822.parquet`, `text` column) | `BASE_URL`, `MAX_SHARD` — `nanochat/dataset.py:26,36` |
| Local cache | `get_base_dir()/base_data` (here: `~/.cache/nanochat/base_data/`) | `dataset.py:34–35` |
| Download | on-demand HTTP GET with retry/backoff | `dataset.py:113` |

`mgr gen-tasks` / `--data-dir` corpora are **synthetic** (generated in-repo,
e.g. the diagnostics/arith/Dyck tasks) — no external license attaches.

## Licensing chain

- **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`, the parent of the shuffled
  repack) is released by HuggingFace under the **Open Data Commons Attribution
  License v1.0 (ODC-By 1.0)**.
- The **underlying text** is derived from **Common Crawl**; use is also subject
  to the **Common Crawl Terms of Use**.
- `karpathy/fineweb-edu-100b-shuffle` is a **re-packaged / re-shuffled copy**
  of that data for streaming convenience; it inherits the upstream ODC-By 1.0
  terms and the Common Crawl ToU.

**Obligation:** ODC-By 1.0 is permissive (use/share/adapt, including
commercially) **but requires attribution**. If this project redistributes the
data, derived datasets, or publishes results trained on it, it must attribute
FineWeb-Edu (and note Common Crawl as the source). We currently only *download*
it on demand and do not redistribute shards, which is the low-risk path.

> Verify before any redistribution or release: re-check the exact license string
> on the HF dataset cards (`HuggingFaceFW/fineweb-edu` and
> `karpathy/fineweb-edu-100b-shuffle`) at publish time — dataset card terms can
> change, and this note records the state as of 2026-06-14.

## Tokenizer assets

- The active tokenizer here is the **tiktoken GPT-2/GPT-4-style** vocab
  (measured: vocab 50257, BOS = 50256 = `<|endoftext|>`); `tiktoken` is **MIT**
  (OpenAI). See `docs/data_parity.md`.
- The train-your-own-BPE path uses HuggingFace `tokenizers` (**Apache-2.0**) or
  `rustbpe`. The project's own `SPECIAL_TOKENS` (`tokenizer.py:15`) are original.

No tokenizer asset imposes a copyleft or attribution burden beyond standard MIT/
Apache notices.

## Recommendations (no code change required)

1. **Add a one-line attribution NOTICE** (README or `docs/`) crediting
   FineWeb-Edu / Common Crawl, to satisfy ODC-By if results are published.
   Cheap insurance; not required merely to download for local research.
2. **Optional integrity pinning** — `bio_inspired_nanochat/dataset.py:142–199`
   already supports size/sha256 verification of shards; porting it here
   (tracked context in `docs/data_parity.md`) would also pin *which* data was
   used, aiding reproducibility and provenance — complementary to licensing.
3. **No action needed** for synthetic `mgr gen-tasks` corpora or tokenizer
   assets.

## Summary

Single external corpus (FineWeb-Edu via karpathy's shuffle), **ODC-By 1.0 +
Common Crawl ToU**, attribution-on-redistribution being the only real
obligation; tokenizer assets are MIT/Apache. Download-only usage as implemented
is low-risk; add an attribution NOTICE before publishing trained artifacts or
redistributing data.
</content>
