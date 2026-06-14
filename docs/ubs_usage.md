# UBS (Ultimate Bug Scanner) — usage in this repo

**Bead:** `model_guided_research-lpg`

UBS flags likely bugs before commit. This repo's policy: **`ubs --diff .`
before every commit that changes code** (exit 0 = safe; exit >0 = fix &
re-run). It is wired into `scripts/ci_lite.sh` stage 3, so running CI-lite
already covers it. No enforcement hook is installed — invocation is manual.

## Install (per AGENTS.md)

```bash
curl -sSL https://raw.githubusercontent.com/Dicklesworthstone/ultimate_bug_scanner/main/install.sh | bash
# binary lands at ~/.local/bin/ubs
ubs doctor          # verify install
```

## The golden rule

```bash
ubs --diff .        # working tree vs HEAD — the everyday pre-commit scan
ubs --staged .      # staged files — right before `git commit`
ubs FILE1 FILE2     # scope to specific files (fast)
ubs .               # whole project (slow; ignores .venv/node_modules)
```

**Scope to changed files.** `ubs --diff .` is fast; `ubs .` is slow — never
full-scan for a small edit.

## Reading output

```
⚠️  Category (N errors)
    file.py:42:5 – Issue description
    💡 Suggested fix
Exit code: 1
```

Parse `file:line:col` → location, 💡 → fix, exit 0/1 → pass/fail. Triage:

- **Critical** (always fix): null-safety, injection, async/await misuse,
  resource leaks.
- **Important** (fix for production): type narrowing, division-by-zero.
- **Info/contextual** (judgment): TODO/FIXME, console logs — fine to skip with
  reason.

Fix the **root cause**, not the symptom; re-run until exit 0.

## In this repo's workflow

1. `scripts/ci_lite.sh --staged` — runs ruff + **ubs --staged** + fast tests in
   one shot before a commit (see `docs/ci_lite.md`).
2. For a standalone scan of your edits: `ubs --diff .`.
3. CI mode for a PR-style check: `ubs --ci --fail-on-warning --diff .`.

UBS is advisory and Python/JS-aware; on this repo (Python 3.13) scope with
`--only=python` if you want to skip non-Python files:

```bash
ubs --diff --only=python .
```

## Notes

- `ubs` is **not** a quality gate that blocks commits automatically — it is a
  fast advisory pass. The authoritative gates remain `ruff`, the pytest suite,
  and the research-loop provenance checks.
- If `ubs` is absent, `scripts/ci_lite.sh` skips stage 3 with a note rather than
  failing — install it to get the bug-scan coverage.
- Tail the latest install/scan session: `ubs sessions --entries 1`.
</content>
