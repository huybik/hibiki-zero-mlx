# AGENTS.md

Guidance for AI coding agents working in this repository.

## Rules
- Do not over-engineer.
- Aim for simplicity and elegance of implementation.
- Make minimal edits.
- Always read the whole file when you open it.
- Read `CONTEXT.md` at the start of every session to understand the project instead of starting from scratch.
- Update `CONTEXT.md` after a large code update; keep `CONTEXT.md` concise.
- Break larger work into phases. Run agents sequentially (one phase at a time) and commit after each phase, so progress stays clear and reviewable.
- Spawn agents to manage your context better, but run them one at a time rather than in parallel.
- Do not add `Co-Authored-By` or any AI attribution to commit messages.

## Tooling
- For Python, use the conda base env at `/opt/homebrew/Caskroom/miniconda/base/bin/python` (Python 3.13, has project deps installed). Do not use `/usr/bin/python3` (3.9, too old) and do not pip-install into `--user`.
- When you need docs for a third-party library, SDK, or external service, fetch current API reference instead of relying on training data.
