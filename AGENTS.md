# AGENTS

- Run `pytest` for Python changes and `bash -n` for shell scripts touched.
- Keep commit messages descriptive.
- Do not commit binary artifacts; use scripts or encodings instead.
- Update `PLAN.md` with progress when tasks are completed.

Formatting policy
- Always rely on `cargo fmt` for Rust formatting. It is the source of truth.
- The pre-commit hook runs `cargo fmt --all` to auto-format changes; CI enforces `cargo fmt -- --check`.
- Agents and contributors do not need to hand-format Rust; run the formatter instead.
