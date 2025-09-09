# Contributing to SciR

 - Ensure the SciPy git submodule is initialized (`scripts/setup-ci-env.sh` handles this) or run `git submodule update --init --depth 1 scipy`.
- Run `python scripts/gen_fixtures.py -n 8` to regenerate test fixtures as needed.
- Execute `pytest` and `cargo test` before committing changes.
- Install pre-commit and run `pre-commit run --files <files>` on staged files.
- Follow AGENTS.md instructions and do not commit generated binaries.
- Format Rust code with `cargo fmt` and keep functions focused and documented.
