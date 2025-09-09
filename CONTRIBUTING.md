# Contributing to SciR

- Ensure the SciPy git submodule is initialized (`scripts/setup-ci-env.sh` handles this) or run `git submodule update --init --depth 1 scipy`.
- Run `python scripts/gen_fixtures.py -n 8` to regenerate test fixtures as needed.
- Execute `pytest` and `cargo test` before committing changes.
- Install pre-commit and run `pre-commit run --files <files>` on staged files.
- Follow AGENTS.md instructions and do not commit generated binaries.
- Format Rust code with `cargo fmt` and keep functions focused and documented.

## GPU testing

- CUDA paths are optional and off by default. Most contributors can work CPU-only.
- To run GPU tests locally on a CUDA-capable host with NVIDIA drivers:
  - `cargo test -p scir-gpu --features cuda`
  - Optional umbrella example: `cargo run -p scir --features gpu --example fir_gpu`
- GitHub-hosted runners don’t expose GPUs; use a self-hosted runner. See `docs/gpu-runner.md`.
- AWS CodeBuild is an alternative for GPU CI. See `docs/codebuild-gpu.md` and `buildspec.gpu.yml`.
