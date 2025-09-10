Docker Compose for CUDA Dev

Usage
- Build and start a CUDA-ready dev container with sccache enabled:
  - docker compose -f docker-compose.gpu.yml up -d --build
  - docker compose -f docker-compose.gpu.yml exec gpu-dev bash

Environment
- sccache is installed and enabled via `RUSTC_WRAPPER=sccache`.
- Local cache volume is mounted at `/var/cache/sccache`.
- To use S3, export env vars before `docker compose`:
  - SCCACHE_BUCKET, SCCACHE_REGION, optional SCCACHE_S3_KEY_PREFIX
  - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or role/IMDS in EC2)

GPUs
- Service requests GPUs via `device_requests: [capabilities: ["gpu"]]`.
- Ensure NVIDIA Container Toolkit is installed on the host.
- If needed, try `runtime: nvidia` or `gpus: all` instead of `device_requests`.

Volumes
- `cargo-registry` and `cargo-git` volumes persist Cargo downloads.
- `sccache` volume persists compiled artifacts across runs.

Examples
- CPU tests:
  - cargo test -p scir-core
- CUDA tests (if GPU available):
  - cargo test -p scir-gpu --features cuda

