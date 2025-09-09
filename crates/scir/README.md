scir (umbrella crate)

Overview
- Convenience crate re-exporting SciR sub-crates and aggregating GPU features for an easier user experience.

Features
- gpu: enables CUDA in `scir-gpu` and GPU APIs in `scir-signal`.

Example
- CPU: `cargo run -p scir --example fir_gpu`
- CUDA: `cargo run -p scir --features gpu --example fir_gpu`

