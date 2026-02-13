scir (crate d'ensemble)

Aperçu
- Crate de commodité réexportant les sous-crates SciR et regroupant les fonctionnalités GPU pour une expérience utilisateur simplifiée.

Fonctionnalités
- gpu : active CUDA dans `scir-gpu` et les API GPU dans `scir-signal`.

Exemple
- CPU : `cargo run -p scir --example fir_gpu`
- CUDA : `cargo run -p scir --features gpu --example fir_gpu`
