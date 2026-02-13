scir (crate paraguas)

Resumen
- Crate de conveniencia que reexporta sub-crates de SciR y agrega funciones de GPU para una experiencia de usuario más sencilla.

Características
- gpu: habilita CUDA en `scir-gpu` y las API de GPU en `scir-signal`.

Ejemplo
- CPU: `cargo run -p scir --example fir_gpu`
- CUDA: `cargo run -p scir --features gpu --example fir_gpu`
