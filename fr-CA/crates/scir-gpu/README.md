```markdown
scir-gpu

Aperçu
- Fondations GPU pour SciR. Fournit une abstraction minimale `DeviceArray<T>`, des bases CPU et des chemins CUDA optionnels derrière la fonctionnalité `cuda`.

Fonctionnalités
- cuda : permet l'utilisation de l'API CUDA Driver avec des noyaux PTX embarqués pour l'addition/multiplication élément par élément f32 et le FIR par lots.

Exigences (CUDA)
- GPU NVIDIA avec pilote récent installé (libcuda présent sur l'hôte).
- Sous Linux, assurez-vous que `libcuda.so` est visible par le conteneur ou le processus; sous Windows, `nvcuda.dll` est requis.

Démarrage rapide
- Tests CPU : `cargo test -p scir-gpu`
- Tests CUDA : `cargo test -p scir-gpu --features cuda`

APIs
- `DeviceArray<T>` : tableaux formés avec `device` et `dtype`. Stockage basé sur CPU aujourd'hui.
- Élément par élément : `add_scalar_auto`, `mul_scalar_auto`, `add_auto` (f32) se redirigent vers CUDA lorsqu'il est disponible.
- FIR : `fir1d_batched_f32_auto(x, taps, device)` choisit CUDA ou CPU et revient au CPU si CUDA n'est pas disponible.
```
