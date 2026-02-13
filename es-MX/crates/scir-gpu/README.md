```markdown
scir-gpu

Descripción general
- Bases de GPU para SciR. Proporciona una abstracción mínima de `DeviceArray<T>`, líneas base de CPU y rutas opcionales de CUDA detrás de la característica `cuda`.

Características
- cuda: habilita el uso de la API del controlador CUDA con kernels PTX incrustados para suma/multiplicación elemento a elemento f32 y FIR por lotes.

Requisitos (CUDA)
- GPU NVIDIA con controlador reciente instalado (libcuda presente en el host).
- En Linux, asegúrese de que `libcuda.so` sea visible para el contenedor o proceso; en Windows, se requiere `nvcuda.dll`.

Inicio rápido
- Pruebas de CPU: `cargo test -p scir-gpu`
- Pruebas de CUDA: `cargo test -p scir-gpu --features cuda`

APIs
- `DeviceArray<T>`: arreglos con forma con `device` y `dtype`. Almacenamiento respaldado por CPU actualmente.
- Elemento a elemento: `add_scalar_auto`, `mul_scalar_auto`, `add_auto` (f32) se despachan a CUDA cuando está disponible.
- FIR: `fir1d_batched_f32_auto(x, taps, device)` elige CUDA o CPU y vuelve a CPU si CUDA no está disponible.
```
