```markdown
# SciR

SciR tiene como objetivo reimplementar las piezas centrales de SciPy en Rust con un enfoque de paridad primero y backends opcionales de GPU.

Licencia
- MIT. Consulte el archivo LICENSE en la raíz del repositorio. Todos los crates declaran `license = "MIT"` en su Cargo.toml.

Inicio rápido
- Compilar espacio de trabajo: `cargo build` (la primera ejecución buscará dependencias).
- Ejecutar pruebas: `cargo test` (las pruebas de GPU están restringidas y se omiten sin GPU).

Crate paraguas y característica de GPU
- Use `crates/scir` como un conveniente crate paraguas que reexporta los crates principales.
- Habilite la característica `gpu` agregada para activar las rutas respaldadas por CUDA y las API de GPU en los crates dependientes:
  - CPU: `cargo run -p scir --example fir_gpu`
  - CUDA: `cargo run -p scir --features gpu --example fir_gpu`
  - Demostración element-wise: `cargo run -p scir --example elementwise_gpu` (agregue `--features gpu` para probar CUDA)
  - Benchmark FIR: `cargo run -p scir --example fir_bench` (agregue `--features gpu` para medir el tiempo de CUDA)

Descripción general del soporte de GPU
- Los backends de GPU están desactivados por defecto y están protegidos por características.
- Cobertura actual de CUDA en `scir-gpu`: suma/multiplicación element-wise y FIR por lotes (f32) a través de la API de controlador CUDA + PTX incrustado.
- Los ayudantes de despacho automático enrutan las operaciones a CUDA cuando se selecciona `Device::Cuda`; si el backend no está disponible, recurren a la CPU.

CI de GPU autoalojado y AWS CodeBuild
- Las acciones de GitHub para GPU requieren ejecutores autoalojados; consulte `.github/workflows/gpu.yml` y `docs/gpu-runner.md` para la configuración y las etiquetas.
- Alternativamente, use AWS CodeBuild con una imagen de contenedor habilitada para CUDA. Consulte:
  - `ci/docker/Dockerfile.cuda` (imagen base CUDA+Rust)
  - `buildspec.gpu.yml` (pasos de compilación/prueba)
  - `ci/codebuild/project.example.json` y `docs/codebuild-gpu.md` (plantilla de proyecto y guía)

Paridad a través de fixtures
- Generamos fixtures de SciPy y los probamos. Consulte `PLAN.md` para obtener más detalles.

Este repositorio actualmente rastrea el trabajo de andamiaje inicial, incluyendo scripts para generar fixtures de referencia.

## Primeros Pasos

Ejecute `scripts/setup-ci-env.sh` para instalar los requisitos previos y descargar el submódulo SciPy en `/scipy`.
Instale las dependencias de Python con `pip install -r requirements.txt` y ejecute las pruebas a través de `pytest` y `cargo test`.

Si omitió el script de configuración, inicialice el submódulo git de SciPy (extraído en `/scipy`) con:

```
git submodule update --init --depth 1 scipy
```

Genere fixtures de FFT de referencia con `python scripts/gen_fixtures.py --sizes 8 16` (los archivos aterrizan en `fixtures/`, que está ignorado por git).
Genere fixtures de optimización con `python scripts/gen_optimize_fixtures.py` y fixtures de señal con `python scripts/gen_signal_fixtures.py` (datos de Butterworth, Chebyshev, Bessel, filtfilt y `resample_poly`). Los fixtures de optimización cubren los resultados de Nelder–Mead, BFGS y L-BFGS.

Los fixtures se almacenan como arreglos `.npy`. Para cada tamaño `<n>`, el script produce:
- `fft_input_<n>.npy`
- `fft_output_<n>.npy`
- `ifft_output_<n>.npy`
- `rfft_output_<n>.npy`
- `irfft_output_<n>.npy`

Las salidas complejas utilizan el tipo de datos complejo nativo de NumPy. Regenerar los fixtures según sea necesario; el directorio permanece sin seguimiento.

## Licencia

Doble licencia bajo Apache-2.0 y MIT.
```
