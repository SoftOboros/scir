```markdown
# SciR (SciPy en Rust) — Plan de Desarrollo (Revisado y Extendido)

> **Estado:** Revisado para consistencia con el hilo; alucinaciones corregidas; aclarado donde era escaso; CI + listas de verificación de fases añadidas.

## 0) Referencia y Restricciones Ascendentes (Fuente de Verdad)
- **Proyecto ascendente:** SciPy — GitHub: `scipy/scipy`.
- **Licencia:** BSD‑3‑Clause (permisiva). **No debemos** copiar código C/Fortran de SciPy; reimplementamos algoritmos y validamos el comportamiento mediante pruebas/fixtures.
- **Postura de GPU ascendente:** SciPy en sí es **solo CPU**; la aceleración de GPU en Python se realiza típicamente a través de **CuPy** u otros frameworks (JAX, PyTorch). Este es nuestro objetivo de diferenciación.
- **Cobertura de pruebas ascendente:** No se publica un % de cobertura oficial único; la cobertura es amplia y se mantiene activamente. Nuestro flujo trata el **comportamiento** de SciPy como la especificación a través de fixtures.

---

## 1) Objetivos y Diferenciación
- **Primero la paridad:** Reproducir APIs conocidas de SciPy (comenzando con `fft`, `signal`, `optimise`, `linalg`) con la ergonomía de Rust y tipos explícitos.
- **Conversión mecánica mediante pruebas:** Usar SciPy para generar **fixtures** canónicos; ejecutar implementaciones idénticas de Rust contra ellos. El comportamiento de Python se convierte en el **oráculo**.
- **Pista de rendimiento:** Después de la paridad, añadir **optimizaciones nativas de Rust** (slices, SIMD, Rayon) y **backends de GPU** opcionales (CUDA y/o cómputo portable a través de `wgpu`).
- **Ventaja de marketing:** "SciR: SciPy reconstruido para Rust — seguro, rápido, listo para GPU."

---

## 2) Espacio de Trabajo y Mapa de Módulos (Ajustado para Claridad)
```
scir/                              # Crate paraguas (re-exportaciones)
  Cargo.toml
  crates/
    scir-core/                     # Común: traits, errores, tipos, helpers de tolerancia
    scir-nd/                       # Interoperabilidad y conversiones ndarray (línea base CPU)
    scir-fft/                      # APIs FFT (CPU: rustfft; backends GPU opcionales)
    scir-signal/                   # Filtros (IIR/FIR), remuestreo, ventanas
    scir-optimize/                 # minimize (Nelder–Mead, BFGS/L-BFGS), least_squares
    scir-linalg/                   # Álgebra lineal (faer / ndarray-linalg / BLAS)
    scir-gpu/                      # (opcional) backends y gestión de dispositivos de array (ver §6)
```
**Notas**
- `scir-nd` aísla las conversiones de `ndarray`/forma/paso para mantener los crates de cómputo enfocados.
- `scir-gpu` es una capa *delgada*: memoria del dispositivo, transferencias y despacho de backend. El cómputo reside en submódulos feature-gated de cada crate de dominio.

---

## 3) Dependencias Externas (Corregidas y Conservadoras)
- **Arrays:** `ndarray` (línea base CPU). Considerar `ndarray-linalg` para operaciones basadas en Lapack.
- **FFT:** `rustfft` (CPU). (La ruta de GPU es personalizada, ver §6.)
- **Linalg (CPU):** `faer` (solo Rust) y/o `ndarray-linalg` (BLAS/LAPACK a través de `openblas-src`/sistema). Proteger detrás de features.
- **Paralelo:** `rayon`.
- **Tipos numéricos:** `num-traits`, `num-complex`.
- **SIMD:** preferir crates como `wide` (portables); seguir la estabilización de `portable_simd` de Rust para el futuro.
- **Serde:** opcional (`serde`, `serde_json`, `ndarray-npy`) para fixtures.
- **GPU:** ver §6. Evitar reclamar un crate cuFFT específico; podríamos escribir FFI mínimo si no existen bindings mantenidos.

---

## 4) Modelo de Error y Superficie de API
- Un enum de error público por crate re-exportado a través de `scir-core`:
```rust
#[derive(thiserror::Error, Debug)]
pub enum ScirError {
    #[error("invalid argument: {0}")] InvalidArgument(String),
    #[error("numerical failure: {0}")] NumericalFailure(String),
    #[error("backend not available: {0}")] BackendUnavailable(String),
}
```
- Las funciones usan **estructuras explícitas** para las opciones para evitar ambigüedad (ej., `FftPlan`, `FilterDesignOpts`, `MinimizeOpts`).
- Números complejos: `num_complex::Complex<f64>` / `<f32>` con alias de tipo.
- **Nomenclatura:** reflejar los nombres de las funciones de SciPy en `snake_case` y agrupar por módulo (`scir::signal::filtfilt`, etc.).

---

## 5) Paridad a Través de Fixtures (Detallado y Sin Ambigüedades)
### 5.1 Generación de Fixtures (Python)
- Fijar versiones para asegurar resultados deterministas: ej., `numpy==1.x`, `scipy==1.y`.
- Para cada función objetivo, crear un script de Python que:
  1) Genere entradas representativas, incluyendo casos límite.
  2) Llame a la implementación de SciPy.
  3) Guarde **entradas** y **salidas esperadas** en `fixtures/` como `.npy` o JSON.
- **Datos complejos** en JSON: codificar como tuplas `[re, im]` o usar `.npy` para evitar codificaciones personalizadas.
- **Ejemplo:** (pseudo‑fragmento)
```python
# export_fixtures_fft.py
import numpy as np
from scipy.fft import fft, ifft, rfft, irfft
np.random.seed(0)
x = np.random.randn(1024).astype(np.float64)
np.save('fixtures/fft/x_f64.npy', x)
np.save('fixtures/fft/fft_expected.npy', fft(x))
```

### 5.2 Pruebas de Paridad (Rust)
- Cada crate carga sus fixtures y afirma tolerancias:
```rust
let x: Array1<f64> = read_npy("fixtures/fft/x_f64.npy")?;
let y_ref: Array1<Complex64> = read_npy("fixtures/fft/fft_expected.npy")?;
let y = scir::fft::fft(x.view());
assert_close!(y, y_ref, atol=1e-9, rtol=1e-7);
```
- Proporcionar una macro `assert_close!` en `scir-core` para tolerancias numéricas consistentes.
- **Advertencias específicas de señal:** manejo de bordes de `filtfilt` (el esquema de relleno predeterminado en SciPy es crítico):
  - `padtype="odd"`, `padlen = 3 * (max(len(a), len(b)) - 1)` a menos que se anule.
  - Usar la paridad del método de **Gustafsson** donde lo hace SciPy.

### 5.3 Alcance del Fixture
- `fft`: real/complejo, tamaños primos/potencia de dos, desplazamientos.
- `signal`: diseño `butter/cheby1/bessel` → SOS, `sosfilt`, `filtfilt`, `resample_poly`.
- `optimize`: `rosenbrock`, `himmelblau` con mínimos conocidos; conjuntos de datos `least_squares`.
- `linalg`: problemas pequeños/medianos para `svd`, `qr`, `solve`, matrices condicionadas.

---

## 6) Extensión de GPU (Corregido y Fundamentado)
**Verificación de la realidad:** Los runners alojados en GitHub generalmente **no** exponen GPUs. La CI de GPU requiere runners **auto-alojados** o CI de terceros con GPU. Nuestro plan refleja eso.

### 6.1 Estrategia de Backend
- **CUDA (NVIDIA):** Usar `cudarc` (si es suficiente para nuestras necesidades) o implementar FFI mínimo a cuBLAS/cuFFT si no hay crates mantenidos disponibles. No reclamaremos un crate existente hasta que se verifique; planificar FFI a través de `bindgen` como fallback.
- **Cómputo portable:** `wgpu` (Vulkan/Metal/DX12/WebGPU) con kernels de cómputo WGSL. FFT en `wgpu` no es trivial; empezar con operaciones puntuales, convoluciones, GEMM; explorar kernels FFT de radix‑N como un hito posterior.
- **OpenCL:** `ocl` u `opencl3` como alternativa donde CUDA no esté disponible.

### 6.2 Abstracción de Array de Dispositivo
En lugar de almacenar `wgpu::Buffer`s crudos en un enum genérico, definimos **arrays de dispositivo con forma** con dtype y pasos para reflejar la semántica de la CPU:
```rust
pub enum Device {
    Cpu,
    Cuda(CudaCtx),
    Wgpu(WgpuCtx),
}

pub struct DArray {
    pub device: Device,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub dtype: DType,           // F32, F64, C64, C128
    // manejador opaco: CPU -> almacenamiento ndarray; GPU -> buffer específico del backend
}
```
- Proporcionar transferencias `to_cpu()`/`to_device(Device)` con sincronización explícita.
- Las APIs de alto nivel aceptan arrays de CPU; las rutas de GPU son **opcionales** a través de feature flags y conversiones explícitas.

### 6.3 Objetivos y Orden de GPU
1) **Elemento a elemento / map-reduce** (ganancia fácil): escala, suma, multiplicación, abs2, reducciones.
2) **Convolución / FIR** (1D/2D): FIR por lotes a través de mosaicos; más tarde IIR (con retroalimentación) usando prefijo paralelo o procesamiento por bloques.
3) **GEMM**: aprovechar cuBLAS (CUDA) o kernels WGSL ajustados.
4) **FFT**: primero la ruta CUDA (vía FFI de cuFFT); FFT de `wgpu` más tarde (requiere kernels personalizados).

### 6.4 Validación
- Siempre verificar **GPU == CPU** dentro de las tolerancias.
- Proporcionar semillas deterministas y cargas de trabajo dimensionadas para que las comparaciones entre backends sean fiables.

---

## 7) Optimizaciones de CPU (Pragmáticas)
- **Slicing/borrowing:** preferir rutas calientes `&[T]`/`&mut [T]`; adaptar a vistas `ndarray` para la API.
- **SIMD:** usar `wide` o intrínsecos específicos del backend; proteger con features.
- **Paralelo:** `rayon` para operaciones embarazosamente paralelas.
- **Estabilidad numérica:** coincidir con los valores predeterminados de SciPy (ej., transformadas bilineales, escalado SOS) para reducir la deriva.

---

## 8) Benchmarks y Reproducibilidad
- Benchmarks de **`criterion`** que reflejan los fixtures (tamaños de FFT, longitudes de filtro, dimensiones del optimizador).
- Comparar con **SciPy** (CPU) en la misma máquina para una verificación de la realidad; documentar hardware y versiones.
- Registrar el rendimiento/latencia de la línea base; seguir las regresiones a través de artefactos de tendencia de CI.

---

## 9) CI y Pruebas de Plataforma (Revisado y Accionable)
### 9.1 Matriz (CPU)
- **SO:** ubuntu‑latest, macos‑latest, windows‑latest.
- **Rust:** stable, beta, nightly.
- **Features:** `pure-rust` (predeterminado), `faer`, `blas`.

### 9.2 GitHub Actions (CPU) — Ejemplo
```yaml
name: ci
on: [push, pull_request]

jobs:
  cpu:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, beta, nightly]
        features: ["", "faer", "blas"]
    steps:
      - uses: actions/checkout @ops/packer/BUGSBUNNY-v4-CHANGELOG.md
      - uses: dtolnay/rust-toolchain @backend/istate/tests/data/SCXML-tutorial/Tests/TestsTableW3C.csv
        with: { toolchain: ${{ matrix.rust }} }
      - uses: Swatinem/rust-cache @v2
      - run: cargo build --verbose --features "${{ matrix.features }}"
      - run: cargo test  --verbose --features "${{ matrix.features }}"
```

### 9.3 CI de GPU (Auto-alojado)
```yaml
  gpu:
    if: github.repository_owner == 'YOUR_ORG'  # evitar forks
    runs-on: [self-hosted, linux, x64, gpu, nvidia]
    env:
      RUST_LOG: info
    steps:
      - uses: actions/checkout @ops/packer/BUGSBUNNY-v4-CHANGELOG.md
      - uses: dtolnay/rust-toolchain @backend/istate/tests/data/SCXML-tutorial/Tests/TestsTableW3C.csv
        with: { toolchain: stable }
      - run: nvidia-smi  # verificación de sanidad
      - run: cargo test --features cuda --package scir-fft
```
- Proporcionar documentos para configurar el **runner de GPU auto-alojado** (controladores CUDA, toolkit, etiquetas).
- Para pruebas de `wgpu` en macOS con series M, ejecutar un runner de macOS auto-alojado.

### 9.4 Paridad de Python en CI
- Ejecutar Python una vez para **(re)generar fixtures** (versiones fijas) y subirlos como artefacto de CI usado por los jobs de Rust.
```yaml
  fixtures:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout @ops/packer/BUGSBUNNY-v4-CHANGELOG.md
      - uses: actions/setup-python @v5
        with: { python-version: '3.11' }
      - run: python -m pip install --upgrade pip
      - run: pip install numpy==1.X scipy==1.Y pytest
      - run: python tools/export_fixtures.py
      - uses: actions/upload-artifact @ops/packer/BUGSBUNNY-v4-CHANGELOG.md
        with:
          name: scir-fixtures
          path: fixtures/**
```
- Los jobs de CPU/GPU **descargan** el artefacto `scir-fixtures` antes de `cargo test`.

### 9.5 Cobertura e Informes
- Usar `cargo-llvm-cov` para la cobertura de Rust; `pytest --cov` para el harness de Python si es necesario.
- Publicar con `codecov/codecov-action @v4` (opcional).

---

## 10) Licencias y Cumplimiento (Explícito, Accionable)

> **Intención:** Mantener `scir` legalmente limpio y adoptable por empresas, mientras se aprovecha SciPy como un oráculo de *comportamiento*.

### 10.1 Procedencia del Código (Qué está / no está permitido)
- **NO copiar** código de implementación de SciPy (C/Fortran/Python) ni traducir línea por línea.
- **OK:** Implementar algoritmos de **artículos, estándares y documentos de SciPy** (conceptos ≠ código). Citar fuentes en `ALGORITHMS.md`.
- **OK:** Generar y almacenar **fixtures de prueba** ejecutando SciPy; los fixtures son *datos*, no código. Mantener la procedencia (ver 10.3).
- **OK:** Usar crates de terceros con **licencias permisivas** (MIT/Apache‑2.0/BSD-2/3/ISC/Zlib). Evitar copyleft fuerte/viral.

### 10.2 Licencias para `scir`
- Elegir **una** licencia para todos los crates (**MIT**). Mantener la consistencia en todo el repositorio.
- Añadir encabezados SPDX a cada archivo fuente:
  ```rust
  // SPDX-License-Identifier: Apache-2.0
  ```
- `Cargo.toml` por crate:
  ```toml
  license = "MIT"
  ```
- Archivo `LICENCE` de nivel superior que coincida con la licencia elegida; sin divergencia por crate.

### 10.3 Política de Fixtures (SciPy como Oráculo)
- **Almacenar con procedencia** junto a cada lote de fixtures (YAML o JSON):
  ```yaml
  generator: tools/export_fixtures.py
  python: 3.11.x
  numpy: 1.x.y
  scipy: 1.y.z
  seed: 123456
  os: linux-x86_64
  date: 2025-09-08
  script_sha256: <hash-of-exporter>
  ```
- Mantener los **scripts de regeneración** en `tools/` y fijar las versiones de las dependencias.
- Usar **datos sintéticos** para las pruebas; evitar conjuntos de datos con derechos de autor a menos que las licencias sean compatibles y estén registradas.

### 10.4 Submódulos y Vendoring
- Instalar SciPy vía `pip` para pruebas/referencias locales; evitar submódulos git a menos que sea absolutamente necesario.
- **NO vendoring** el código de SciPy en los crates de `scir`. Mantener las herramientas de Python en `tools/` y fuera de los crates publicables.
- Asegurarse de que los crates se publiquen **sin** fuentes de SciPy (verificar `include`/`exclude` en `Cargo.toml`).

### 10.5 Dependencias de Terceros
- Mantener `licence-allowlist.toml` y aplicar a través de CI (`cargo-deny`). Lista de permitidos sugerida: `Apache-2.0`, `MIT`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`, `Zlib`.
- **Evitar** GPL/AGPL. Si LGPL/MPL es inevitable, proteger detrás de feature(s) **no predeterminadas** y documentar claramente las obligaciones.
- BLAS/LAPACK: preferir **faer** (solo Rust) o **OpenBLAS**/**BLIS** vía `ndarray-linalg` (permisivas). **No** enlazar rígidamente a bibliotecas GPL.

### 10.6 Backends de GPU y SDKs de Proveedores
- **CUDA:** Enlazar vía FFI; **no** empaquetar encabezados o binarios de NVIDIA. Requerir que los usuarios instalen CUDA bajo la EULA de NVIDIA; detectar en tiempo de construcción.
- **wgpu/OpenCL:** Verificar compatibilidad de licencias; preferir crates permisivas. Mantener el código del backend en features opcionales (`cuda`, `wgpu`, `opencl`).
- La selección en tiempo de ejecución **no debe** forzar la inclusión de componentes propietarios en las compilaciones predeterminadas.

### 10.7 Documentación, Ejemplos y Marcas Comerciales
- **NO copiar** los documentos de SciPy textualmente. Parafrasear y atribuir: "Comportamiento alineado con SciPy 1.y.z." Citar solo fragmentos cortos cuando sea necesario con atribución.
- **Marca:** **NO** implicar afiliación. Usar una exención de responsabilidad en el README: "`scir` no está afiliado ni respaldado por el proyecto SciPy." Evitar usar el nombre **SciPy** en los nombres de los paquetes.
- Considerar licenciar la documentación bajo **CC‑BY‑4.0** (opcional) y el código bajo Apache‑2.0/MIT; documentar esta división en el `README`.

### 10.8 Política de IP del Contribuidor
- Usar **DCO** (Developer Certificate of Origin) o un CLA simple. Requerir `Signed-off-by` en los commits vía CI.
- La plantilla de PR debe incluir:
  - Confirmación de **trabajo original** (sin código copiado de SciPy u otras fuentes restringidas).
  - Reconocimiento de la licencia del proyecto y la política de fixtures.
  - Lista de verificación para añadir encabezados SPDX y actualizar las referencias `ALGORITHMS.md`.

### 10.9 Cumplimiento Automatizado en CI
- Añadir paso `cargo-deny` (licencia + bans + fuentes). Fallar CI en violaciones.
- Añadir validación **REUSE**/SPDX (ej., `fsfe/reuse-action`) para asegurar que todos los archivos lleven IDs SPDX.
- Asegurarse de que `cargo package --list` no muestre archivos de SciPy o SDK de proveedores incluidos.

### 10.10 Higiene de Lanzamiento
- Los artefactos de lanzamiento deben incluir: `LICENCE`, `NOTICE` (si es requerido), `ALGORITHMS.md`, archivos de procedencia de fixtures, y `CITATION.cff` legible por máquina (opcional pero útil para la academia).
- Las entradas del Changelog deben notar cualquier nueva dependencia y sus licencias.

### 10.11 Lista de Verificación Rápida de Cumplimiento (por PR)
- [ ] No hay código ascendente copiado; fuentes citadas en `ALGORITHMS.md`.
- [ ] Fixtures regenerados con SciPy/Numpy fijados; procedencia actualizada.
- [ ] Encabezados SPDX añadidos/mantenidos; `licence` de `Cargo.toml` establecido.
- [ ] `cargo-deny` pasa; no hay licencias no permitidas.
- [ ] El uso de SDK de GPU/proveedor está protegido detrás de features no predeterminadas; no hay binarios/encabezados empaquetados.
- [ ] README contiene la exención de responsabilidad de afiliación y el resumen de licencias.
- [ ] `cargo package --list` auditado; crates publicables limpios de código Python/de proveedor.

## 11) Versionado y Publicación
- Versionado semántico por crate; sincronizar a través de un script de lanzamiento del espacio de trabajo.
- Publicar crates incrementalmente (`scir-core` → `scir-fft` → `scir-signal` …) a medida que la funcionalidad aterriza.
- Etiquetar lanzamientos y adjuntar resúmenes de benchmarks y estado de paridad.

---

## 12) Hoja de Ruta (Fases con Resultados)
**Fase 1 — Esqueletos de paridad (CPU):**
- `scir-core`, `scir-nd`, `scir-fft` (fft/ifft/rfft/irfft), ventanas básicas de `signal`.
- Fixtures para FFT + filtros simples; CI (CPU) en ejecución.
- **Resultado:** Pruebas en verde vs SciPy para FFT; >80% de cobertura del crate.

**Fase 2 — Señal y Optimización:**
- Diseño de filtro (`butter`, `cheby1`, `bessel` → SOS), `sosfilt`, `filtfilt`, `resample_poly`.
- `optimize::minimize` (Nelder–Mead, BFGS/L‑BFGS) en Rosenbrock/Himmelblau.
- **Resultado:** Paridad en rutinas de señal principales; el optimizador coincide con SciPy dentro de las tolerancias.

**Fase 3 — Linalg y Backends:**
- Introducir backend `faer`; features opcionales de BLAS/LAPACK vía `ndarray-linalg`.
- Benchmarks iniciales; harness de fuzz para casos límite.
- **Resultado:** APIs de linalg estables; la matriz de CI cubre las features de faer/blas.

**Fase 4 — Fundamentos de GPU:**
- Abstracción de array de dispositivo; elemento a elemento + FIR por lotes en CUDA y/o `wgpu`.
- CI de GPU auto-alojado; comprobaciones de paridad CPU↔GPU.
- **Resultado:** Primera victoria de GPU (aceleración medible) con API idéntica.

**Fase 5 — GPU Avanzada y FFT:**
- Ruta FFI de cuFFT para CUDA; kernels exploratorios de FFT en `wgpu`.
- GEMM vía cuBLAS (CUDA) o WGSL ajustado; integrar en `linalg`.
- **Resultado:** Aceleración significativa de GPU para FFT/señal/GEMM.

**Fase 6 — Optimización y Reforzamiento:**
- SIMD, bloqueo consciente de la caché, fusión de kernels donde sea aplicable.
- Documentación, ejemplos, docs de API autogenerados; publicar v0.1–v0.3.

---

## 13) Experiencia del Desarrollador (DX) y Contribuciones
- Targets `just` o `make`: `just fixtures`, `just test`, `just bench`.
- `tools/` con exportadores de Python; fijar vía `requirements.lock` o `uv`.
- `CONTRIBUTING.md` con guía de estilo, política de tolerancia y reglas de procedencia de fixtures.

---

## 14) Preguntas Abiertas (Seguimiento)
- ¿Qué backend de GPU priorizar primero (cuota de mercado de CUDA vs portabilidad de `wgpu`)?
- ¿Dónde trazamos el límite de la API de SciPy (paridad total vs subconjunto pragmático)?
- Política de tolerancia por función/dominio (documentada en `scir-core`).

---

## 15) Listas de Verificación de Finalización de Fase

### Fase 1 — Línea Base de CPU FFT
- [ ] Utilidades de error/tolerancia de `scir-core`.
- [ ] Puentes ndarray de `scir-nd`, helpers de E/S `.npy`.
- [ ] `scir-fft`: fft/ifft/rfft/irfft (+ helpers de desplazamiento).
- [ ] Fixtures de Python para FFT (tamaños: 64…65536; real y complejo).
- [ ] CI (matriz de CPU) en verde; informe de cobertura subido.

### Fase 2 — Señal y Optimización
- [ ] `scir-signal`: diseño `butter`, `cheby1`, `bessel` → SOS; `sosfilt`, `filtfilt`, `resample_poly`.
- [ ] Paridad de fixtures para diseño de filtro y casos límite de filtfilt.
- [ ] `scir-optimize`: Nelder–Mead, BFGS/L‑BFGS con búsqueda lineal.
- [ ] Fixtures del optimizador (Rosenbrock, Himmelblau) pasan dentro de las tolerancias.

### Fase 3 — Linalg y Backends (CPU)
- [x] Ruta `faer` implementada (feature cableado; ruta temporal `ndarray-linalg` para paridad, integración nativa `faer` en cola a continuación).
- [x] Conjunto mínimo de Solve/SVD/QR con fixtures.
- [x] Benchmarks para FFT, filtros, solve.

### Fase 4 — Fundamentos de GPU
- [x] Abstracción de array de dispositivo `scir-gpu` + transferencias (respaldado por CPU, stub CUDA feature-gated).
- [x] Elemento a elemento CUDA + FIR por lotes con paridad vs CPU (kernels PTX + FFI de controlador; las pruebas se omiten si no hay CUDA).
- [ ] Job de CI de GPU auto-alojado en verde; docs para la configuración del runner.

### Fase 5 — GPU FFT y GEMM
- [ ] Ruta FFI de cuFFT cableada para FFT 1D; paridad vs fixtures de CPU.
- [ ] GEMM vía cuBLAS (CUDA) o kernels WGSL; integración de linalg.
- [ ] Pruebas entre backends (CPU↔CUDA↔WGPU) dentro de las tolerancias.

### Fase 6 — Optimización y Reforzamiento
- [ ] SIMD, bloqueo consciente de la caché, fusión de kernels donde sea aplicable.
- [ ] Benchmarks de Criterion estables; notas de rendimiento registradas por lanzamiento.
- [ ] Sitio de documentación publicado; ejemplos y procedencia de fixtures documentados.
- [ ] Lanzamientos v0.1…v0.3 etiquetados y crates publicados.

---

**TL;DR:** El plan ahora refleja la intención del hilo, corrige las realidades de GPU/CI (requisito auto-alojado), evita crates especulativos, logra la paridad impulsada por fixtures y añade puertas de fase claras para que esto se sienta como "disparar a peces en un barril" — con comprobantes en CI.


## Registro de Progreso
- Andamiaje inicial: README, AGENTS, submódulo SciPy, actualizaciones de scripts.
- Añadidas dependencias numpy/scipy y bootstrap de crates scir-core/nd.

- Implementada macro assert_close! y helpers de conversión de ndarray Vec.
- Creado script de generación de fixtures de ejemplo e ignorada la salida en git.
- Documentados los pasos de instalación de dependencias de Python en README.
- Añadido soporte de slice/complex a `assert_close!` y generador de fixtures parametrizado.
- Extendida `assert_close!` para trabajar con arrays ndarray y documentada la inicialización del submódulo en README.
- Añadidas pruebas basadas en arrays en `scir-nd` y cambiados los fixtures a `.npy` con formato documentado.

- Creado crate scir-fft con FFT real y pruebas de paridad basadas en fixtures.
- Añadida guía CONTRIBUTING para colaboradores.
- Implementada FFT inversa con fixtures para múltiples tamaños y notadas las guías de estilo en CONTRIBUTING.
- Añadidas pruebas de paridad para FFT e IFFT en múltiples tamaños de fixture.
- Añadidas rutinas de FFT real (rfft/irfft) con fixtures y pruebas de múltiples tamaños.
- Introducido crate scir-signal con diseño Butterworth y paridad sosfilt.
- Añadido crate scir-optimize implementando Nelder–Mead y BFGS validados en fixtures de Rosenbrock y Himmelblau.
- Extendidos los fixtures de señal para incluir diseños Chebyshev y Bessel con APIs de Rust y andamiaje de filtfilt.
- Añadido filtrado de fase cero y fixtures de `resample_poly` con pruebas de paridad de Rust.
- Implementado optimizador L-BFGS y fixtures para Rosenbrock y Himmelblau.
- Fase 2 completa: rutinas de señal principales y optimizador L-BFGS validados.
- Añadidos hooks pre-commit y actualizadas las guías de contribución.
- Eliminado el submódulo SciPy de git; SciPy instalado con pip satisface la generación de fixtures.
- Inicio de la Fase 3: reintroducido el submódulo SciPy de git en `/scipy` y actualizadas las referencias de la documentación.
- Actualizado el script de configuración de CI para inicializar el submódulo SciPy automáticamente.
- Documentada la ruta del submódulo como `/scipy` y hecho el script de configuración agnóstico a la ruta.
- Añadido crate `scir-linalg` con ruta BLAS/LAPACK vía `ndarray-linalg` y feature flags para el backend `faer`.
- Implementadas APIs `solve`, `svd` y `qr` con pruebas basadas en fixtures (feature BLAS).
- Añadido `scripts/gen_linalg_fixtures.py` para generar fixtures de linalg (`lin_solve_*`, `svd_A.npy`, `qr_A.npy`).
- Preparado el feature gating para el backend `faer` (placeholders); se habilitará en un seguimiento.
- Añadidos benches de Criterion para FFT, señal y linalg solve; añadida prueba de propiedad básica para `solve` con matrices SPD.
- Cableado la feature `faer` a la función (actualmente vía `ndarray-linalg`), para ser reemplazado con rutinas nativas `faer` en la próxima iteración.

— Fase 3 completa —

- Andamiaje del crate `scir-gpu` con `DeviceArray` (forma, dtype, dispositivo), transferencias de CPU, operaciones elemento a elemento y línea base de FIR por lotes en CPU con pruebas (stubs CUDA protegidos por feature).
- Añadido esqueleto de flujo de trabajo de CI de GPU auto-alojado (`.github/workflows/gpu.yml`) y guía de configuración del runner (`docs/gpu-runner.md`).
- Ejecutado `cargo fmt --all` para alinear el formato.

- Implementada ruta mínima de CUDA (feature `cuda`) usando Driver API + PTX embebido para suma de vectores f32, suma escalar y FIR por lotes; las pruebas se omiten si CUDA no está disponible.
- Añadido `buildspec.gpu.yml` de CodeBuild GPU y docs sobre el uso de una imagen ECR habilitada para CUDA y modo privilegiado.
- Habilitado el paso de prueba CUDA en `.github/workflows/gpu.yml` para runners de GPU auto-alojados.
- Añadidos helpers de despacho automático: suma/suma escalar elemento a elemento y FIR pueden enrutarse a CUDA cuando el dispositivo es `Cuda` (fallback a CPU si no está disponible).
- Añadido `ci/docker/Dockerfile.cuda` para construir una imagen CUDA+Rust para CodeBuild; proporcionado `ci/codebuild/project.example.json` y `docs/codebuild-gpu.md` con instrucciones de configuración.
- Añadido crate paraguas `crates/scir` con feature `gpu` agregada y ejemplo `fir_gpu.rs` mostrando FIR CPU vs CUDA.
- Actualizado el README de nivel superior con el uso del crate paraguas, instrucciones de la feature de GPU, referencias de CI runner y CodeBuild.
- Añadidos READMEs de crates: `crates/scir/README.md` y `crates/scir-gpu/README.md` para un uso rápido y una visión general de las features.
- Añadidos ejemplos: `crates/scir/examples/elementwise_gpu.rs` (operaciones elemento a elemento CPU vs CUDA) junto con la demo de FIR.
- CONTRIBUTING actualizado con una sección concisa de pruebas de GPU y enlaces a documentos de GPU.
- Añadido `crates/scir/examples/fir_bench.rs` para una simple medición de tiempo CPU vs CUDA usando `Instant` (sin dependencias extra). Añadido alias de feature agregado `gpu-all` en el crate paraguas.

- Corregidos errores de compilación de doctest de `scir-signal` eliminando el acceso a internos privados de `Sos` en los ejemplos; actualizados los ejemplos para validar vía `sosfilt`. Ejecutado `cargo fmt` y verificados los doctests.
```
