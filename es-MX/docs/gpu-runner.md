GPU CI Runner (Autohospedado) Configuración

Descripción general
- Los runners alojados en GitHub no exponen GPU. Utilice una máquina autohospedada con una GPU NVIDIA para trabajos CUDA o una máquina Apple Silicon/macOS para Metal.

Pasos para Linux (CUDA)
- Instale el controlador NVIDIA y CUDA Toolkit que coincidan con su GPU (por ejemplo, CUDA 12.x).
- Asegúrese de que `libcuda.so` (Linux) o `nvcuda.dll` (Windows) esté disponible en la ruta de búsqueda de la biblioteca dinámica (se requiere la API del controlador CUDA para la carga de PTX).
- Cree un runner dedicado con las etiquetas: gpu, nvidia, linux, x64.
- Instale la cadena de herramientas de Rust (rustup) y asegúrese de que `cargo` esté en el PATH.
- Verifique con `nvidia-smi` y `nvcc --version` (opcional para compilaciones FFI).
- Ejecute el servicio de runner de GitHub Actions y adjunte las etiquetas anteriores.

Pasos para macOS (Metal) (para futuras pruebas wgpu)
- Utilice macOS con Apple Silicon.
- Instale la cadena de herramientas de Rust.
- Etiquete el runner: gpu, macos, arm64, metal.

Configuración del repositorio
- Los trabajos de GPU se definen en `.github/workflows/gpu.yml` y apuntan a runners autohospedados por etiqueta.
- Las características dependientes de CUDA están detrás de `--features cuda` y están desactivadas por defecto.

Validación
- Ejecute `cargo test -p scir-gpu` para validar las abstracciones respaldadas por CPU.
- Para compilaciones CUDA (futuro), ejecute `cargo test -p scir-gpu --features cuda` en un runner con capacidad CUDA.

Opción AWS CodeBuild (GPU)
- Cree un proyecto de CodeBuild usando una imagen ECR personalizada basada en una imagen base de NVIDIA CUDA (por ejemplo, `nvidia/cuda:12.X-devel-ubuntu22.04`) con la cadena de herramientas de Rust instalada.
- Habilite el modo privilegiado en la configuración del entorno para que el contenedor pueda acceder a las GPU (el tiempo de ejecución de NVIDIA en el host debe estar configurado).
- Elija una configuración de cómputo habilitada para GPU según lo admitido por su cuenta y región de AWS.
- Agregue el `buildspec.gpu.yml` de este repositorio al proyecto; ejecuta `cargo build/test` con `--features cuda`.
- Asegúrese de que la AMI del host y la flota de CodeBuild admitan GPU y que el controlador CUDA esté instalado en el host subyacente.
