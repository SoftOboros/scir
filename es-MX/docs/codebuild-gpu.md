AWS CodeBuild para CUDA/GPU

Visión general
- Utilice una imagen ECR basada en una imagen de desarrollo de NVIDIA CUDA con la cadena de herramientas Rust preinstalada. Este repositorio proporciona `ci/docker/Dockerfile.cuda` como punto de partida.
- Habilite el modo privilegiado para que el contenedor de compilación pueda acceder a las GPU a través del tiempo de ejecución de NVIDIA del host.
- Elija un tipo de cómputo habilitado para GPU en su región (los nombres varían; consulte la documentación de AWS o los límites de su cuenta). El JSON de ejemplo utiliza `BUILD_GENERAL1_LARGE` como marcador de posición.

Pasos
1) Compile y suba la imagen de Docker a ECR
   - aws ecr create-repository --repository-name scir-cuda-rust
   - docker build -t scir-cuda-rust:latest -f ci/docker/Dockerfile.cuda .
   - docker tag scir-cuda-rust:latest YOUR_AWS_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/scir-cuda-rust:latest
   - aws ecr get-login-password --region YOUR_REGION | docker login --username AWS --password-stdin YOUR_AWS_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com
   - docker push YOUR_AWS_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/scir-cuda-rust:latest

2) Cree el proyecto CodeBuild
   - Utilice `ci/codebuild/project.example.json` como plantilla; reemplace `YOUR_ORG`, ID de cuenta, región y rol de servicio.
   - Asegúrese de que `privilegedMode` sea verdadero y que su tipo de cómputo admita compilaciones de GPU.
   - Establezca el buildspec en `buildspec.gpu.yml` en este repositorio.

3) Ejecute la compilación
   - Proporcione la fuente como este repositorio de GitHub (o un repositorio de CodeCommit conectado).
   - Inicie la compilación; ejecutará `cargo build/test -p scir-gpu --features cuda`.
   - Las pruebas de CUDA se omiten correctamente si la GPU no está disponible.

Notas
- El contenedor espera que el controlador CUDA esté presente en el host; las flotas de GPU de CodeBuild lo proporcionan cuando la computación de GPU está habilitada y el modo privilegiado está activado.
- Para repositorios privados de GitHub, configure una credencial de origen en CodeBuild o utilice una integración de CodePipeline.
