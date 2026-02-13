```markdown
Docker Compose para Desarrollo CUDA

Uso
- Construir e iniciar un contenedor de desarrollo CUDA con sccache habilitado:
  - docker compose -f docker-compose.gpu.yml up -d --build
  - docker compose -f docker-compose.gpu.yml exec gpu-dev bash

Entorno
- sccache está instalado y habilitado a través de `RUSTC_WRAPPER=sccache`.
- El volumen de caché local está montado en `/var/cache/sccache`.
- Para usar S3, exporte las variables de entorno antes de `docker compose`:
  - SCCACHE_BUCKET, SCCACHE_REGION, SCCACHE_S3_KEY_PREFIX opcional (sin barra final; por ejemplo, `/scir/x86_64-unknown-linux-gnu`)
  - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (o rol/IMDS en EC2)

GPUs
- El servicio solicita GPUs a través de `device_requests: [capabilities: ["gpu"]]`.
- Asegúrese de que NVIDIA Container Toolkit esté instalado en el host.
- Si es necesario, pruebe `runtime: nvidia` o `gpus: all` en lugar de `device_requests`.

Volúmenes
- Los volúmenes `cargo-registry` y `cargo-git` persisten las descargas de Cargo.
- El volumen `sccache` persiste los artefactos compilados en todas las ejecuciones.

Ejemplos
- Pruebas de CPU:
  - cargo test -p scir-core
- Pruebas CUDA (si la GPU está disponible):
  - cargo test -p scir-gpu --features cuda
```
