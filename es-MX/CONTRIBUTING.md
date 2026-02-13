```markdown
# Contribuyendo a SciR

- Asegúrate de que el submódulo de git de SciPy esté inicializado (`scripts/setup-ci-env.sh` se encarga de esto) o ejecuta `git submodule update --init --depth 1 scipy`.
- Ejecuta `python scripts/gen_fixtures.py -n 8` para regenerar los fixtures de prueba según sea necesario.
- Ejecuta `pytest` y `cargo test` antes de confirmar los cambios.
- Instala pre-commit y ejecuta `pre-commit run --files <files>` en los archivos preparados.
- Sigue las instrucciones de AGENTS.md y no confirmes los binarios generados.
- Formatea el código Rust con `cargo fmt` y mantén las funciones enfocadas y documentadas.

## Pruebas de GPU

- Las rutas CUDA son opcionales y están desactivadas por defecto. La mayoría de los contribuidores pueden trabajar solo con CPU.
- Para ejecutar pruebas de GPU localmente en un host compatible con CUDA con controladores NVIDIA:
  - `cargo test -p scir-gpu --features cuda`
  - Ejemplo opcional general: `cargo run -p scir --features gpu --example fir_gpu`
- Los ejecutores alojados en GitHub no exponen GPUs; usa un ejecutor autoalojado. Consulta `docs/gpu-runner.md`.
- AWS CodeBuild es una alternativa para CI de GPU. Consulta `docs/codebuild-gpu.md` y `buildspec.gpu.yml`.
```
