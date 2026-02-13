# AGENTES

- Ejecutar `pytest` para cambios en Python y `bash -n` para scripts de shell modificados.
- Mantener los mensajes de commit descriptivos.
- No hacer commit de artefactos binarios; usar scripts o codificaciones en su lugar.
- Actualizar `PLAN.md` con el progreso cuando las tareas estén completas.

Política de formato
- Siempre confiar en `cargo fmt` para el formato de Rust. Es la fuente de verdad.
- El hook de pre-commit ejecuta `cargo fmt --all` para auto-formatear los cambios; CI hace cumplir `cargo fmt -- --check`.
- Los agentes y colaboradores no necesitan formatear Rust a mano; en su lugar, ejecutar el formateador.
