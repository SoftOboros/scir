```markdown
# Contribuer à SciR

- Assurez-vous que le submodule git SciPy est initialisé (`scripts/setup-ci-env.sh` s'en occupe) ou exécutez `git submodule update --init --depth 1 scipy`.
- Exécutez `python scripts/gen_fixtures.py -n 8` pour regénérer les fixtures de test au besoin.
- Exécutez `pytest` et `cargo test` avant de commettre les changements.
- Installez `pre-commit` et exécutez `pre-commit run --files <files>` sur les fichiers mis en scène.
- Suivez les instructions de AGENTS.md et ne commettez pas les binaires générés.
- Formatez le code Rust avec `cargo fmt` et gardez les fonctions ciblées et documentées.

## Tests GPU

- Les chemins CUDA sont facultatifs et désactivés par défaut. La plupart des contributeurs peuvent travailler uniquement avec le CPU.
- Pour exécuter des tests GPU localement sur un hôte compatible CUDA avec des pilotes NVIDIA :
  - `cargo test -p scir-gpu --features cuda`
  - Exemple d'intégration facultatif : `cargo run -p scir --features gpu --example fir_gpu`
- Les runners hébergés par GitHub n'exposent pas les GPU ; utilisez un runner auto-hébergé. Voir `docs/gpu-runner.md`.
- AWS CodeBuild est une alternative pour le CI GPU. Voir `docs/codebuild-gpu.md` et `buildspec.gpu.yml`.
```
