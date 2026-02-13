```markdown
# SciR

SciR vise à réimplémenter des éléments centraux de SciPy en Rust avec une approche axée sur la parité et des backends GPU optionnels.

Licence
- MIT. Voir le fichier LICENSE à la racine du dépôt. Tous les crates déclarent `license = "MIT"` dans leur Cargo.toml.

Démarrage rapide
- Construire l'espace de travail : `cargo build` (la première exécution récupérera les dépendances).
- Exécuter les tests : `cargo test` (les tests GPU sont conditionnels et sont ignorés sans GPU).

Crate parapluie et fonctionnalité GPU
- Utiliser `crates/scir` comme un parapluie pratique qui réexporte les crates de base.
- Activer la fonctionnalité agrégée `gpu` pour activer les chemins supportés par CUDA et les API GPU dans les crates dépendantes :
  - CPU : `cargo run -p scir --example fir_gpu`
  - CUDA : `cargo run -p scir --features gpu --example fir_gpu`
  - Démo élément par élément : `cargo run -p scir --example elementwise_gpu` (ajouter `--features gpu` pour essayer CUDA)
  - Benchmark FIR : `cargo run -p scir --example fir_bench` (ajouter `--features gpu` pour chronométrer CUDA)

Aperçu du support GPU
- Les backends GPU sont désactivés par défaut et protégés par une fonctionnalité.
- Couverture CUDA actuelle dans `scir-gpu` : addition/multiplication élément par élément et FIR par lots (f32) via l'API CUDA Driver + PTX intégré.
- Les aides d'auto-dispatch acheminent les opérations vers CUDA lorsque `Device::Cuda` est sélectionné ; elles se replient sur le CPU si le backend est indisponible.

CI GPU auto-hébergé et AWS CodeBuild
- Les GitHub Actions pour les GPU nécessitent des runners auto-hébergés ; voir `.github/workflows/gpu.yml` et `docs/gpu-runner.md` pour la configuration et les étiquettes.
- Alternativement, utiliser AWS CodeBuild avec une image de conteneur compatible CUDA. Voir :
  - `ci/docker/Dockerfile.cuda` (image de base CUDA+Rust)
  - `buildspec.gpu.yml` (étapes de construction/test)
  - `ci/codebuild/project.example.json` et `docs/codebuild-gpu.md` (modèle de projet et guide)

Parité via des fixtures
- Nous générons des fixtures SciPy et les testons par rapport à elles. Voir `PLAN.md` pour plus de détails.

Ce dépôt suit actuellement les travaux d'échafaudage préliminaires, y compris les scripts pour générer les fixtures de référence.

## Commencer

Exécutez `scripts/setup-ci-env.sh` pour installer les prérequis et télécharger le sous-module SciPy à `/scipy`.
Installez les dépendances Python avec `pip install -r requirements.txt` et exécutez les tests via `pytest` et `cargo test`.

Si vous avez ignoré le script de configuration, initialisez le sous-module git SciPy (extrait à `/scipy`) avec :

```
git submodule update --init --depth 1 scipy
```

Générez des fixtures FFT de référence avec `python scripts/gen_fixtures.py --sizes 8 16` (les fichiers atterrissent dans `fixtures/`, qui est ignoré par git).
Générez des fixtures d'optimisation avec `python scripts/gen_optimize_fixtures.py` et des fixtures de signal avec `python scripts/gen_signal_fixtures.py` (données Butterworth, Chebyshev, Bessel, filtfilt et `resample_poly`). Les fixtures d'optimisation couvrent les résultats Nelder–Mead, BFGS et L-BFGS.

Les fixtures sont stockées sous forme de tableaux `.npy`. Pour chaque taille `<n>`, le script produit :
- `fft_input_<n>.npy`
- `fft_output_<n>.npy`
- `ifft_output_<n>.npy`
- `rfft_output_<n>.npy`
- `irfft_output_<n>.npy`

Les sorties complexes utilisent le dtype complexe natif de NumPy. Régénérez les fixtures au besoin ; le répertoire reste non suivi.

## Licence

Double licence sous Apache-2.0 et MIT.
```
