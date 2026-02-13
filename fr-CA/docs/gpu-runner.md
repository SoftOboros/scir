GPU CI Runner (auto-hébergé) Configuration

Aperçu
- Les runners hébergés par GitHub n'exposent pas les GPU. Utilisez une machine auto-hébergée avec un GPU NVIDIA pour les tâches CUDA ou une machine Apple Silicon/macOS pour Metal.

Étapes Linux (CUDA)
- Installez le pilote NVIDIA et le kit d'outils CUDA correspondant à votre GPU (par exemple, CUDA 12.x).
- Assurez-vous que `libcuda.so` (Linux) ou `nvcuda.dll` (Windows) est disponible dans le chemin de recherche de la bibliothèque dynamique (l'API du pilote CUDA est requise pour le chargement PTX).
- Créez un runner dédié avec les étiquettes : gpu, nvidia, linux, x64.
- Installez la chaîne d'outils Rust (rustup) et assurez-vous que `cargo` est dans le PATH.
- Vérifiez avec `nvidia-smi` et `nvcc --version` (facultatif pour les builds FFI).
- Exécutez le service de runner GitHub Actions et attachez les étiquettes ci-dessus.

Étapes macOS (Metal) (pour les futurs tests wgpu)
- Utilisez macOS avec Apple Silicon.
- Installez la chaîne d'outils Rust.
- Étiquetez le runner : gpu, macos, arm64, metal.

Configuration du dépôt
- Les tâches GPU sont définies dans `.github/workflows/gpu.yml` et ciblent les runners auto-hébergés par étiquette.
- Les fonctionnalités dépendantes de CUDA sont derrière `--features cuda` et sont désactivées par défaut.

Validation
- Exécutez `cargo test -p scir-gpu` pour valider les abstractions prises en charge par le CPU.
- Pour les builds CUDA (futur), exécutez `cargo test -p scir-gpu --features cuda` sur un runner compatible CUDA.

Option AWS CodeBuild (GPU)
- Créez un projet CodeBuild en utilisant une image ECR personnalisée basée sur une image de base NVIDIA CUDA (par exemple, `nvidia/cuda:12.X-devel-ubuntu22.04`) avec la chaîne d'outils Rust installée.
- Activez le mode privilégié dans les paramètres d'environnement afin que le conteneur puisse accéder aux GPU (le runtime NVIDIA sur l'hôte doit être configuré).
- Choisissez une configuration de calcul compatible GPU telle que prise en charge par votre compte et votre région AWS.
- Ajoutez le fichier `buildspec.gpu.yml` de ce dépôt au projet ; il exécute `cargo build/test` avec `--features cuda`.
- Assurez-vous que l'AMI hôte et la flotte CodeBuild prennent en charge les GPU et que le pilote CUDA est installé sur l'hôte sous-jacent.
