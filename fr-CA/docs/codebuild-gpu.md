AWS CodeBuild pour CUDA/GPU

Aperçu
- Utilisez une image ECR basée sur une image de développement NVIDIA CUDA avec la chaîne d'outils Rust préinstallée. Ce dépôt fournit `ci/docker/Dockerfile.cuda` comme point de départ.
- Activez le mode privilégié afin que le conteneur de build puisse accéder aux GPU via le runtime NVIDIA de l'hôte.
- Choisissez un type de calcul compatible GPU dans votre région (les noms varient; consultez la documentation AWS ou vos limites de compte). L'exemple JSON utilise `BUILD_GENERAL1_LARGE` comme espace réservé.

Étapes
1) Construire et pousser l'image Docker vers ECR
   - aws ecr create-repository --repository-name scir-cuda-rust
   - docker build -t scir-cuda-rust:latest -f ci/docker/Dockerfile.cuda .
   - docker tag scir-cuda-rust:latest YOUR_AWS_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/scir-cuda-rust:latest
   - aws ecr get-login-password --region YOUR_REGION | docker login --username AWS --password-stdin YOUR_AWS_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com
   - docker push YOUR_AWS_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/scir-cuda-rust:latest

2) Créer le projet CodeBuild
   - Utilisez `ci/codebuild/project.example.json` comme modèle; remplacez `YOUR_ORG`, l'ID du compte, la région et le rôle de service.
   - Assurez-vous que `privilegedMode` est vrai et que votre type de calcul prend en charge les builds GPU.
   - Définissez le buildspec sur `buildspec.gpu.yml` dans ce dépôt.

3) Exécuter le build
   - Fournissez la source comme ce dépôt GitHub (ou un dépôt CodeCommit connecté).
   - Démarrez le build; il exécutera `cargo build/test -p scir-gpu --features cuda`.
   - Les tests CUDA sont ignorés avec élégance si le GPU n'est pas disponible.

Remarques
- Le conteneur s'attend à ce que le pilote CUDA soit présent sur l'hôte; les flottes GPU de CodeBuild le fournissent lorsque le calcul GPU est activé et que le mode privilégié est activé.
- Pour les dépôts GitHub privés, configurez une accréditation de source dans CodeBuild ou utilisez une intégration CodePipeline.
