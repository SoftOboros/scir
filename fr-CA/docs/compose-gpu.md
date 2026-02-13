```markdown
Docker Compose pour le développement CUDA

Utilisation
- Construire et démarrer un conteneur de développement prêt pour CUDA avec sccache activé :
  - docker compose -f docker-compose.gpu.yml up -d --build
  - docker compose -f docker-compose.gpu.yml exec gpu-dev bash

Environnement
- sccache est installé et activé via `RUSTC_WRAPPER=sccache`.
- Le volume de cache local est monté à `/var/cache/sccache`.
- Pour utiliser S3, exportez les variables d'environnement avant `docker compose` :
  - SCCACHE_BUCKET, SCCACHE_REGION, SCCACHE_S3_KEY_PREFIX facultatif (pas de barre oblique finale ; par exemple, `/scir/x86_64-unknown-linux-gnu`)
  - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (ou rôle/IMDS dans EC2)

GPUs
- Le service demande des GPUs via `device_requests: [capabilities: ["gpu"]]`.
- Assurez-vous que NVIDIA Container Toolkit est installé sur l'hôte.
- Si nécessaire, essayez `runtime: nvidia` ou `gpus: all` au lieu de `device_requests`.

Volumes
- Les volumes `cargo-registry` et `cargo-git` persistent les téléchargements de Cargo.
- Le volume `sccache` persiste les artefacts compilés entre les exécutions.

Exemples
- Tests CPU :
  - cargo test -p scir-core
- Tests CUDA (si GPU disponible) :
  - cargo test -p scir-gpu --features cuda
```
