# AGENTS

- Exécutez `pytest` pour les modifications Python et `bash -n` pour les scripts shell modifiés.
- Gardez les messages de commit descriptifs.
- Ne commettez pas d'artefacts binaires ; utilisez plutôt des scripts ou des encodages.
- Mettez à jour `PLAN.md` avec l'avancement lorsque les tâches sont terminées.

Politique de formatage
- Fiez-vous toujours à `cargo fmt` pour le formatage Rust. C'est la source de vérité.
- Le hook de pré-commit exécute `cargo fmt --all` pour auto-formater les modifications ; l'intégration continue applique `cargo fmt -- --check`.
- Les agents et les contributeurs n'ont pas besoin de formater Rust manuellement ; exécutez plutôt le formateur.
