scir-core

Vue d'ensemble
- Utilitaires de base pour SciR (SciPy reconstruit en Rust) : aides à la tolérance numérique, utilitaires de nombres complexes, et une macro réutilisable `assert_close!` utilisée dans tous les crates.

Points saillants
- Comparaison cohérente des nombres à virgule flottante via `assert_close!` pour les tranches et les tableaux ndarray (réels et complexes).
- Blocs de construction minimaux et éprouvés partagés par `scir-fft`, `scir-signal`, `scir-optimize`, et d'autres.

Liens
- Dépôt : https://github.com/SoftOboros/scir
- Documentation : https://docs.rs/scir-core
