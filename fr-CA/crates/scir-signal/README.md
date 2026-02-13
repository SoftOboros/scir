scir-signal

Vue d'ensemble
- Traitement du signal pour SciR : conception de filtres classiques (Butterworth, Chebyshev I, Bessel), filtrage par sections de second ordre, filtrage `filtfilt` à phase nulle et `resample_poly`.
- Priorité à la parité : validé par rapport à SciPy via des fixtures.
- GPU optionnel : le chemin FIR peut s'auto-diriger vers CUDA lorsqu'il est activé.

Liens
- Dépôt : https://github.com/SoftOboros/scir
- Docs : https://docs.rs/scir-signal
