```markdown
# SciR (SciPy en Rust) — Plan de développement (Révisé et étendu)

> **Statut :** Révisé pour assurer la cohérence avec le fil de discussion ; corrections des hallucinations ; précisions ajoutées là où c'était nécessaire ; CI + listes de contrôle des phases ajoutées.

## 0) Référence amont et contraintes (Source de vérité)
- **Projet amont :** SciPy — GitHub : `scipy/scipy`.
- **Licence :** BSD-3-Clause (permissive). Nous **ne devons pas** copier le code C/Fortran de SciPy ; nous réimplémentons les algorithmes et validons le comportement via des tests/fixtures.
- **Position du GPU en amont :** SciPy lui-même est **CPU uniquement** ; l'accélération GPU en Python se fait généralement via **CuPy** ou d'autres frameworks (JAX, PyTorch). C'est notre objectif de différenciation.
- **Couverture de test en amont :** Aucun pourcentage officiel de couverture n'est publié ; la couverture est large et activement maintenue. Notre approche traite le **comportement** de SciPy comme la spécification via les fixtures.

---

## 1) Objectifs et différenciation
- **Priorité à la parité :** Reproduire les API SciPy bien connues (en commençant par `fft`, `signal`, `optimise`, `linalg`) avec l'ergonomie de Rust et des types explicites.
- **Conversion mécanique via des tests :** Utiliser SciPy pour générer des **fixtures** canoniques ; exécuter des implémentations Rust identiques par rapport à celles-ci. Le comportement de Python devient l'**oracle**.
- **Marge de performance :** Après la parité, ajouter des **optimisations natives Rust** (tranches, SIMD, Rayon) et des **backends GPU** opt-in (CUDA et/ou calcul portable via `wgpu`).
- **Avantage marketing :** « SciR : SciPy reconstruit pour Rust — sûr, rapide, prêt pour le GPU. »

---

## 2) Espace de travail et carte des modules (Ajusté pour la clarté)
```
scir/                              # Crate parapluie (ré-exporte)
  Cargo.toml
  crates/
    scir-core/                     # Commun : traits, erreurs, types, utilitaires de tolérance
    scir-nd/                       # Interopérabilité ndarray et conversions (base CPU)
    scir-fft/                      # API FFT (CPU : rustfft ; backends GPU optionnels)
    scir-signal/                   # Filtres (IIR/FIR), rééchantillonnage, fenêtres
    scir-optimize/                 # minimise (Nelder–Mead, BFGS/L-BFGS), least_squares
    scir-linalg/                   # Algèbre linéaire (faer / ndarray-linalg / BLAS)
    scir-gpu/                      # (optionnel) backends et gestion des périphériques tableau (voir §6)
```
**Notes**
- `scir-nd` isole les conversions `ndarray`/forme/pas pour que les crates de calcul restent concentrées.
- `scir-gpu` est une couche *mince* : mémoire du périphérique, transferts et répartition du backend. Le calcul réside dans des sous-modules à fonctionnalité restreinte de chaque crate de domaine.

---

## 3) Dépendances externes (Corrigées et conservatrices)
- **Tableaux :** `ndarray` (base CPU). Envisager `ndarray-linalg` pour les opérations basées sur Lapack.
- **FFT :** `rustfft` (CPU). (Le chemin GPU est personnalisé, voir §6.)
- **Linalg (CPU) :** `faer` (Rust pur) et/ou `ndarray-linalg` (BLAS/LAPACK via `openblas-src`/système). Gater derrière des fonctionnalités.
- **Parallèle :** `rayon`.
- **Types numériques :** `num-traits`, `num-complex`.
- **SIMD :** préférer les crates comme `wide` (portable) ; suivre la stabilisation de `portable_simd` de Rust pour l'avenir.
- **Serde :** optionnel (`serde`, `serde_json`, `ndarray-npy`) pour les fixtures.
- **GPU :** voir §6. Éviter de revendiquer une crate cuFFT spécifique ; nous pourrions écrire un FFI minimal s'il n'existe pas de bindings maintenus.

---

## 4) Modèle d'erreur et surface d'API
- Une énumération d'erreurs publique par crate réexportée via `scir-core` :
```rust
#[derive(thiserror::Error, Debug)]
pub enum ScirError {
    #[error("invalid argument: {0}")] InvalidArgument(String),
    #[error("numerical failure: {0}")] NumericalFailure(String),
    #[error("backend not available: {0}")] BackendUnavailable(String),
}
```
- Les fonctions utilisent des **structures explicites** pour les options afin d'éviter toute ambiguïté (par exemple, `FftPlan`, `FilterDesignOpts`, `MinimizeOpts`).
- Nombres complexes : `num_complex::Complex<f64>` / `<f32>` avec des alias de type.
- **Nommage :** refléter les noms des fonctions SciPy en `snake_case` et regrouper par module (`scir::signal::filtfilt`, etc.).

---

## 5) Parité via les fixtures (Détaillé et non ambigu)
### 5.1 Génération de fixtures (Python)
- Épingler les versions pour garantir des résultats déterministes : par exemple, `numpy==1.x`, `scipy==1.y`.
- Pour chaque fonction cible, créer un script Python qui :
  1) Génère des entrées représentatives, y compris les cas limites.
  2) Appelle l'implémentation SciPy.
  3) Sauvegarde les **entrées** et les **sorties attendues** dans `fixtures/` au format `.npy` ou JSON.
- **Données complexes** en JSON : encoder sous forme de tuples `[re, im]` ou utiliser `.npy` pour éviter les encodages personnalisés.
- **Exemple :** (pseudo-extrait)
```python
# export_fixtures_fft.py
import numpy as np
from scipy.fft import fft, ifft, rfft, irfft
np.random.seed(0)
x = np.random.randn(1024).astype(np.float64)
np.save('fixtures/fft/x_f64.npy', x)
np.save('fixtures/fft/fft_expected.npy', fft(x))
```

### 5.2 Tests de parité (Rust)
- Chaque crate charge ses fixtures et affirme les tolérances :
```rust
let x: Array1<f64> = read_npy("fixtures/fft/x_f64.npy")?;
let y_ref: Array1<Complex64> = read_npy("fixtures/fft/fft_expected.npy")?;
let y = scir::fft::fft(x.view());
assert_close!(y, y_ref, atol=1e-9, rtol=1e-7);
```
- Fournir une macro `assert_close!` dans `scir-core` pour des tolérances numériques cohérentes.
- **Précautions spécifiques au signal :** gestion des cas limites de `filtfilt` (le schéma de remplissage par défaut dans SciPy est critique) :
  - `padtype="odd"`, `padlen = 3 * (max(len(a), len(b)) - 1)` sauf si remplacé.
  - Utiliser la parité de la méthode **Gustafsson** là où SciPy le fait.

### 5.3 Portée des fixtures
- `fft` : réel/complexe, tailles prime/puissance de deux, décalages.
- `signal` : conception `butter/cheby1/bessel` → SOS, `sosfilt`, `filtfilt`, `resample_poly`.
- `optimize` : `rosenbrock`, `himmelblau` avec des minima connus ; ensembles de données `least_squares`.
- `linalg` : problèmes petits/moyens pour `svd`, `qr`, `solve`, matrices conditionnées.

---

## 6) Extension GPU (Corrigée et fondée)
**Vérification de la réalité :** Les exécuteurs hébergés sur GitHub **n'exposent généralement pas** les GPU. La CI GPU nécessite des exécuteurs **auto-hébergés** ou une CI tierce avec GPU. Notre plan en tient compte.

### 6.1 Stratégie de backend
- **CUDA (NVIDIA) :** Utiliser `cudarc` (si suffisant pour nos besoins) ou implémenter un FFI minimal vers cuBLAS/cuFFT si des crates maintenues ne sont pas disponibles. Nous ne revendiquerons pas une crate existante avant vérification ; prévoir un FFI via `bindgen` comme solution de repli.
- **Calcul portable :** `wgpu` (Vulkan/Metal/DX12/WebGPU) avec des noyaux de calcul WGSL. FFT sur `wgpu` est non trivial ; commencer par les opérations ponctuelles, les convolutions, GEMM ; explorer les noyaux FFT radix-N comme jalon ultérieur.
- **OpenCL :** `ocl` ou `opencl3` comme alternative là où CUDA n'est pas disponible.

### 6.2 Abstraction de tableau de périphérique
Au lieu de stocker des `wgpu::Buffer` bruts dans une énumération générique, nous définissons des **tableaux de périphériques formés** avec dtype et strides pour refléter la sémantique du CPU :
```rust
pub enum Device {
    Cpu,
    Cuda(CudaCtx),
    Wgpu(WgpuCtx),
}

pub struct DArray {
    pub device: Device,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub dtype: DType,           // F32, F64, C64, C128
    // handle opaque : CPU -> stockage ndarray ; GPU -> tampon spécifique au backend
}
```
- Fournir des transferts `to_cpu()`/`to_device(Device)` avec synchronisation explicite.
- Les API de haut niveau acceptent les tableaux CPU ; les chemins GPU sont **optionnels** via des flags de fonctionnalités et des conversions explicites.

### 6.3 Cibles et ordre du GPU
1) **Par éléments / map-reduce** (gain facile) : échelle, addition, multiplication, abs2, réductions.
2) **Convolution / FIR** (1D/2D) : FIR par lots via le pavage ; plus tard IIR (avec rétroaction) en utilisant un préfixe parallèle ou un traitement par blocs.
3) **GEMM** : exploiter cuBLAS (CUDA) ou les noyaux WGSL réglés.
4) **FFT** : chemin CUDA en premier (via cuFFT FFI) ; FFT `wgpu` plus tard (nécessite des noyaux personnalisés).

### 6.4 Validation
- Toujours vérifier **GPU == CPU** dans les tolérances.
- Fournir des graines déterministes et des charges de travail dimensionnées pour rendre les comparaisons inter-backends fiables.

---

## 7) Optimisations CPU (Pragmatique)
- **Découpage/emprunt :** préférer les chemins chauds `&[T]`/`&mut [T]` ; s'adapter aux vues `ndarray` pour l'API.
- **SIMD :** utiliser `wide` ou les intrinsèques spécifiques au backend ; gater avec des fonctionnalités.
- **Parallèle :** `rayon` pour les opérations embarrassingly parallel.
- **Stabilité numérique :** correspondre aux valeurs par défaut de SciPy (par exemple, transformations bilinéaires, mise à l'échelle SOS) pour réduire la dérive.

---

## 8) Benchmarks et reproductibilité
- Benchmarks **`criterion`** reflétant les fixtures (tailles FFT, longueurs de filtre, dimensions de l'optimiseur).
- Comparer avec **SciPy** (CPU) sur la même machine pour une vérification de la réalité ; documenter le matériel et les versions.
- Enregistrer le débit/la latence de base ; suivre les régressions via les artefacts de tendance de la CI.

---

## 9) CI et tests de plateforme (Révisé et exploitable)
### 9.1 Matrice (CPU)
- **OS :** ubuntu-latest, macos-latest, windows-latest.
- **Rust :** stable, beta, nightly.
- **Fonctionnalités :** `pure-rust` (par défaut), `faer`, `blas`.

### 9.2 Actions GitHub (CPU) — Exemple
```yaml
name: ci
on: [push, pull_request]

jobs:
  cpu:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, beta, nightly]
        features: ["", "faer", "blas"]
    steps:
      - uses: actions/checkout @ops/packer/BUGSBUNNY-v4-CHANGELOG.md
      - uses: dtolnay/rust-toolchain @backend/istate/tests/data/SCXML-tutorial/Tests/TestsTableW3C.csv
        with: { toolchain: ${{ matrix.rust }} }
      - uses: Swatinem/rust-cache @v2
      - run: cargo build --verbose --features "${{ matrix.features }}"
      - run: cargo test  --verbose --features "${{ matrix.features }}"
```

### 9.3 CI GPU (Auto-hébergé)
```yaml
  gpu:
    if: github.repository_owner == 'YOUR_ORG'  # éviter les forks
    runs-on: [self-hosted, linux, x64, gpu, nvidia]
    env:
      RUST_LOG: info
    steps:
      - uses: actions/checkout @ops/packer/BUGSBUNNY-v4-CHANGELOG.md
      - uses: dtolnay/rust-toolchain @backend/istate/tests/data/SCXML-tutorial/Tests/TestsTableW3C.csv
        with: { toolchain: stable }
      - run: nvidia-smi  # vérification de bon fonctionnement
      - run: cargo test --features cuda --package scir-fft
```
- Fournir de la documentation pour la configuration de l'**exécuteur GPU auto-hébergé** (pilotes CUDA, toolkit, étiquettes).
- Pour les tests `wgpu` sur macOS avec des séries M, exécuter un exécuteur macOS auto-hébergé.

### 9.4 Parité Python en CI
- Exécuter Python une fois pour **(re)générer les fixtures** (versions épinglées) et les télécharger comme artefact CI utilisé par les jobs Rust.
```yaml
  fixtures:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout @ops/packer/BUGSBUNNY-v4-CHANGELOG.md
      - uses: actions/setup-python @v5
        with: { python-version: '3.11' }
      - run: python -m pip install --upgrade pip
      - run: pip install numpy==1.X scipy==1.Y pytest
      - run: python tools/export_fixtures.py
      - uses: actions/upload-artifact @ops/packer/BUGSBUNNY-v4-CHANGELOG.md
        with:
          name: scir-fixtures
          path: fixtures/**
```
- Les jobs CPU/GPU **téléchargent** l'artefact `scir-fixtures` avant `cargo test`.

### 9.5 Couverture et rapports
- Utiliser `cargo-llvm-cov` pour la couverture Rust ; `pytest --cov` pour le harnais Python si nécessaire.
- Publier avec `codecov/codecov-action @v4` (facultatif).

---

## 10) Licences et conformité (Explicite, exploitable)

> **Intention :** Garder `scir` juridiquement propre et adoptable par les entreprises tout en exploitant SciPy comme un oracle *comportemental*.

### 10.1 Provenance du code (Ce qui est / n'est pas autorisé)
- **Ne PAS copier** le code d'implémentation de SciPy (C/Fortran/Python) ou traduire ligne par ligne.
- **OK :** Implémenter des algorithmes à partir de **documents, de normes et de la documentation SciPy** (les concepts ≠ code). Citer les sources dans `ALGORITHMS.md`.
- **OK :** Générer et stocker des **fixtures de test** en exécutant SciPy ; les fixtures sont des *données*, pas du code. Conserver la provenance (voir 10.3).
- **OK :** Utiliser des crates tierces avec des **licences permissives** (MIT/Apache-2.0/BSD-2/3/ISC/Zlib). Éviter le copyleft fort/viral.

### 10.2 Licences pour `scir`
- Choisir **une seule** licence pour toutes les crates (**MIT**). Rester cohérent sur l'ensemble du dépôt.
- Ajouter des en-têtes SPDX à chaque fichier source :
  ```rust
  // SPDX-License-Identifier: Apache-2.0
  ```
- `Cargo.toml` par crate :
  ```toml
  license = "MIT"
  ```
- Fichier `LICENCE` de niveau supérieur correspondant à la licence choisie ; pas de divergence par crate.

### 10.3 Politique de fixtures (SciPy comme oracle)
- **Stocker avec provenance** à côté de chaque lot de fixtures (YAML ou JSON) :
  ```yaml
  generator: tools/export_fixtures.py
  python: 3.11.x
  numpy: 1.x.y
  scipy: 1.y.z
  seed: 123456
  os: linux-x86_64
  date: 2025-09-08
  script_sha256: <hachage-de-l'exportateur>
  ```
- Garder les **scripts de régénération** dans `tools/` et épingler les versions des dépendances.
- Utiliser des **données synthétiques** pour les tests ; éviter les jeux de données protégés par des droits d'auteur, sauf si les licences sont compatibles et enregistrées.

### 10.4 Sous-modules et vendoring
- Installer SciPy via `pip` pour les tests/références locaux ; éviter les sous-modules git sauf si absolument nécessaire.
- **Ne PAS vendre** le code SciPy dans les crates `scir`. Garder les outils Python sous `tools/` et hors des crates publiables.
- S'assurer que les crates sont publiées **sans** les sources SciPy (vérifier `include`/`exclude` dans `Cargo.toml`).

### 10.5 Dépendances tierces
- Maintenir `licence-allowlist.toml` et l'appliquer via CI (`cargo-deny`). Liste blanche suggérée : `Apache-2.0`, `MIT`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`, `Zlib`.
- **Éviter** GPL/AGPL. Si LGPL/MPL est inévitable, gater derrière des fonctionnalités **non par défaut** et documenter clairement les obligations.
- BLAS/LAPACK : préférer **faer** (Rust pur) ou **OpenBLAS**/**BLIS** via `ndarray-linalg` (permissif). **Ne PAS** lier en dur à des bibliothèques GPL.

### 10.6 Backends GPU et SDKs fournisseurs
- **CUDA :** Lier via FFI ; **ne pas** inclure les en-têtes ou les binaires NVIDIA. Exiger des utilisateurs qu'ils installent CUDA sous l'EULA de NVIDIA ; détecter à la compilation.
- **wgpu/OpenCL :** Vérifier la compatibilité de la licence ; préférer les crates permissives. Garder le code du backend dans des fonctionnalités optionnelles (`cuda`, `wgpu`, `opencl`).
- La sélection à l'exécution ne doit **pas** forcer l'inclusion de composants propriétaires dans les builds par défaut.

### 10.7 Documentation, exemples et marques de commerce
- **Ne PAS copier** la documentation SciPy verbatim. Paraphraser et attribuer : « Comportement aligné avec SciPy 1.y.z. » Citer uniquement de courts fragments si nécessaire avec attribution.
- **Marque :** **Ne PAS** impliquer d'affiliation. Utiliser un avertissement dans le README : « `scir` n'est pas affilié ou approuvé par le projet SciPy. » Éviter d'utiliser le nom **SciPy** dans les noms de packages.
- Envisager de licencier la documentation sous **CC-BY-4.0** (facultatif) et le code sous Apache-2.0/MIT ; documenter cette division dans le `README`.

### 10.8 Politique de PI des contributeurs
- Utiliser **DCO** (Developer Certificate of Origin) ou un CLA simple. Exiger `Signed-off-by` dans les commits via CI.
- Le modèle de PR doit inclure :
  - Confirmation du **travail original** (pas de code copié de SciPy ou d'autres sources restreintes).
  - Reconnaissance de la licence du projet et de la politique des fixtures.
  - Liste de contrôle pour l'ajout des en-têtes SPDX et la mise à jour des références `ALGORITHMS.md`.

### 10.9 Conformité automatisée en CI
- Ajouter une étape `cargo-deny` (licence + interdictions + sources). Échouer la CI en cas de violations.
- Ajouter la validation **REUSE**/SPDX (par exemple, `fsfe/reuse-action`) pour s'assurer que tous les fichiers portent des identifiants SPDX.
- S'assurer que `cargo package --list` ne montre aucun fichier SciPy ou SDK fournisseur inclus.

### 10.10 Hygiène de la publication
- Les artefacts de publication doivent inclure : `LICENCE`, `NOTICE` (si nécessaire), `ALGORITHMS.md`, les fichiers de provenance des fixtures et un `CITATION.cff` lisible par machine (facultatif mais utile pour le milieu universitaire).
- Les entrées du journal des modifications doivent noter toutes les nouvelles dépendances et leurs licences.

### 10.11 Liste de contrôle rapide de conformité (par PR)
- [ ] Pas de code amont copié ; sources citées dans `ALGORITHMS.md`.
- [ ] Fixtures régénérées avec SciPy/Numpy épinglés ; provenance mise à jour.
- [ ] En-têtes SPDX ajoutés/maintenus ; `licence` de `Cargo.toml` définie.
- [ ] `cargo-deny` passe ; aucune licence non autorisée.
- [ ] L'utilisation du SDK GPU/fournisseur est restreinte par des fonctionnalités non par défaut ; pas de binaires/en-têtes groupés.
- [ ] Le README contient un avertissement d'affiliation et un résumé des licences.
- [ ] `cargo package --list` audité ; les crates publiables sont exemptes de code Python/fournisseur.

## 11) Gestion des versions et publication
- Versionnement sémantique par crate ; synchronisation via un script de publication d'espace de travail.
- Publier les crates progressivement (`scir-core` → `scir-fft` → `scir-signal` …) au fur et à mesure que les fonctionnalités sont implémentées.
- Marquer les versions et joindre les résumés de benchmark et l'état de parité.

---

## 12) Feuille de route (Phases avec résultats)
**Phase 1 — Squelettes de parité (CPU) :**
- `scir-core`, `scir-nd`, `scir-fft` (fft/ifft/rfft/irfft), fenêtres `signal` de base.
- Fixtures pour FFT + filtres simples ; CI (CPU) en cours d'exécution.
- **Résultat :** Tests verts vs SciPy pour FFT ; >80% de couverture de crate.

**Phase 2 — Signal et Optimisation :**
- Conception de filtres (`butter`, `cheby1`, `bessel` → SOS), `sosfilt`, `filtfilt`, `resample_poly`.
- `optimize::minimize` (Nelder–Mead, BFGS/L-BFGS) sur Rosenbrock/Himmelblau.
- **Résultat :** Parité sur les routines de signal de base ; l'optimiseur correspond à SciPy dans les tolérances.

**Phase 3 — Linalg et backends :**
- Introduction du backend `faer` ; fonctionnalités BLAS/LAPACK optionnelles via `ndarray-linalg`.
- Benchmarks initiaux ; harnais de fuzzing pour les cas limites.
- **Résultat :** API linalg stables ; la matrice CI couvre les fonctionnalités faer/blas.

**Phase 4 — Fondations GPU :**
- Abstraction de tableau de périphérique ; par éléments + FIR par lots sur CUDA et/ou `wgpu`.
- CI GPU auto-hébergée ; vérifications de parité CPU↔GPU.
- **Résultat :** Premier gain GPU (accélération mesurable) avec API identique.

**Phase 5 — GPU avancé et FFT :**
- Chemin FFI cuFFT pour CUDA ; noyaux FFT `wgpu` exploratoires.
- GEMM via cuBLAS (CUDA) ou WGSL réglé ; intégration dans `linalg`.
- **Résultat :** Accélération GPU significative pour FFT/signal/GEMM.

**Phase 6 — Optimisation et durcissement :**
- SIMD, blocage conscient du cache, fusion de noyaux le cas échéant.
- Documentation, exemples, documentation API auto-générée ; publication v0.1–v0.3.

---

## 13) Expérience développeur (DX) et contribution
- Cibles `just` ou `make` : `just fixtures`, `just test`, `just bench`.
- `tools/` avec exportateurs Python ; épingler via `requirements.lock` ou `uv`.
- `CONTRIBUTING.md` avec guide de style, politique de tolérance et règles de provenance des fixtures.

---

## 14) Questions ouvertes (Suivi)
- Quel backend GPU prioriser en premier (part de marché CUDA vs portabilité `wgpu`) ?
- Où traçons-nous la limite de l'API SciPy (parité totale vs sous-ensemble pragmatique) ?
- Politique de tolérance par fonction/domaine (documentée dans `scir-core`).

---

## 15) Listes de contrôle de fin de phase

### Phase 1 — Base FFT CPU
- [ ] Utilitaires d'erreur/tolérance `scir-core`.
- [ ] Ponts `ndarray` `scir-nd`, assistants d'E/S `.npy`.
- [ ] `scir-fft` : fft/ifft/rfft/irfft (+ assistants de décalage).
- [ ] Fixtures Python pour FFT (tailles : 64…65536 ; réel et complexe).
- [ ] CI (matrice CPU) verte ; rapport de couverture téléchargé.

### Phase 2 — Signal et Optimisation
- [ ] `scir-signal` : `butter`, `cheby1`, `bessel` → SOS ; `sosfilt`, `filtfilt`, `resample_poly`.
- [ ] Parité des fixtures pour la conception de filtres et les cas limites de `filtfilt`.
- [ ] `scir-optimize` : Nelder–Mead, BFGS/L-BFGS avec recherche linéaire.
- [ ] Les fixtures de l'optimiseur (Rosenbrock, Himmelblau) passent dans les tolérances.

### Phase 3 — Linalg et backends (CPU)
- [x] Chemin `faer` implémenté (fonctionnalité câblée ; chemin temporaire `ndarray-linalg` pour la parité, intégration native `faer` en file d'attente).
- [x] Ensemble minimal Solve/SVD/QR avec fixtures.
- [x] Benchmarks pour FFT, filtres, solve.

### Phase 4 — Fondations GPU
- [x] Abstraction de tableau de périphérique `scir-gpu` + transferts (basé CPU, stub CUDA à fonctionnalité restreinte).
- [x] Ajout par éléments CUDA + FIR par lots avec parité vs CPU (noyaux PTX + FFI pilote ; les tests sont ignorés si CUDA non disponible).
- [ ] Job CI GPU auto-hébergé vert ; documentation pour la configuration de l'exécuteur.

### Phase 5 — FFT et GEMM GPU
- [ ] Chemin FFI cuFFT câblé pour FFT 1D ; parité vs fixtures CPU.
- [ ] GEMM via cuBLAS (CUDA) ou noyaux WGSL ; intégration linalg.
- [ ] Tests inter-backends (CPU↔CUDA↔WGPU) dans les tolérances.

### Phase 6 — Optimisation et durcissement
- [ ] SIMD, blocage conscient du cache, fusion de noyaux le cas échéant.
- [ ] Benchmarks Criterion stables ; notes de performance enregistrées par version.
- [ ] Site de documentation publié ; exemples et provenance des fixtures documentés.
- [ ] Versions v0.1…v0.3 taguées et crates publiées.

---

**TL;DR :** Le plan reflète maintenant l'intention du fil de discussion, corrige les réalités GPU/CI (exigence auto-hébergée), évite les crates spéculatives, garantit la parité basée sur les fixtures et ajoute des portes de phase claires pour que cela puisse ressembler à « tirer sur des poissons dans un baril » — avec des preuves en CI.


## Journal des progrès
- Échafaudage initial : README, AGENTS, sous-module SciPy, mises à jour de script.
- Ajout des dépendances numpy/scipy et amorce des crates scir-core/nd.

- Implémentation de la macro assert_close! et des assistants de conversion ndarray Vec.
- Création d'un script de génération de fixtures d'exemple et d'une sortie ignorée dans git.
- Documentation des étapes d'installation des dépendances Python dans le README.
- Ajout du support slice/complex à `assert_close!` et générateur de fixtures paramétré.
- Extension de `assert_close!` pour fonctionner avec les tableaux ndarray et documentation de l'initialisation du sous-module dans le README.
- Ajout de tests basés sur des tableaux dans `scir-nd` et passage des fixtures à `.npy` avec un format documenté.

- Création de la crate scir-fft avec FFT réel et tests de parité basés sur les fixtures.
- Ajout du guide CONTRIBUTING pour les contributeurs.
- Implémentation de la FFT inverse avec des fixtures pour plusieurs tailles et notes sur les directives de style dans CONTRIBUTING.
- Ajout de tests de parité pour FFT et IFFT sur plusieurs tailles de fixtures.
- Ajout de routines FFT réelles (rfft/irfft) avec des fixtures et des tests multi-tailles.
- Introduction de la crate scir-signal avec la conception Butterworth et la parité sosfilt.
- Ajout de la crate scir-optimize implémentant Nelder–Mead et BFGS validés sur les fixtures Rosenbrock et Himmelblau.
- Extension des fixtures de signal pour inclure les conceptions Chebyshev et Bessel avec les API Rust et l'échafaudage filtfilt.
- Ajout de filtres à phase nulle et de fixtures `resample_poly` avec des tests de parité Rust.
- Implémentation de l'optimiseur L-BFGS et des fixtures pour Rosenbrock et Himmelblau.
- Phase 2 terminée : routines de signal de base et optimiseur L-BFGS validés.
- Ajout de crochets de pré-commit et mise à jour des directives de contribution.
- Suppression du sous-module git SciPy ; SciPy installé par pip satisfait la génération de fixtures.
- Lancement de la Phase 3 : réintroduction du sous-module git SciPy à `/scipy` et mise à jour des références de documentation.
- Mise à jour du script de configuration de la CI pour initialiser automatiquement le sous-module SciPy.
- Documentation du chemin du sous-module comme `/scipy` et script de configuration indépendant du chemin.
- Ajout de la crate `scir-linalg` avec le chemin BLAS/LAPACK via `ndarray-linalg` et des flags de fonctionnalités pour le backend `faer`.
- Implémentation des API `solve`, `svd` et `qr` avec des tests basés sur les fixtures (fonctionnalité BLAS).
- Ajout de `scripts/gen_linalg_fixtures.py` pour générer des fixtures linalg (`lin_solve_*`, `svd_A.npy`, `qr_A.npy`).
- Préparation de la restriction des fonctionnalités pour le backend `faer` (placeholders) ; activation dans un suivi.
- Ajout de bancs d'essai Criterion pour FFT, signal et solve linalg ; ajout d'un test de propriété de base pour `solve` avec des matrices SPD.
- Câblage de la fonctionnalité `faer` à la fonction (actuellement via `ndarray-linalg`), à remplacer par des routines `faer` natives lors de la prochaine itération.

— Phase 3 terminée —

- Échafaudage de la crate `scir-gpu` avec `DeviceArray` (forme, dtype, périphérique), transferts CPU, opérations par éléments et base CPU FIR par lots avec des tests (stubs CUDA à fonctionnalité restreinte).
- Ajout d'un squelette de workflow CI GPU auto-hébergé (`.github/workflows/gpu.yml`) et d'un guide de configuration de l'exécuteur (`docs/gpu-runner.md`).
- Exécution de `cargo fmt --all` pour aligner le formatage.

- Implémentation d'un chemin CUDA minimal (fonctionnalité `cuda`) utilisant l'API Driver + PTX intégré pour l'addition de vecteurs f32, l'addition scalaire et le FIR par lots ; les tests sont ignorés si CUDA n'est pas disponible.
- Ajout de CodeBuild GPU `buildspec.gpu.yml` et de la documentation sur l'utilisation d'une image ECR compatible CUDA et du mode privilégié.
- Activation de l'étape de test CUDA dans `.github/workflows/gpu.yml` pour les exécuteurs GPU auto-hébergés.
- Ajout d'assistants de répartition automatique : l'addition/addition scalaire par éléments et le FIR peuvent être acheminés vers CUDA lorsque le périphérique est `Cuda` (repli sur CPU si indisponible).
- Ajout de `ci/docker/Dockerfile.cuda` pour construire une image CUDA+Rust pour CodeBuild ; fourni `ci/codebuild/project.example.json` et `docs/codebuild-gpu.md` avec des instructions de configuration.
- Ajout d'une crate parapluie `crates/scir` avec une fonctionnalité `gpu` agrégée et un exemple `fir_gpu.rs` montrant le FIR CPU vs CUDA.
- Mise à jour du README de niveau supérieur avec l'utilisation de la crate parapluie, les instructions de la fonctionnalité GPU, les références à l'exécuteur CI et à CodeBuild.
- Ajout de READMEs de crate : `crates/scir/README.md` et `crates/scir-gpu/README.md` pour un aperçu rapide de l'utilisation et des fonctionnalités.
- Ajout d'exemples : `crates/scir/examples/elementwise_gpu.rs` (opérations par éléments CPU vs CUDA) à côté de la démo FIR.
- CONTRIBUTING mis à jour avec une section concise sur les tests GPU et des liens vers la documentation GPU.
- Ajout de `crates/scir/examples/fir_bench.rs` pour un simple chronométrage CPU vs CUDA utilisant `Instant` (pas de dépendances supplémentaires). Alias de fonctionnalité agrégée `gpu-all` ajouté dans la crate parapluie.

- Correction des erreurs de compilation des doctests `scir-signal` en supprimant l'accès aux internes privés de `Sos` dans les exemples ; exemples mis à jour pour valider via `sosfilt`. Exécution de `cargo fmt` et vérification que les doctests passent.
```
