# QIFT halo stars

Code and results for *Quantum information field analysis of LAMOST spectra: when do topology and
entanglement help to find Galactic halo stars?* (Kotlarz et al., submitted to CAOSP).
Script comments and reports are in Polish.

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 01  | `01_spectra_astro.py` | `astro_reports/catalog_with_redshift.csv`, `spectra_deredshifted.npz` |
| 01b | `01b_gaia_kinematyka.py` | `gwiazdy_qift_kinematyka.csv` (Gaia DR3, UVW, v_tot, reference sets) |
| 02a | `02a_pobierz_C_N_aktywnosc.py [FITS_DIR]` | `astro_reports/kotwice_C_N_aktywnosc.csv` ([C/Fe], [N/Fe], activity) |
| 02b | `02b_latent_signatures.py` | `astro_reports/qiftdata.csv` (7 latents), VAE model |
| 03  | `03_hilbert_topology_<run>.py` | `hilbertspace_topology/<run>/` |
| 04  | `04_qift_eigenstates_<run>.py` | `qift_reports/<run>/` |
| 05  | `05_rysunki_artykulu.py`, `05b_jakosc_latentow.py`, `05c_splatanie_kontrola.py`, `05d_sprzezenie_Fe.py` | `artykul/` (figures 1–9, numbers) |

Run everything from the repository root. Step 02a needs the original LAMOST FITS files
(`spec-obsid-<obsid>.fits`) in `./widma` or in the directory given by `QIFT_SPECTRA_DIR` / the first argument.

## Configurations in the paper

| Paper | Reference | 7 qubits | 5 qubits | 2 qubits |
|-------|-----------|----------|----------|----------|
| U | whole sample | `p1a_1klasa_cala_7q` | `p3a_1klasa_cala_5q` | `p3a_1klasa_cala_2q` |
| D (mass gap) | disc | `p1b_1klasa_dysk_7q` | `p3b_1klasa_dysk_5q` | `p3b_1klasa_dysk_2q` |
| D (θ=0, mass gap, contrastive) | disc | `k1_kontrast_wszystkie_7q` | `k3_kontrast_chemia_5q` | `k2_kontrast_Fe_alpha_2q` |
| B | disc + halo | `p2_2klasy_7q` | `p4_2klasy_5q` | `p4_2klasy_2q` |

Qubits: 7 = log g, T_eff, Fe, α, C, N, activity; 5 = the five chemical latents; 2 = Fe, α.

## Requirements and data

`pip install -r requirements.txt`. Typical run times on a laptop CPU: 03+04 for 2 qubits < 1 min,
5 qubits ~5 min, 7 qubits 20–30 min (contrastive and mass-gap topologies re-learned in 15 folds).

Data: LAMOST DR (low-resolution spectra and stellar parameters), *Gaia* DR3 (I/355), DD-Payne
abundances (Xiang et al. 2019, J/ApJS/245/34), Ca II H&K activity (Zhang et al. 2022, Zenodo).
Code: GPL-3.0. Derived data and results: please cite the paper.
