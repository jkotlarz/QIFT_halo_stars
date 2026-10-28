# -*- coding: utf-8 -*-
"""
02_latent_signatures.py
=======================
KROK 2 potoku QIFT-spectra: przełożenie sygnatur spektralnych na zmienne
latentne (fizycznie ustrukturyzowany VAE) + raporty korelacji sygnatur
z parametrami gwiazd.

Co robi ten plik
----------------
1. Wczytuje widma deredshiftowane i katalog z pliku 01 (astro_reports/).
2. Buduje okna sygnatur widmowych zdefiniowane przez lambda_min / lambda_max
   (grupy: fe, alpha, hydrogen, act, ch, cn; próżnia) i raportuje ich pokrycie na siatce.
3. Liczy "jasność" sygnatur (średni strumień w oknie, głębokość linii,
   pseudo-szerokość równoważną EW) i ich korelacje z Teff, log g, [Fe/H],
   [α/M] (Pearson, Spearman), widmo korelacji r(λ) piksel po pikselu,
   wykresy fragmentów sygnatur kolorowane parametrami.
4. Trenuje (lub wczytuje) funkcyjny VAE:
      enkoder Conv1d -> w_atm (2D: atmosfera) + z_chem (5D: z_Fe, z_alpha, z_C, z_N, z_act)
      dekodery regionalne (Fe, α, C, N, aktywność) rekonstruujące tylko okna sygnatur
      dyskryminator adwersaryjny (z_Fe, z_alpha nie mogą nieść Teff/logg)
      kary fizyczne: dopasowanie latentów do celów + kary za PRZECIEK
      latentów do nieprawidłowych parametrów (kwadraty korelacji).
5. Raporty latentów: macierz korelacji latent × parametr, macierz przecieku,
   wykresy rozrzutu, corner plot, rekonstrukcje okien, MSE rekonstrukcji.
6. Zapisuje qiftdata.csv (katalog + latenty + jasności sygnatur + MSE)
   – wejście dla plików 03–05.

Wszystkie raporty: ASTRO_REPORTS_DIR (astro_reports/), prefiks "02_".
Uruchomienie: python 02_latent_signatures.py

Wersja 2 (2026-10-22): siedem latentów
--------------------------------------
  w_atm_1, w_atm_2 (atmosfera)  |  z_Fe, z_alpha, z_C, z_N, z_act (stochastyczne)
  - okna w PRÓŻNI (LAMOST); stare okna przeliczone z powietrza,
  - Ca II H usunięte z grupy α (zastąpione Ca I 4227 i 6163), rdzeń Hα przeniesiony
    do aktywności (w grupie wodoru zostały skrzydła),
  - nowe dekodery: C (pasmo G CH), N (pasma CN 3883 i 4215; wejście z_C i z_N),
    aktywność (rdzenie Ca II H, K, Hα, Ca II IRT; wejście z_act i atmosfera),
  - kotwice z_C, z_N, z_act z pliku astro_reports/kotwice_C_N_aktywnosc.csv
    (skrypt 02a_pobierz_C_N_aktywnosc.py); brak kotwicy = latent bez kary dopasowania,
  - bez kary dekorelującej między latentami chemicznymi (relacje mają zostać dla topologii QIFT).
  Jeśli plik kotwic nie istnieje, nowe latenty są uczone tylko z rekonstrukcji okien.
  Uwaga: widma wejściowe są standaryzowane (z-score), więc metryki jasności sygnatur są
  liczone jako RÓŻNICE względem kontinuum (w jednostkach odchylenia standardowego widma),
  a nie ilorazy.
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.init as nn_init
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# ======================= PARAMETRY URUCHOMIENIOWE ===========================
# ============================================================================

# --- Wejście (wyniki pliku 01) ---
ASTRO_REPORTS_DIR = Path("astro_reports")
DERED_NPZ_PATH = ASTRO_REPORTS_DIR / "spectra_deredshifted.npz"
CATALOG_CSV_PATH = ASTRO_REPORTS_DIR / "catalog_with_redshift.csv"

# --- Wyjście ---
MODEL_DIR = ASTRO_REPORTS_DIR / "model"
MODEL_PATH = MODEL_DIR / "physically_structured_vae.pt"
MODEL_META_JSON = MODEL_DIR / "physically_structured_vae_meta.json"
QIFTDATA_OUT_PATH = ASTRO_REPORTS_DIR / "qiftdata.csv"

# --- Trening czy wczytanie modelu ---
TRAIN_MODEL = True            # True: trenuj od zera; False: wczytaj MODEL_PATH

# --- Parametry fizyczne ---
PARAM_COLS = ["teff", "logg", "feh", "alpha_m"]
PARAM_LABELS = {"teff": "Teff [K]", "logg": "log g", "feh": "[Fe/H]", "alpha_m": "[α/M]"}

# --- Kotwice dla nowych latentów (wynik 02a_pobierz_C_N_aktywnosc.py; łączenie po obsid) ---
ANCHORS_CSV = ASTRO_REPORTS_DIR / "kotwice_C_N_aktywnosc.csv"
ANCHOR_COLS = {"kotwica_C": "c_fe_kotw", "kotwica_N": "n_fe_kotw", "kotwica_akt": "akt_kotw"}
EXTRA_PARAM_LABELS = {"c_fe_kotw": "[C/Fe]", "n_fe_kotw": "[N/Fe]", "akt_kotw": "log S − log S_bazowe"}
MIN_ANCHORED_STARS = 30       # mniej gwiazd z kotwicą -> latent uczony bez kotwicy (waga dopasowania 0)


def air_to_vac(lam_air: float) -> float:
    """Długość fali w powietrzu -> próżnia (wzór IAU / Morton 2000) [Å]."""
    s2 = (1e4 / lam_air) ** 2
    return lam_air * (1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2))


# --- Okna sygnatur widmowych [Å] w układzie spoczynkowym ---
# Widma LAMOST mają długości fal w PRÓŻNI. Okna z polem "frame": "air" są przeliczane
# na próżnię przy wczytaniu (w wersji 1 tego nie robiono – przesunięcie 1–2 Å).
# "cont": pasma pseudokontinuum do metryk jasności (domyślnie ±CONTINUUM_PAD_A wokół okna).
SIGNATURE_REGIONS = [
    # --- Fe (okna z wersji 1, w powietrzu) ---
    {"name": "Fe_4849_4855", "group": "fe", "lambda_min": 4849.0, "lambda_max": 4855.0, "frame": "air"},
    {"name": "Fe_5315_5321", "group": "fe", "lambda_min": 5315.0, "lambda_max": 5321.0, "frame": "air"},
    {"name": "Fe_5710_5750", "group": "fe", "lambda_min": 5710.0, "lambda_max": 5750.0, "frame": "air"},
    # --- α: Ca II H usunięte (rdzeń mierzy aktywność); w zamian Ca I 4227 i 6163 (próżnia) ---
    {"name": "CaI_4227",     "group": "alpha", "lambda_min": 4225.9, "lambda_max": 4229.9, "frame": "vac"},
    {"name": "TiII_4369",    "group": "alpha", "lambda_min": 4368.7, "lambda_max": 4374.0, "frame": "air"},
    {"name": "TiO_4533",     "group": "alpha", "lambda_min": 4533.0, "lambda_max": 4537.0, "frame": "air"},
    {"name": "MgI_5152",     "group": "alpha", "lambda_min": 5152.0, "lambda_max": 5158.0, "frame": "air"},
    {"name": "MgI_5178",     "group": "alpha", "lambda_min": 5178.0, "lambda_max": 5183.0, "frame": "air"},
    {"name": "CaI_6163",     "group": "alpha", "lambda_min": 6161.9, "lambda_max": 6165.9, "frame": "vac"},
    # --- wodór: tylko SKRZYDŁA Hα (wskaźnik Teff); rdzeń należy do aktywności ---
    {"name": "Halpha_wing_b", "group": "hydrogen", "lambda_min": 6550.0, "lambda_max": 6560.0, "frame": "vac"},
    {"name": "Halpha_wing_r", "group": "hydrogen", "lambda_min": 6569.0, "lambda_max": 6580.0, "frame": "vac"},
    # --- aktywność chromosferyczna: rdzenie (z profilem linii wokół) ---
    {"name": "CaII_K",       "group": "act", "lambda_min": 3929.8, "lambda_max": 3939.8, "frame": "vac",
     "cont": [(3892.17, 3912.17), (3992.20, 4012.20)]},
    {"name": "CaII_H",       "group": "act", "lambda_min": 3964.6, "lambda_max": 3974.6, "frame": "vac",
     "cont": [(3892.17, 3912.17), (3992.20, 4012.20)]},          # blend z Hε 3971,2
    {"name": "Halpha_core",  "group": "act", "lambda_min": 6561.1, "lambda_max": 6568.1, "frame": "vac",
     "cont": [(6545.0, 6555.0), (6575.0, 6585.0)]},
    {"name": "CaII_IRT_8500", "group": "act", "lambda_min": 8496.4, "lambda_max": 8504.4, "frame": "vac",
     "cont": [(8475.0, 8485.0), (8562.0, 8572.0)]},
    {"name": "CaII_IRT_8544", "group": "act", "lambda_min": 8540.4, "lambda_max": 8548.4, "frame": "vac",
     "cont": [(8475.0, 8485.0), (8562.0, 8572.0)]},
    {"name": "CaII_IRT_8665", "group": "act", "lambda_min": 8660.5, "lambda_max": 8668.5, "frame": "vac",
     "cont": [(8562.0, 8572.0), (8680.0, 8690.0)]},
    # --- węgiel: pasmo G (CH), jak indeks Lick G4300 ---
    {"name": "CH_Gband",     "group": "ch", "lambda_min": 4282.6, "lambda_max": 4317.6, "frame": "vac",
     "cont": [(4267.6, 4283.8), (4320.1, 4336.3)]},
    # --- azot przez CN (siła CN zależy od C i N) ---
    {"name": "CN_3883",      "group": "cn", "lambda_min": 3847.1, "lambda_max": 3884.1, "frame": "vac",
     "cont": [(3884.1, 3917.1)]},
    {"name": "CN_4215",      "group": "cn", "lambda_min": 4143.3, "lambda_max": 4178.3, "frame": "vac",
     "cont": [(4085.0, 4097.5), (4245.3, 4285.3)]},
]
for _r in SIGNATURE_REGIONS:                       # przeliczenie okien z powietrza na próżnię
    if _r.get("frame", "vac") == "air":
        _r["lambda_min_air"], _r["lambda_max_air"] = _r["lambda_min"], _r["lambda_max"]
        _r["lambda_min"] = round(air_to_vac(_r["lambda_min"]), 2)
        _r["lambda_max"] = round(air_to_vac(_r["lambda_max"]), 2)
        _r["frame"] = "vac (z air)"

# --- Latenty ---
ATM_LATENTS = ["w_atm_1", "w_atm_2"]                        # deterministyczne
CHEM_LATENTS = ["z_Fe", "z_alpha", "z_C", "z_N", "z_act"]   # stochastyczne (mu, logvar)
LATENT_NAMES = ATM_LATENTS + CHEM_LATENTS
ADV_LATENTS = ["z_Fe", "z_alpha"]   # dyskryminator Teff/logg działa na tych latentach (jak w v1)

# --- Dekodery: grupy okien, wejścia i latent, który dany dekoder uczy ---
# Gradient z rekonstrukcji płynie tylko do latentu "trains"; pozostałe wejścia są odłączone
# (detach), tak jak w v1: Fe uczy z_Fe, α uczy z_alpha przy znanym z_Fe itd.
DECODERS = {
    "fe":    {"groups": ["fe", "hydrogen"],    "inputs": ["z_Fe", "w_atm_1", "w_atm_2"], "trains": "z_Fe"},
    "alpha": {"groups": ["alpha", "hydrogen"], "inputs": ["z_Fe", "z_alpha", "w_atm_1", "w_atm_2"], "trains": "z_alpha"},
    "c":     {"groups": ["ch"],                "inputs": ["z_C", "z_Fe", "w_atm_1", "w_atm_2"], "trains": "z_C"},
    "n":     {"groups": ["cn"],                "inputs": ["z_C", "z_N", "z_Fe", "w_atm_1", "w_atm_2"], "trains": "z_N"},
    "act":   {"groups": ["act"],               "inputs": ["z_act", "z_Fe", "w_atm_1", "w_atm_2"], "trains": "z_act"},
}
DECODER_GROUPS = {k: v["groups"] for k, v in DECODERS.items()}
# Parametr używany do kolorowania wykresów fragmentów sygnatur danej grupy
GROUP_COLOR_PARAM = {"fe": "feh", "alpha": "alpha_m", "hydrogen": "teff",
                     "act": "akt_kotw", "ch": "c_fe_kotw", "cn": "n_fe_kotw"}
CONTINUUM_PAD_A = 6.0         # szerokość pasm kontinuum po obu stronach okna [Å] (gdy brak "cont")
FRAGMENT_PLOT_PAD_A = 8.0     # margines wykresów fragmentów [Å]
N_FRAGMENT_SPECTRA = 250      # ile widm rysować w wykresach fragmentów
N_PARAM_BINS = 5              # liczba przedziałów (kwantyli) do profili średnich

# --- Przypisanie latentów do parametrów (cele fizyczne; NaN w celu = brak kary dla tej gwiazdy) ---
LATENT_TARGETS = {"w_atm_1": "logg", "w_atm_2": "teff", "z_Fe": "feh", "z_alpha": "alpha_m",
                  "z_C": "c_fe_kotw", "z_N": "n_fe_kotw", "z_act": "akt_kotw"}

TARGET_PARAMS = PARAM_COLS + list(ANCHOR_COLS.values())     # wszystkie cele/parametry kar

# --- Kary dopasowania latent -> cel (MSE do standaryzowanego celu) ---
FIT_WEIGHTS = {"w_atm_1": 1.0, "w_atm_2": 1.0, "z_Fe": 1.0, "z_alpha": 1.0,
               "z_C": 1.0, "z_N": 1.0, "z_act": 1.0}

# --- Kary za przeciekanie latentów do nieprawidłowych parametrów ---
# Każda kara = redukcja (sum/mean) z corr²(latent, parametr) po liście par (pomija NaN).
# ZMIANA względem v1: NIE karzemy korelacji między latentami chemicznymi (chemical_cross_leak
# ma wagę 0) – relacje [α/Fe]–[Fe/H] i C–N mają zostać w danych, bo na nich ma pracować
# topologia QIFT. Nie karzemy też związku z_C, z_N z log g (mieszanie C/N u olbrzymów jest
# fizyczne); karzemy tylko związek z Teff, który wynika z czułości pasm molekularnych na temperaturę.
LEAKAGE_PENALTIES = {
    "chemical_cross_leak": {         # (wyłączone) z_Fe nie niesie [α/M], z_alpha nie niesie [Fe/H]
        "weight": 0.0, "reduction": "sum",
        "pairs": [("z_Fe", "alpha_m"), ("z_alpha", "feh")]},
    "chemistry_in_atmosphere": {     # w_atm nie niesie chemii
        "weight": 1.0, "reduction": "mean",
        "pairs": [("w_atm_1", "feh"), ("w_atm_1", "alpha_m"),
                  ("w_atm_2", "feh"), ("w_atm_2", "alpha_m")]},
    "atmosphere_in_chemistry": {     # z_Fe, z_alpha nie niosą atmosfery
        "weight": 1.0, "reduction": "mean",
        "pairs": [("z_Fe", "teff"), ("z_Fe", "logg"),
                  ("z_alpha", "teff"), ("z_alpha", "logg")]},
    "teff_in_CN": {                  # z_C, z_N nie niosą Teff (log g dozwolone)
        "weight": 0.5, "reduction": "mean",
        "pairs": [("z_C", "teff"), ("z_N", "teff")]},
    "atmosphere_in_activity": {      # z_act = nadwyżka ponad fotosferę, nie Teff ani log g
        "weight": 0.5, "reduction": "mean",
        "pairs": [("z_act", "teff"), ("z_act", "logg")]},
    "atmosphere_cross_leak": {       # (opcjonalnie) rozdział logg / Teff w w_atm
        "weight": 0.0, "reduction": "sum",
        "pairs": [("w_atm_1", "teff"), ("w_atm_2", "logg")]},
}

# --- Normalizacja wejścia enkodera ---
#   "per_pixel" – (F − średnia piksela) / std piksela (w teście na danych syntetycznych kara
#                 fizyczna spadła z ~4.5 do ~0.9 po 3 epokach),
#   "none"      – surowy strumień, jak w oryginalnym 01.preparedataframe2.py.
# Dekodery zawsze rekonstruują surowy strumień w oknach sygnatur.
INPUT_NORMALIZATION = "none"

# --- Hiperparametry treningu ---
SEED = 42
USE_CUDA_IF_AVAILABLE = True
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
BETA_KL = 5e-3
LAMBDA_ADV_SCHEDULE = [(0.30, 0.0), (0.50, 0.01), (0.75, 0.011)]   # (ułamek epok, lambda)
DEFAULT_LAMBDA_ADV = 0.012
TEFF_BINS = 12
LOGG_BINS = 10

# --- Wykresy ---
N_RECON_EXAMPLES = 3
DPI = 180


# ============================================================================
# ========================= FUNKCJE POMOCNICZE ================================
# ============================================================================

def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_report(path: Path, title: str, sections: list[tuple[str, list[str]]]) -> Path:
    lines = ["=" * 80, title, f"Wygenerowano: {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 80, ""]
    for heading, body in sections:
        lines += [heading, "-" * len(heading)] + list(body) + [""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ raport: {path}")
    return Path(path)


def save_fig(fig, path: Path, dpi: int = DPI) -> Path:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ wykres: {path}")
    return Path(path)


def df_lines(df: pd.DataFrame, decimals: int = 4, index: bool = True) -> list[str]:
    fmt = lambda v: f"{v:.{decimals + 1}g}"
    return ["  " + l for l in df.to_string(index=index, float_format=fmt).splitlines()]


def setup_environment(seed: int = SEED, use_cuda: bool = USE_CUDA_IF_AVAILABLE) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


DEVICE = setup_environment()


def standardize(arr):
    m, s = np.nanmean(arr), np.nanstd(arr)
    if not np.isfinite(s) or s == 0:
        s = 1.0
    return ((arr - m) / s).astype(np.float32), float(m), float(s)


def make_bins(values, n_bins):
    """Przedziały kwantylowe (etykiety klas dla dyskryminatora)."""
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    edges = np.maximum.accumulate(edges)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return np.digitize(values, edges[1:-1]).astype(np.int64), edges.astype(np.float32)


def corr_with_p(x, y, method="pearson"):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan, np.nan
    r = stats.pearsonr(x[ok], y[ok]) if method == "pearson" else stats.spearmanr(x[ok], y[ok])
    return float(r[0]), float(r[1])


# ============================================================================
# ======================== 1. WCZYTANIE DANYCH ================================
# ============================================================================

def load_inputs(npz_path: Path = DERED_NPZ_PATH, csv_path: Path = CATALOG_CSV_PATH):
    """Wczytuje widma deredshiftowane (X, wave_grid) i katalog z pliku 01."""
    print("=" * 80)
    print("KROK 2.1: WCZYTANIE WIDM DEREDSHIFTOWANYCH I KATALOGU")
    print("=" * 80)
    with np.load(npz_path, allow_pickle=True) as d:
        X = d["X"].astype(np.float32)
        wave = d["wave_grid"].astype(np.float32)
    df = pd.read_csv(csv_path)
    if len(df) != X.shape[0]:
        raise ValueError(f"Katalog ({len(df)}) i widma ({X.shape[0]}) mają różną liczbę wierszy")
    if not np.all(np.isfinite(X)):
        n_bad = int((~np.isfinite(X)).sum())
        print(f"  ⚠ {n_bad} wartości NaN/inf w X – zastępuję medianą piksela")
        med = np.nanmedian(X, axis=0)
        X = np.where(np.isfinite(X), X, med[None, :]).astype(np.float32)
    print(f"✓ X={X.shape}, λ={wave.min():.1f}–{wave.max():.1f} Å, katalog {df.shape}")
    df = attach_anchors(df)
    return X, wave, df


def attach_anchors(df: pd.DataFrame, path: Path = ANCHORS_CSV) -> pd.DataFrame:
    """Dołącza kotwice C, N i aktywności (po obsid). Brak pliku -> kolumny NaN."""
    df = df.copy()
    for new in ANCHOR_COLS.values():
        df[new] = np.nan
    if not Path(path).exists():
        print(f"  ⚠ brak {path} – z_C, z_N, z_act będą uczone bez kotwic (uruchom 02a_pobierz_C_N_aktywnosc.py)")
        return df
    a = pd.read_csv(path)
    cols = ["obsid"] + [c for c in ANCHOR_COLS if c in a.columns]
    a = a[cols].drop_duplicates("obsid").rename(columns=ANCHOR_COLS)
    df = df.drop(columns=list(ANCHOR_COLS.values())).merge(a, on="obsid", how="left")
    for new in ANCHOR_COLS.values():
        if new not in df:
            df[new] = np.nan
    print("  ✓ kotwice: " + ", ".join(f"{c}: {df[c].notna().sum()}" for c in ANCHOR_COLS.values()))
    return df


def report_params(df: pd.DataFrame) -> list[str]:
    """Parametry do raportów: 4 podstawowe + kotwice, jeśli jest ich dość."""
    return PARAM_COLS + [c for c in ANCHOR_COLS.values() if c in df and df[c].notna().sum() >= MIN_ANCHORED_STARS]


def plabel(p: str) -> str:
    return PARAM_LABELS.get(p, EXTRA_PARAM_LABELS.get(p, p))


# ============================================================================
# ==================== 2. OKNA SYGNATUR I POKRYCIE ============================
# ============================================================================

def region_mask(wave, region) -> np.ndarray:
    return (wave >= region["lambda_min"]) & (wave <= region["lambda_max"])


def build_signature_index(wave, regions=SIGNATURE_REGIONS, decoder_groups=DECODER_GROUPS,
                          out_dir: Path = ASTRO_REPORTS_DIR):
    """
    Dla każdego okna liczy piksele siatki w [lambda_min, lambda_max]
    i buduje zbiory indeksów pikseli dla dekoderów (suma okien z grup).
    Raport: 02_signature_regions_coverage.txt/.csv.
    Zwraca: dict dekoder -> indeksy pikseli, tabela pokrycia.
    """
    print("=" * 80)
    print("KROK 2.2: OKNA SYGNATUR (lambda_min – lambda_max)")
    print("=" * 80)
    rows = []
    for r in regions:
        m = region_mask(wave, r)
        rows.append(dict(name=r["name"], group=r["group"], frame=r.get("frame", "vac"), lambda_min=r["lambda_min"],
                         lambda_max=r["lambda_max"], width_A=r["lambda_max"] - r["lambda_min"],
                         n_pixels=int(m.sum()),
                         grid_first=float(wave[m][0]) if m.any() else np.nan,
                         grid_last=float(wave[m][-1]) if m.any() else np.nan))
    cov = pd.DataFrame(rows)
    dec_idx = {}
    for dec, groups in decoder_groups.items():
        mask = np.zeros(len(wave), bool)
        for r in regions:
            if r["group"] in groups:
                mask |= region_mask(wave, r)
        dec_idx[dec] = np.where(mask)[0]
        if len(dec_idx[dec]) == 0:
            raise ValueError(f"Dekoder '{dec}' nie ma żadnych pikseli – sprawdź SIGNATURE_REGIONS")
    warn = [f"  ⚠ okno {n} ma < 3 piksele – metryki jasności mogą być niestabilne"
            for n, p in zip(cov["name"], cov["n_pixels"]) if p < 3]
    cov.to_csv(out_dir / "02_signature_regions_coverage.csv", index=False)
    write_report(out_dir / "02_signature_regions_coverage.txt", "POKRYCIE OKIEN SYGNATUR NA SIATCE", [
        ("Okna", df_lines(cov, 3, index=False)),
        ("Dekodery", [f"  {d}: grupy {DECODER_GROUPS[d]} -> {len(i)} pikseli" for d, i in dec_idx.items()]),
        ("Ostrzeżenia", warn or ["  brak"]),
    ])
    return dec_idx, cov


# ============================================================================
# ============== 3. JASNOŚĆ SYGNATUR I KORELACJE Z PARAMETRAMI ================
# ============================================================================

def compute_signature_brightness(X, wave, regions=SIGNATURE_REGIONS, pad=CONTINUUM_PAD_A) -> pd.DataFrame:
    """
    Dla każdego okna i każdej gwiazdy liczy:
      sig_<okno>_flux  – średni strumień w oknie ("jasność sygnatury"),
      sig_<okno>_depth – głębokość linii: F_cont − min(F),
      sig_<okno>_ew    – pseudo-szerokość równoważna: ∫(F_cont − F) dλ [Å × jedn. strumienia],
    gdzie F_cont = kontinuum liniowe przez mediany pasm kontinuum: z pola "cont" okna, a gdy
    go nie ma – pasm o szerokości `pad` po obu stronach okna.
    ZMIANA v2: różnice zamiast ilorazów, bo widma wejściowe są standaryzowane (z-score),
    a iloraz F/F_cont dla strumienia ze średnią odjętą nie ma sensu fizycznego.
    """
    out = {}
    dl = np.gradient(wave.astype(np.float64))
    for r in regions:
        m = region_mask(wave, r)
        if not m.any():
            continue
        if r.get("cont"):
            bands = [(wave >= a) & (wave <= b) for a, b in r["cont"]]
        else:
            bands = [(wave >= r["lambda_min"] - pad) & (wave < r["lambda_min"]),
                     (wave > r["lambda_max"]) & (wave <= r["lambda_max"] + pad)]
        bands = [b for b in bands if b.any()]
        f_win = X[:, m].astype(np.float64)
        if not bands:
            cont = np.max(f_win, axis=1)[:, None] * np.ones(m.sum())[None, :]
        elif len(bands) == 1:
            cont = np.median(X[:, bands[0]], axis=1)[:, None] * np.ones(m.sum())[None, :]
        else:
            (b1, b2) = bands[0], bands[-1]
            c1, c2 = np.median(X[:, b1], axis=1), np.median(X[:, b2], axis=1)
            x1, x2 = wave[b1].mean(), wave[b2].mean()
            t = (wave[m] - x1) / max(x2 - x1, 1e-6)
            cont = c1[:, None] + (c2 - c1)[:, None] * t[None, :]
        diff = cont - f_win
        n = r["name"]
        out[f"sig_{n}_flux"] = f_win.mean(axis=1)
        out[f"sig_{n}_depth"] = diff.max(axis=1)
        out[f"sig_{n}_ew"] = np.sum(diff * dl[m][None, :], axis=1)
    return pd.DataFrame(out)


def report_signature_correlations(sig: pd.DataFrame, df: pd.DataFrame,
                                  out_dir: Path = ASTRO_REPORTS_DIR) -> pd.DataFrame:
    """
    Korelacje (Pearson i Spearman, z p-value) każdej metryki jasności
    sygnatur z Teff, log g, [Fe/H], [α/M]. Raport tekstowy z rankingiem,
    CSV, heatmapy dla flux / depth / EW, siatka wykresów rozrzutu EW vs parametr.
    """
    print("=" * 80)
    print("KROK 2.3: KORELACJE JASNOŚCI SYGNATUR Z PARAMETRAMI GWIAZD")
    print("=" * 80)
    params = report_params(df)
    rows = []
    for col in sig.columns:
        for p in params:
            rp, pp = corr_with_p(sig[col].to_numpy(), df[p].to_numpy(float), "pearson")
            rs, ps = corr_with_p(sig[col].to_numpy(), df[p].to_numpy(float), "spearman")
            _, win, metric = col.split("_", 1)[0], col[4:].rsplit("_", 1)[0], col.rsplit("_", 1)[1]
            rows.append(dict(signature=win, metric=metric, param=p, pearson_r=rp, pearson_p=pp,
                             spearman_r=rs, spearman_p=ps))
    corr = pd.DataFrame(rows)
    corr.to_csv(out_dir / "02_signature_brightness_correlations.csv", index=False)

    sections = []
    for metric in ["flux", "depth", "ew"]:
        piv = corr[corr.metric == metric].pivot(index="signature", columns="param", values="pearson_r")
        piv = piv.reindex(columns=params)
        sections.append((f"Pearson r: metryka '{metric}' (okno × parametr)", df_lines(piv, 3)))
    # ranking najsilniejszych
    top = corr.assign(abs_r=corr.pearson_r.abs()).sort_values("abs_r", ascending=False).head(25)
    sections.append(("Ranking 25 najsilniejszych związków (|Pearson r|)",
                     df_lines(top[["signature", "metric", "param", "pearson_r", "spearman_r", "pearson_p"]],
                              4, index=False)))
    # parametr dominujący dla każdego okna (EW)
    dom = []
    for win, g in corr[corr.metric == "ew"].groupby("signature"):
        g = g.assign(a=g.pearson_r.abs()).sort_values("a", ascending=False)
        dom.append(f"  {win:<16s} najsilniej z: {g.iloc[0]['param']:<8s} (r={g.iloc[0]['pearson_r']:+.3f}),"
                   f" drugi: {g.iloc[1]['param']:<8s} (r={g.iloc[1]['pearson_r']:+.3f})")
    sections.append(("Parametr dominujący w EW każdego okna", dom))
    sections.append(("Interpretacja", [
        "  flux  – średni strumień w oknie (im mniejszy, tym silniejsza absorpcja),",
        "  depth – F_cont − min(F): głębokość centralna linii,",
        "  ew    – ∫(F_cont − F) dλ: całkowita absorpcja względem lokalnego kontinuum.",
        "  Okna Fe powinny korelować głównie z [Fe/H], okna alpha z [α/M] i [Fe/H],",
        "  skrzydła Hα głównie z Teff, rdzenie Ca II / Hα z aktywnością, pasmo G z [C/Fe],",
        "  pasma CN z [N/Fe] i [C/Fe]; metryki są różnicami względem kontinuum (widma standaryzowane).",
    ]))
    write_report(out_dir / "02_signature_brightness_correlations.txt",
                 "KORELACJE JASNOŚCI SYGNATUR Z PARAMETRAMI GWIAZD", sections)

    # heatmapy
    fig, axes = plt.subplots(1, 3, figsize=(21, 0.55 * sig.shape[1] / 3 + 3))
    fig.subplots_adjust(wspace=0.55)
    for ax, metric in zip(axes, ["flux", "depth", "ew"]):
        piv = corr[corr.metric == metric].pivot(index="signature", columns="param",
                                                values="pearson_r").reindex(columns=params)
        im = ax.imshow(piv.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(params)), params, rotation=30)
        ax.set_yticks(range(len(piv)), piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"Pearson r: {metric}")
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.suptitle("Korelacje jasności sygnatur z parametrami gwiazd", fontweight="bold")
    save_fig(fig, out_dir / "02_signature_brightness_corr_heatmaps.png")

    # siatka rozrzutu EW vs parametry
    ew_cols = [c for c in sig.columns if c.endswith("_ew")]
    fig, axes = plt.subplots(len(ew_cols), len(params), figsize=(4 * len(params), 2.6 * len(ew_cols)), squeeze=False)
    for i, c in enumerate(ew_cols):
        for j, p in enumerate(params):
            ax = axes[i, j]
            ax.scatter(df[p], sig[c], s=3, alpha=0.35, c=df["teff"] if p != "teff" else df["feh"],
                       cmap="viridis")
            r, _ = corr_with_p(sig[c].to_numpy(), df[p].to_numpy(float))
            ax.set_title(f"{c[4:-3]} vs {p}: r={r:+.2f}", fontsize=8)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel("EW [Å]", fontsize=8)
    fig.suptitle("Pseudo-EW sygnatur vs parametry (kolor: Teff, dla kolumny Teff: [Fe/H])", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir / "02_signature_ew_scatter_grid.png")
    return corr


def report_pixel_correlation_spectrum(X, wave, df, regions=SIGNATURE_REGIONS,
                                      out_dir: Path = ASTRO_REPORTS_DIR) -> pd.DataFrame:
    """
    Widmo korelacji r(λ): korelacja Pearsona strumienia w każdym pikselu
    z każdym parametrem. Okna sygnatur zacieniowane – pozwala ocenić, czy
    okna leżą w miejscach o najsilniejszej informacji o danym parametrze.
    Zapisuje też 20 najsilniej skorelowanych pikseli dla każdego parametru.
    """
    print("=" * 80)
    print("KROK 2.4: WIDMO KORELACJI r(λ)")
    print("=" * 80)
    Xc = X - X.mean(axis=0, keepdims=True)
    Xs = Xc / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))[None, :]
    out = pd.DataFrame({"wavelength": wave})
    params = report_params(df)
    for p in params:
        y = df[p].to_numpy(float); ok = np.isfinite(y)
        Xo = X[ok] - X[ok].mean(axis=0, keepdims=True)
        Xo = Xo / np.where(Xo.std(axis=0) == 0, 1, Xo.std(axis=0))[None, :]
        yo = (y[ok] - y[ok].mean()) / y[ok].std()
        out[f"r_{p}"] = (Xo * yo[:, None]).mean(axis=0)
    out.to_csv(out_dir / "02_pixel_correlation_spectrum.csv", index=False)

    colors = {"fe": "orange", "alpha": "green", "hydrogen": "red", "act": "purple", "ch": "brown", "cn": "teal"}
    fig, axes = plt.subplots(len(params), 1, figsize=(16, 3 * len(params)), sharex=True)
    for ax, p in zip(axes, params):
        ax.plot(wave, out[f"r_{p}"], lw=0.6, color="k")
        for r in regions:
            ax.axvspan(r["lambda_min"], r["lambda_max"], color=colors.get(r["group"], "gray"), alpha=0.35)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_ylabel(f"r(F, {p})"); ax.set_ylim(-1, 1); ax.grid(alpha=0.2)
    axes[0].set_title("Korelacja strumienia z parametrami piksel po pikselu "
                      "(pomarańcz: Fe, zieleń: α, czerwień: H, fiolet: aktywność, brąz: CH, morski: CN)")
    axes[-1].set_xlabel("λ [Å]")
    save_fig(fig, out_dir / "02_pixel_correlation_spectrum.png")

    sec = []
    for p in params:
        top = out.reindex(out[f"r_{p}"].abs().sort_values(ascending=False).index).head(20)
        in_win = [any(r["lambda_min"] <= w <= r["lambda_max"] for r in regions) for w in top.wavelength]
        sec.append((f"20 pikseli najsilniej skorelowanych z {p}",
                    [f"  λ={w:8.2f} Å  r={v:+.3f}  {'[w oknie sygnatury]' if f else ''}"
                     for w, v, f in zip(top.wavelength, top[f"r_{p}"], in_win)]))
    sec.append(("Średnie |r| w oknach vs poza oknami", [
        f"  {p:<8s}: okna={out.loc[np.any([region_mask(wave, r) for r in regions], axis=0), f'r_{p}'].abs().mean():.3f}"
        f"  poza={out.loc[~np.any([region_mask(wave, r) for r in regions], axis=0), f'r_{p}'].abs().mean():.3f}"
        for p in params]))
    write_report(out_dir / "02_pixel_correlation_spectrum.txt", "WIDMO KORELACJI r(λ)", sec)
    return out


def plot_signature_fragments(X, wave, df, regions=SIGNATURE_REGIONS, out_dir: Path = ASTRO_REPORTS_DIR):
    """
    Dla każdego okna: (lewy panel) fragmenty widm wielu gwiazd kolorowane
    parametrem grupy (Fe -> [Fe/H], α -> [α/M], H -> Teff), (prawy panel)
    profile średnie w kwantylach tego parametru. Dodatkowo wykres zbiorczy
    profili medianowych wszystkich okien.
    """
    print("=" * 80)
    print("KROK 2.5: WYKRESY FRAGMENTÓW SYGNATUR")
    print("=" * 80)
    frag_dir = ensure_dir(out_dir / "signature_fragments")
    rng = np.random.default_rng(SEED)
    sel = rng.choice(X.shape[0], size=min(N_FRAGMENT_SPECTRA, X.shape[0]), replace=False)
    for r in regions:
        p = GROUP_COLOR_PARAM.get(r["group"], "teff")
        if p not in df or df[p].notna().sum() < MIN_ANCHORED_STARS:
            p = "teff"
        m = (wave >= r["lambda_min"] - FRAGMENT_PLOT_PAD_A) & (wave <= r["lambda_max"] + FRAGMENT_PLOT_PAD_A)
        if m.sum() < 2:
            continue
        vals = df[p].to_numpy(float)
        norm = plt.Normalize(np.nanpercentile(vals, 2), np.nanpercentile(vals, 98))
        cmap = plt.get_cmap("plasma")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        for i in sel[np.argsort(vals[sel])]:
            if not np.isfinite(vals[i]):
                continue
            axes[0].plot(wave[m], X[i, m], color=cmap(norm(vals[i])), lw=0.5, alpha=0.5)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=axes[0], label=plabel(p))
        bins = pd.qcut(vals, N_PARAM_BINS, duplicates="drop")
        for k, (b, idx) in enumerate(pd.Series(np.arange(len(vals))).groupby(bins, observed=True)):
            prof = X[idx.values][:, m].mean(axis=0)
            axes[1].plot(wave[m], prof, lw=1.8, color=cmap(k / max(N_PARAM_BINS - 1, 1)),
                         label=f"{p} ∈ ({b.left:.2f}, {b.right:.2f}]  n={len(idx)}")
        for ax in axes:
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax.axvspan(r["lambda_min"], r["lambda_max"], color="gray", alpha=0.15)
            ax.set_xlabel("λ [Å]"); ax.grid(alpha=0.25)
        axes[0].set_ylabel("strumień")
        axes[0].set_title(f"{r['name']} [{r['lambda_min']}–{r['lambda_max']} Å, próżnia]: {len(sel)} widm")
        axes[1].set_title(f"Profile średnie w kwantylach {p}")
        axes[1].legend(fontsize=7)
        save_fig(fig, frag_dir / f"02_fragment_{r['name']}.png")

    # wykres zbiorczy
    n = len(regions)
    ncol = 3
    fig, axes = plt.subplots(math.ceil(n / ncol), ncol, figsize=(15, 3.6 * math.ceil(n / ncol)), squeeze=False)
    for ax, r in zip(axes.flat, regions):
        m = (wave >= r["lambda_min"] - FRAGMENT_PLOT_PAD_A) & (wave <= r["lambda_max"] + FRAGMENT_PLOT_PAD_A)
        p16, p50, p84 = np.percentile(X[:, m], [16, 50, 84], axis=0)
        ax.fill_between(wave[m], p16, p84, alpha=0.3)
        ax.plot(wave[m], p50, color="navy")
        ax.axvspan(r["lambda_min"], r["lambda_max"], color="orange", alpha=0.2)
        ax.set_title(f"{r['name']} ({r['group']})", fontsize=9); ax.grid(alpha=0.25)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle("Profile medianowe okien sygnatur (pasmo 16–84%)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir / "02_signature_windows_overview.png")


# ============================================================================
# ===================== 4. MODEL: FUNKCYJNY VAE ===============================
# ============================================================================

def init_linear_params(in_f, out_f, device=DEVICE):
    w = torch.empty(out_f, in_f, device=device)
    nn_init.kaiming_uniform_(w, a=math.sqrt(5))
    bound = 1 / math.sqrt(in_f) if in_f > 0 else 0
    b = torch.empty(out_f, device=device).uniform_(-bound, bound)
    return {"weight": w.requires_grad_(True), "bias": b.requires_grad_(True)}


def init_conv1d_params(in_c, out_c, k, device=DEVICE):
    w = torch.empty(out_c, in_c, k, device=device)
    nn_init.kaiming_uniform_(w, a=math.sqrt(5))
    bound = 1 / math.sqrt(in_c * k)
    b = torch.empty(out_c, device=device).uniform_(-bound, bound)
    return {"weight": w.requires_grad_(True), "bias": b.requires_grad_(True)}


def init_prelu_param(device=DEVICE):
    return torch.full((1,), 0.25, device=device, requires_grad=True)


def init_encoder_params(input_len, device=DEVICE):
    """Enkoder: 3×(Conv1d k=7 + PReLU + MaxPool 2) -> FC 128 -> FC 64 -> głowy w (2), mu, logvar (len(CHEM_LATENTS))."""
    enc_out_len = ((input_len // 2) // 2) // 2
    return {
        "input_len": input_len, "enc_out_len": enc_out_len,
        "conv1": init_conv1d_params(1, 8, 7, device), "prelu1": init_prelu_param(device),
        "conv2": init_conv1d_params(8, 16, 7, device), "prelu2": init_prelu_param(device),
        "conv3": init_conv1d_params(16, 32, 7, device), "prelu3": init_prelu_param(device),
        "fc1": init_linear_params(32 * enc_out_len, 128, device), "prelu4": init_prelu_param(device),
        "fc2": init_linear_params(128, 64, device), "prelu5": init_prelu_param(device),
        "fc_w": init_linear_params(64, len(ATM_LATENTS), device),
        "fc_mu": init_linear_params(64, len(CHEM_LATENTS), device),
        "fc_logvar": init_linear_params(64, len(CHEM_LATENTS), device),
    }


def encoder_forward(x, p):
    """Zwraca w_atm (2), z (próbka), mu, logvar (po len(CHEM_LATENTS))."""
    h = x
    for k in (1, 2, 3):
        h = F.conv1d(h, p[f"conv{k}"]["weight"], p[f"conv{k}"]["bias"], padding=3)
        h = F.max_pool1d(F.prelu(h, p[f"prelu{k}"]), 2)
    h = h.reshape(h.size(0), -1)
    h = F.prelu(F.linear(h, p["fc1"]["weight"], p["fc1"]["bias"]), p["prelu4"])
    h = F.prelu(F.linear(h, p["fc2"]["weight"], p["fc2"]["bias"]), p["prelu5"])
    w = F.linear(h, p["fc_w"]["weight"], p["fc_w"]["bias"])
    mu = F.linear(h, p["fc_mu"]["weight"], p["fc_mu"]["bias"])
    logvar = F.linear(h, p["fc_logvar"]["weight"], p["fc_logvar"]["bias"]).clamp(-10, 10)
    z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    return w, z, mu, logvar


def init_region_decoder_params(in_dim, out_len, device=DEVICE):
    return {"fc1": init_linear_params(in_dim, 32, device), "prelu1": init_prelu_param(device),
            "fc2": init_linear_params(32, 64, device), "prelu2": init_prelu_param(device),
            "fc3": init_linear_params(64, 128, device), "prelu3": init_prelu_param(device),
            "fc4": init_linear_params(128, out_len, device)}


def region_decoder_forward(x, p):
    h = x
    for k in (1, 2, 3):
        h = F.prelu(F.linear(h, p[f"fc{k}"]["weight"], p[f"fc{k}"]["bias"]), p[f"prelu{k}"])
    return F.linear(h, p["fc4"]["weight"], p["fc4"]["bias"])


def init_discriminator_params(device=DEVICE):
    return {"shared1": init_linear_params(len(ADV_LATENTS), 32, device), "shared2": init_linear_params(32, 32, device),
            "teff_head": init_linear_params(32, TEFF_BINS, device),
            "logg_head": init_linear_params(32, LOGG_BINS, device)}


def discriminator_forward(z, p):
    h = F.relu(F.linear(z, p["shared1"]["weight"], p["shared1"]["bias"]))
    h = F.relu(F.linear(h, p["shared2"]["weight"], p["shared2"]["bias"]))
    return (F.linear(h, p["teff_head"]["weight"], p["teff_head"]["bias"]),
            F.linear(h, p["logg_head"]["weight"], p["logg_head"]["bias"]))


def init_model_params(input_len, dec_lens: dict, device=DEVICE):
    """Enkoder + dekoder dla każdego wpisu DECODERS (wejścia wg DECODERS[k]['inputs'])."""
    P = {"encoder": init_encoder_params(input_len, device),
         "linear_w": init_linear_params(2, 2, device),
         "discriminator": init_discriminator_params(device)}
    for k, cfg in DECODERS.items():
        P[f"{k}_decoder"] = init_region_decoder_params(len(cfg["inputs"]), dec_lens[k], device)
    return P


def collect_param_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return [obj] if obj.requires_grad else []
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in collect_param_tensors(v)]
    return []


def count_params(obj) -> int:
    return int(sum(t.numel() for t in collect_param_tensors(obj)))


# ============================================================================
# ==================== 5. KARY FIZYCZNE I PRZECIEK ============================
# ============================================================================

def corr_squared(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Kwadrat współczynnika korelacji Pearsona (różniczkowalny); pomija pary z NaN w b."""
    a = a.reshape(-1); b = b.reshape(-1)
    ok = torch.isfinite(b)
    if int(ok.sum()) < 3:
        return a.sum() * 0.0
    a = a[ok] - a[ok].mean()
    b = b[ok] - b[ok].mean()
    r = (a * b).mean() / (torch.sqrt(a.pow(2).mean().clamp_min(eps)) *
                          torch.sqrt(b.pow(2).mean().clamp_min(eps)))
    return r.pow(2)


def latent_dict(w_atm, mu) -> dict:
    d = {n: w_atm[:, i] for i, n in enumerate(ATM_LATENTS)}
    d.update({n: mu[:, i] for i, n in enumerate(CHEM_LATENTS)})
    return d


def chem_index(name: str) -> int:
    return CHEM_LATENTS.index(name)


def physical_loss(w_atm, mu, targets: dict, fit_weights: dict = None):
    """
    L_phys = Σ_latent FIT_WEIGHTS·MSE(latent, cel)  +  Σ_kara weight·redukcja(corr²(latent, parametr)).
    targets: dict parametr -> tensor standaryzowany [B] (NaN = brak kotwicy, pomijany).
    Zwraca (total, dict składowych).
    """
    fit_weights = FIT_WEIGHTS if fit_weights is None else fit_weights
    lat = latent_dict(w_atm, mu)
    terms, total = {}, 0.0
    for name, tgt in LATENT_TARGETS.items():
        t = targets[tgt]; ok = torch.isfinite(t)
        v = ((lat[name][ok] - t[ok]) ** 2).mean() if int(ok.sum()) > 0 else lat[name].sum() * 0.0
        terms[f"fit_{name}"] = v
        total = total + fit_weights.get(name, 0.0) * v
    for pname, cfg in LEAKAGE_PENALTIES.items():
        vals = torch.stack([corr_squared(lat[l], targets[p]) for l, p in cfg["pairs"]])
        v = vals.sum() if cfg["reduction"] == "sum" else vals.mean()
        terms[f"leak_{pname}"] = v
        total = total + cfg["weight"] * v
    terms["physical_total"] = total
    return total, terms


def kl_diag(mu, logvar):
    return 0.5 * torch.mean(torch.sum(mu.pow(2) + torch.exp(logvar) - logvar - 1.0, dim=1))


def lambda_adv_at(epoch, epochs):
    frac = (epoch + 1) / max(epochs, 1)
    for thr, val in LAMBDA_ADV_SCHEDULE:
        if frac <= thr:
            return val
    return DEFAULT_LAMBDA_ADV


# ============================================================================
# ============================ 6. TRENING =====================================
# ============================================================================

def input_norm_stats(X, mode=INPUT_NORMALIZATION):
    """Średnia i odchylenie piksela do normalizacji wejścia enkodera."""
    if mode == "per_pixel":
        return X.mean(0).astype(np.float32), (X.std(0) + 1e-6).astype(np.float32)
    if mode == "none":
        return np.zeros(X.shape[1], np.float32), np.ones(X.shape[1], np.float32)
    raise ValueError(f"Nieznane INPUT_NORMALIZATION={mode}")


def normalize_input(X, P) -> np.ndarray:
    nm = P.get("input_norm")
    if nm is None:
        return X
    return ((X - nm["mean"].cpu().numpy()) / nm["std"].cpu().numpy()).astype(np.float32)


def prepare_training_data(X, df, dec_idx, Xin=None):
    """Standaryzacja celów (z NaN), przedziały dla dyskryminatora, tensory."""
    Xin = X if Xin is None else Xin
    scal, targ = {}, {}
    for p in TARGET_PARAMS:
        targ[p], m, s = standardize(df[p].to_numpy(np.float32))
        scal[p] = {"mean": m, "std": s, "n_valid": int(np.isfinite(df[p].to_numpy(float)).sum())}
    teff_bins, teff_edges = make_bins(targ["teff"], TEFF_BINS)
    logg_bins, logg_edges = make_bins(targ["logg"], LOGG_BINS)
    ds = TensorDataset(
        torch.tensor(Xin, dtype=torch.float32).unsqueeze(1),
        torch.tensor(np.column_stack([targ[p] for p in TARGET_PARAMS]), dtype=torch.float32),
        torch.tensor(teff_bins), torch.tensor(logg_bins),
        *[torch.tensor(X[:, dec_idx[k]], dtype=torch.float32) for k in DECODERS])
    meta = {"scaling": scal, "teff_bin_edges": teff_edges.tolist(), "logg_bin_edges": logg_edges.tolist()}
    return ds, targ, meta


def effective_fit_weights(df) -> tuple[dict, list[str]]:
    """Latent, którego cel ma < MIN_ANCHORED_STARS wartości, uczony bez kary dopasowania."""
    w, notes = dict(FIT_WEIGHTS), []
    for l, t in LATENT_TARGETS.items():
        n = int(np.isfinite(df[t].to_numpy(float)).sum()) if t in df else 0
        if n < MIN_ANCHORED_STARS and w.get(l, 0) > 0:
            w[l] = 0.0
            notes.append(f"  ⚠ {l}: tylko {n} gwiazd z kotwicą '{t}' – waga dopasowania = 0 (uczenie z rekonstrukcji)")
        elif t in ANCHOR_COLS.values():
            notes.append(f"  {l}: kotwica '{t}' dla {n} gwiazd")
    return w, notes


def decoder_input(name, w, z):
    """Wejście dekodera: latent 'trains' z gradientem, pozostałe odłączone (detach)."""
    cfg = DECODERS[name]; parts = []
    for l in cfg["inputs"]:
        if l in ATM_LATENTS:
            parts.append(w[:, ATM_LATENTS.index(l)].unsqueeze(1).detach())
        else:
            v = z[:, chem_index(l)].unsqueeze(1)
            parts.append(v if l == cfg["trains"] else v.detach())
    return torch.cat(parts, 1)


def train_epoch(loader, P, opts, lam_adv, beta_kl=BETA_KL, fit_weights=None) -> dict:
    """
    Jedna epoka; na każdym batchu:
      1) dyskryminator uczy się odczytywać Teff/logg z ADV_LATENTS.detach(),
      2) krok atmosfery: L_phys + β·KL − λ_adv·CE_adv  (enkoder + linear_w),
      3) po jednym kroku na każdy dekoder z DECODERS: rekonstrukcja jego okien;
         gradient płynie tylko do latentu 'trains' (i do dekodera).
    """
    o_enc, o_w, o_d, o_dec = opts["encoder"], opts["linear_w"], opts["discriminator"], opts["decoders"]
    sums, n = {}, 0
    adv_idx = [chem_index(l) for l in ADV_LATENTS]

    def acc(k, v, bs):
        sums[k] = sums.get(k, 0.0) + float(v) * bs

    for batch in loader:
        xb, tg, tb, lb = [t.to(DEVICE) for t in batch[:4]]
        xdec = {k: t.to(DEVICE) for k, t in zip(DECODERS, batch[4:])}
        targets = {p: tg[:, i] for i, p in enumerate(TARGET_PARAMS)}
        bs = xb.size(0)
        # 1) dyskryminator
        with torch.no_grad():
            _, z_det, _, _ = encoder_forward(xb, P["encoder"])
        tl, ll = discriminator_forward(z_det[:, adv_idx], P["discriminator"])
        l_d = F.cross_entropy(tl, tb) + F.cross_entropy(ll, lb)
        o_d.zero_grad(); l_d.backward(); o_d.step()
        # 2) atmosfera + kary fizyczne
        o_enc.zero_grad(); o_w.zero_grad()
        w, z, mu, lv = encoder_forward(xb, P["encoder"])
        tl, ll = discriminator_forward(z[:, adv_idx], P["discriminator"])
        l_adv = F.cross_entropy(tl, tb) + F.cross_entropy(ll, lb)
        l_kl = kl_diag(mu, lv)
        l_phys, terms = physical_loss(w, mu, targets, fit_weights)
        l_atm = l_phys + beta_kl * l_kl - lam_adv * l_adv
        l_atm.backward(); o_enc.step(); o_w.step()
        total = l_atm.item()
        # 3) dekodery
        for k, cfg in DECODERS.items():
            o_enc.zero_grad(); o_dec[k].zero_grad()
            w, z, mu, lv = encoder_forward(xb, P["encoder"])
            pred = region_decoder_forward(decoder_input(k, w, z), P[f"{k}_decoder"])
            l_rec = F.mse_loss(pred, xdec[k])
            j = chem_index(cfg["trains"])
            l_k = l_rec + beta_kl * kl_diag(mu[:, j:j + 1], lv[:, j:j + 1])
            l_k.backward(); o_enc.step(); o_dec[k].step()
            acc(f"rec_{k}", l_rec.item(), bs); total += l_k.item()

        acc("total", total, bs)
        acc("kl", l_kl.item(), bs); acc("adv_disc", l_d.item(), bs); acc("adv_enc", l_adv.item(), bs)
        for k, v in terms.items():
            acc(k, v.item(), bs)
        n += bs
    return {k: v / max(n, 1) for k, v in sums.items()}


def train_model(X, df, dec_idx, out_dir: Path = ASTRO_REPORTS_DIR):
    """
    Pełny trening VAE; zapisuje model (.pt), metadane (.json), metryki
    epok (CSV), raport tekstowy treningu i wykres przebiegu kar.
    """
    print("=" * 80)
    print("KROK 2.6: TRENING FIZYCZNIE USTRUKTURYZOWANEGO VAE")
    print("=" * 80)
    ensure_dir(MODEL_DIR)
    fit_w, fit_notes = effective_fit_weights(df)
    for line in fit_notes:
        print(line)
    mean, std = input_norm_stats(X)
    ds, targ, meta = prepare_training_data(X, df, dec_idx, (X - mean) / std)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    P = init_model_params(X.shape[1], {k: len(dec_idx[k]) for k in DECODERS})
    P["input_norm"] = {"mode": INPUT_NORMALIZATION, "mean": torch.tensor(mean), "std": torch.tensor(std)}
    opts = {k: torch.optim.Adam(collect_param_tensors(P[k]), lr=LEARNING_RATE)
            for k in ["encoder", "linear_w", "discriminator"]}
    opts["decoders"] = {k: torch.optim.Adam(collect_param_tensors(P[f"{k}_decoder"]), lr=LEARNING_RATE)
                        for k in DECODERS}
    rows = []
    for ep in range(EPOCHS):
        lam = lambda_adv_at(ep, EPOCHS)
        row = train_epoch(loader, P, opts, lam, fit_weights=fit_w)
        row.update(epoch=ep + 1, lambda_adv=lam)
        rows.append(row)
        print(f"  epoka {ep + 1:3d}/{EPOCHS}  total={row['total']:.4f}  "
              + "  ".join(f"rec_{k}={row[f'rec_{k}']:.4f}" for k in DECODERS)
              + f"  phys={row['physical_total']:.4f}  λ_adv={lam}")
    hist = pd.DataFrame(rows)
    hist.to_csv(out_dir / "02_training_metrics.csv", index=False)
    torch.save(P, MODEL_PATH)
    meta.update(dict(created=f"{datetime.now():%Y-%m-%d %H:%M:%S}", input_len=int(X.shape[1]),
                     decoder_pixels={k: int(len(dec_idx[k])) for k in DECODERS},
                     signature_regions=SIGNATURE_REGIONS, decoders=DECODERS,
                     atm_latents=ATM_LATENTS, chem_latents=CHEM_LATENTS, adv_latents=ADV_LATENTS,
                     latent_targets=LATENT_TARGETS, fit_weights=fit_w, fit_notes=fit_notes,
                     leakage_penalties=LEAKAGE_PENALTIES, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     learning_rate=LEARNING_RATE, beta_kl=BETA_KL, input_normalization=INPUT_NORMALIZATION,
                     lambda_adv_schedule=LAMBDA_ADV_SCHEDULE, default_lambda_adv=DEFAULT_LAMBDA_ADV,
                     n_parameters={k: count_params(v) for k, v in P.items() if k != "input_norm"}))
    MODEL_META_JSON.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"  ✓ model: {MODEL_PATH}")
    report_training(hist, meta, out_dir)
    return P, meta, hist


def load_model(dec_idx, input_len):
    """Wczytuje zapisany model i sprawdza zgodność z bieżącą siatką, oknami i latentami."""
    print("=" * 80)
    print("KROK 2.6: WCZYTANIE WYTRENOWANEGO MODELU")
    print("=" * 80)
    P = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    meta = json.loads(MODEL_META_JSON.read_text(encoding="utf-8")) if MODEL_META_JSON.exists() else {}
    if P["encoder"]["input_len"] != input_len:
        raise ValueError(f"Model ma input_len={P['encoder']['input_len']}, dane {input_len}")
    if P["encoder"]["fc_mu"]["weight"].shape[0] != len(CHEM_LATENTS):
        raise ValueError("Model ma inną liczbę latentów chemicznych (model z wersji 1?) – ustaw TRAIN_MODEL=True.")
    for k in DECODERS:
        if f"{k}_decoder" not in P:
            raise ValueError(f"W modelu brak dekodera '{k}' – ustaw TRAIN_MODEL=True.")
        n_out = P[f"{k}_decoder"]["fc4"]["weight"].shape[0]
        if n_out != len(dec_idx[k]):
            raise ValueError(f"Dekoder {k}: model {n_out} pikseli, okna {len(dec_idx[k])} – "
                             "zmieniono SIGNATURE_REGIONS? Ustaw TRAIN_MODEL=True.")
    print(f"  ✓ {MODEL_PATH}")
    return P, meta


def report_training(hist: pd.DataFrame, meta: dict, out_dir: Path = ASTRO_REPORTS_DIR):
    """Raport tekstowy i wykresy przebiegu treningu (rekonstrukcja, KL, adwersarz, kary)."""
    first, last = hist.iloc[0], hist.iloc[-1]
    keys = [c for c in hist.columns if c not in ("epoch",)]
    sec = [f"  {k:<36s} start={first[k]:.6f}  koniec={last[k]:.6f}  min={hist[k].min():.6f}" for k in keys]
    write_report(out_dir / "02_training_report.txt", "RAPORT TRENINGU VAE", [
        ("Architektura", [
            "  enkoder: 3×[Conv1d(k=7) + PReLU + MaxPool(2)] (1→8→16→32 kanałów) → FC128 → FC64",
            f"  głowy: w_atm ({len(ATM_LATENTS)}, deterministyczne), mu/logvar ({len(CHEM_LATENTS)}: {', '.join(CHEM_LATENTS)})",
            *[f"  dekoder {k:<6s}: {cfg['inputs']} (uczy {cfg['trains']}) → 32 → 64 → 128 → okna {cfg['groups']}"
              f" ({meta.get('decoder_pixels', {}).get(k, '?')} pikseli)" for k, cfg in DECODERS.items()],
            f"  dyskryminator: {ADV_LATENTS} → 32 → 32 → (Teff: {TEFF_BINS} klas, logg: {LOGG_BINS} klas)",
            f"  normalizacja wejścia: {meta.get('input_normalization', INPUT_NORMALIZATION)}",
            "  liczba parametrów: " + ", ".join(f"{k}={v}" for k, v in meta.get("n_parameters", {}).items()),
        ]),
        ("Cele latentów i kary", [f"  {l} → {t}  (waga dopasowania {meta.get('fit_weights', FIT_WEIGHTS).get(l)})"
                                  for l, t in LATENT_TARGETS.items()] + list(meta.get("fit_notes", []))
         + [f"  kara {k}: waga={v['weight']}, redukcja={v['reduction']}, pary={v['pairs']}"
            for k, v in LEAKAGE_PENALTIES.items()]),
        ("Hiperparametry", [f"  epoki={EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}, β_KL={BETA_KL}, "
                            f"λ_adv: {LAMBDA_ADV_SCHEDULE} / domyślnie {DEFAULT_LAMBDA_ADV}"]),
        ("Metryki (pierwsza / ostatnia epoka)", sec),
    ])
    groups = [("Rekonstrukcja", [f"rec_{k}" for k in DECODERS]),
              ("KL i adwersarz", ["kl", "adv_disc", "adv_enc"]),
              ("Dopasowanie latentów do celów", [c for c in hist.columns if c.startswith("fit_")]),
              ("Kary za przeciek", [c for c in hist.columns if c.startswith("leak_")])]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (t, cols) in zip(axes.flat, groups):
        for c in cols:
            ax.plot(hist["epoch"], hist[c], marker="o", ms=2, label=c)
        ax.set_title(t); ax.set_xlabel("epoka"); ax.legend(fontsize=7); ax.grid(alpha=0.25)
        if t == "Rekonstrukcja":
            ax.set_yscale("log")
    ax2 = axes[0, 1].twinx()
    ax2.step(hist["epoch"], hist["lambda_adv"], color="gray", ls="--", where="post")
    ax2.set_ylabel("λ_adv", color="gray")
    fig.suptitle("Przebieg treningu VAE", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir / "02_training_metrics.png")


# ============================================================================
# =================== 7. LATENTY, REKONSTRUKCJA, RAPORTY ======================
# ============================================================================

@torch.no_grad()


def encode_and_reconstruct(X, P, dec_idx, batch: int = 256):
    """Latenty (mu dla z_chem) i rekonstrukcje okien wszystkich dekoderów."""
    W, MU, REC = [], [], {k: [] for k in DECODERS}
    Xin = normalize_input(X, P)
    for s in range(0, X.shape[0], batch):
        xb = torch.tensor(Xin[s:s + batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)
        w, _, mu, _ = encoder_forward(xb, P["encoder"])
        for k in DECODERS:
            REC[k].append(region_decoder_forward(decoder_input(k, w, mu), P[f"{k}_decoder"]).cpu().numpy())
        W.append(w.cpu().numpy()); MU.append(mu.cpu().numpy())
    W, MU = np.vstack(W), np.vstack(MU)
    lat = pd.DataFrame({**{n: W[:, i] for i, n in enumerate(ATM_LATENTS)},
                        **{n: MU[:, i] for i, n in enumerate(CHEM_LATENTS)}})
    rec = {k: np.vstack(v) for k, v in REC.items()}
    tot, npx = 0.0, 0
    for k in DECODERS:
        lat[f"recon_mse_{k}"] = np.mean((X[:, dec_idx[k]] - rec[k]) ** 2, axis=1)
        tot = tot + lat[f"recon_mse_{k}"] * len(dec_idx[k]); npx += len(dec_idx[k])
    lat["recon_mse_total"] = tot / npx
    return lat, rec


def report_latent_correlations(lat: pd.DataFrame, df: pd.DataFrame, out_dir: Path = ASTRO_REPORTS_DIR):
    """
    Macierz korelacji latent × parametr (Pearson, Spearman), macierz przecieku
    r² z zaznaczeniem celów, wartości kar przecieku na całym zbiorze, R²
    dopasowania liniowego latent -> cel, wykresy: heatmapy, siatka 4×4
    rozrzutów latent vs parametr, corner plot latentów.
    """
    print("=" * 80)
    print("KROK 2.7: RAPORT LATENTÓW I PRZECIEKU")
    print("=" * 80)
    params = report_params(df)
    P_ = pd.DataFrame(index=LATENT_NAMES, columns=params, dtype=float)
    S_ = P_.copy(); Pv = P_.copy()
    for l in LATENT_NAMES:
        for p in params:
            P_.loc[l, p], Pv.loc[l, p] = corr_with_p(lat[l].to_numpy(), df[p].to_numpy(float))
            S_.loc[l, p], _ = corr_with_p(lat[l].to_numpy(), df[p].to_numpy(float), "spearman")
    P_.to_csv(out_dir / "02_latent_param_corr_pearson.csv")
    S_.to_csv(out_dir / "02_latent_param_corr_spearman.csv")
    R2 = P_ ** 2
    rows = []
    for l, t in LATENT_TARGETS.items():
        if t not in params:
            rows.append(dict(latent=l, target=t, r_target=np.nan, r2_target=np.nan, max_offtarget_param="",
                             max_offtarget_r2=np.nan, purity=np.nan))
            continue
        off = R2.loc[l].drop(t).dropna()
        rows.append(dict(latent=l, target=t, r_target=P_.loc[l, t], r2_target=R2.loc[l, t],
                         max_offtarget_param=off.idxmax(), max_offtarget_r2=off.max(),
                         purity=R2.loc[l, t] / max(R2.loc[l].sum(), 1e-12)))
    summ = pd.DataFrame(rows)
    summ.to_csv(out_dir / "02_latent_leakage_summary.csv", index=False)
    leak_vals = []
    for k, cfg in LEAKAGE_PENALTIES.items():
        v = [R2.loc[l, p] for l, p in cfg["pairs"] if p in params]
        val = (np.nansum(v) if cfg["reduction"] == "sum" else np.nanmean(v)) if v else np.nan
        leak_vals.append(f"  {k:<26s} = {val:.5f}  (waga {cfg['weight']}, pary: "
                         + ", ".join(f"{l}~{p}:{R2.loc[l, p]:.4f}" for l, p in cfg["pairs"] if p in params) + ")")
    lat_corr = lat[LATENT_NAMES].corr()
    write_report(out_dir / "02_latent_report.txt", "RAPORT ZMIENNYCH LATENTNYCH I PRZECIEKU", [
        ("Korelacje Pearsona latent × parametr", df_lines(P_, 4)),
        ("p-value (Pearson)", df_lines(Pv.astype(float), 6)),
        ("Korelacje Spearmana latent × parametr", df_lines(S_, 4)),
        ("Przeciek: cel vs najsilniejszy parametr niewłaściwy (r²); purity = r²_cel / Σr²",
         df_lines(summ, 4, index=False)),
        ("Wartości kar przecieku na całym zbiorze (z r² powyżej)", leak_vals),
        ("Korelacje między latentami (niezależność osi)", df_lines(lat_corr, 4)),
        ("Statystyki latentów", df_lines(lat[LATENT_NAMES].describe().T, 4)),
        ("Interpretacja", [
            "  Idealnie: |r| z celem bliskie 1, r² z pozostałymi parametrami bliskie 0, purity → 1.",
            "  Wysokie r² poza celem = przeciek informacji (np. Teff w z_Fe) – rozważ zwiększenie wagi kary.",
            "  Uwaga: parametry katalogowe same są skorelowane (patrz 01_catalog_parameters.txt),",
            "  więc część 'przecieku' jest fizycznie nieunikniona.",
            "  v2: korelacje między latentami chemicznymi NIE są karane – relacje [α/Fe]–[Fe/H] i C–N",
            "  mają zostać (patrz 'Korelacje między latentami'); z_C, z_N mogą korelować z log g (mieszanie).",
        ]),
    ])

    # heatmapy
    fig, axes = plt.subplots(1, 2, figsize=(4 + 1.6 * len(params), 1 + 0.9 * len(LATENT_NAMES)))
    for ax, M, t in [(axes[0], P_, "Pearson r"), (axes[1], R2, "r² (przeciek; ramka = cel)")]:
        im = ax.imshow(M.values.astype(float), cmap="coolwarm" if t.startswith("P") else "Reds",
                       vmin=-1 if t.startswith("P") else 0, vmax=1)
        ax.set_xticks(range(len(params)), params, rotation=30); ax.set_yticks(range(len(LATENT_NAMES)), LATENT_NAMES)
        for i, l in enumerate(LATENT_NAMES):
            for j, p in enumerate(params):
                ax.text(j, i, f"{M.loc[l, p]:.2f}", ha="center", va="center", fontsize=9)
                if LATENT_TARGETS[l] == p:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="k", lw=2.5))
        ax.set_title(t)
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle("Latenty vs parametry gwiazd", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir / "02_latent_param_corr_heatmap.png")

    # siatka latent × parametr
    fig, axes = plt.subplots(len(LATENT_NAMES), len(params), figsize=(3.3 * len(params), 2.4 * len(LATENT_NAMES)),
                             squeeze=False)
    for i, l in enumerate(LATENT_NAMES):
        for j, p in enumerate(params):
            ax = axes[i, j]
            x, y = df[p].to_numpy(float), lat[l].to_numpy()
            ax.scatter(x, y, s=3, alpha=0.35, color="crimson" if LATENT_TARGETS[l] == p else "gray")
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 2:
                a, b = np.polyfit(x[ok], y[ok], 1)
                xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
                ax.plot(xs, a * xs + b, "k-", lw=1)
            ax.set_title(f"r={P_.loc[l, p]:+.2f}", fontsize=8)
            if i == len(LATENT_NAMES) - 1:
                ax.set_xlabel(plabel(p))
            if j == 0:
                ax.set_ylabel(l)
            ax.tick_params(labelsize=7)
    fig.suptitle("Macierz rozrzutu latent × parametr (czerwone: zamierzony cel)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir / "02_latent_param_scatter_grid.png")

    # corner plot
    n = len(LATENT_NAMES)
    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(lat[LATENT_NAMES[i]], bins=40, color="steelblue")
            elif i > j:
                sc = ax.scatter(lat[LATENT_NAMES[j]], lat[LATENT_NAMES[i]], c=df["teff"], s=3,
                                cmap="coolwarm_r", alpha=0.6)
            else:
                ax.scatter(lat[LATENT_NAMES[j]], lat[LATENT_NAMES[i]], c=df["feh"], s=3,
                           cmap="viridis", alpha=0.6)
            if i == n - 1:
                ax.set_xlabel(LATENT_NAMES[j])
            if j == 0:
                ax.set_ylabel(LATENT_NAMES[i])
            ax.tick_params(labelsize=7)
    fig.suptitle("Corner plot latentów (dół: kolor Teff, góra: kolor [Fe/H])", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir / "02_latent_corner.png")
    return P_, summ


def report_reconstruction(X, wave, lat, rec, dec_idx, df, out_dir: Path = ASTRO_REPORTS_DIR):
    """
    Jakość rekonstrukcji okien: statystyki MSE, MSE vs parametry (korelacje),
    histogram MSE, wykresy rekonstrukcji dla gwiazd o najmniejszym,
    medianowym i największym MSE w każdym oknie.
    """
    print("=" * 80)
    print("KROK 2.8: RAPORT REKONSTRUKCJI OKIEN")
    print("=" * 80)
    mse_cols = [f"recon_mse_{k}" for k in DECODERS] + ["recon_mse_total"]
    sec = [f"  {c:<18s} " + "  ".join(f"{k}={v:.3e}" for k, v in lat[c].describe().items() if k != "count")
           for c in mse_cols]
    corr_lines = []
    for p in report_params(df) + (["snrg"] if "snrg" in df.columns else []):
        r, pv = corr_with_p(lat["recon_mse_total"].to_numpy(), df[p].to_numpy(float), "spearman")
        corr_lines.append(f"  Spearman(MSE_total, {p:<9s}) = {r:+.3f}  (p={pv:.2e})")
    worst = lat["recon_mse_total"].sort_values(ascending=False).head(15).index
    w_lines = [f"  row={i:5d}  MSE={lat.loc[i, 'recon_mse_total']:.3e}  Teff={df.loc[i, 'teff']:.0f} "
               f"logg={df.loc[i, 'logg']:.2f} [Fe/H]={df.loc[i, 'feh']:+.2f} [α/M]={df.loc[i, 'alpha_m']:+.2f} "
               f"{df.loc[i, 'subclass'] if 'subclass' in df else ''}" for i in worst]
    write_report(out_dir / "02_reconstruction_report.txt", "RAPORT REKONSTRUKCJI OKIEN SYGNATUR", [
        ("MSE rekonstrukcji", sec), ("Zależność MSE od parametrów", corr_lines),
        ("15 gwiazd najgorzej rekonstruowanych", w_lines)])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(np.log10(lat["recon_mse_total"]), bins=50, color="slateblue", edgecolor="k")
    ax.set_xlabel("log10(MSE rekonstrukcji)"); ax.set_ylabel("N"); ax.grid(alpha=0.25)
    ax.set_title("Rozkład błędu rekonstrukcji okien")
    save_fig(fig, out_dir / "02_reconstruction_mse_hist.png")

    order = lat["recon_mse_total"].sort_values().index
    picks = {"najlepsza": order[0], "mediana": order[len(order) // 2], "najgorsza": order[-1]}
    full = np.full_like(X, np.nan)
    for k in DECODERS:                       # każde okno pokazane z dekodera, który je rekonstruuje
        full[:, dec_idx[k]] = rec[k]
    ncol = len(SIGNATURE_REGIONS)
    fig, axes = plt.subplots(len(picks), ncol, figsize=(2.2 * ncol, 2.6 * len(picks)), squeeze=False)
    for i, (lab, rid) in enumerate(picks.items()):
        for j, r in enumerate(SIGNATURE_REGIONS):
            ax = axes[i, j]
            m = (wave >= r["lambda_min"] - 3) & (wave <= r["lambda_max"] + 3)
            ax.plot(wave[m], X[rid, m], "k-", lw=1, label="widmo")
            mm = region_mask(wave, r)
            ax.plot(wave[mm], full[rid, mm], "r.-", lw=1, ms=3, label="rekonstrukcja")
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(r["name"], fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{lab}\nrow={rid}", fontsize=8)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Rekonstrukcje okien sygnatur (najlepsza / medianowa / najgorsza gwiazda)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir / "02_reconstruction_examples.png")


def save_qiftdata(df, lat, sig, out_path: Path = QIFTDATA_OUT_PATH) -> pd.DataFrame:
    """Zapis qiftdata.csv: katalog + latenty + MSE + jasności sygnatur (wejście plików 03–05)."""
    print("=" * 80)
    print("KROK 2.9: ZAPIS qiftdata.csv")
    print("=" * 80)
    merged = pd.concat([df.reset_index(drop=True), lat.reset_index(drop=True),
                        sig.reset_index(drop=True)], axis=1)
    merged.to_csv(out_path, index=False)
    print(f"  ✓ {out_path}  shape={merged.shape}")
    return merged


# ============================================================================
# ======================= URUCHOMIENIE PIPELINE'U =============================
# ============================================================================

def run_pipeline():
    ensure_dir(ASTRO_REPORTS_DIR)
    X, wave, df = load_inputs()
    dec_idx, _ = build_signature_index(wave)

    sig = compute_signature_brightness(X, wave)
    report_signature_correlations(sig, df)
    report_pixel_correlation_spectrum(X, wave, df)
    plot_signature_fragments(X, wave, df)

    if TRAIN_MODEL or not MODEL_PATH.exists():
        P, meta, _ = train_model(X, df, dec_idx)
    else:
        P, meta = load_model(dec_idx, X.shape[1])

    lat, rec = encode_and_reconstruct(X, P, dec_idx)
    report_latent_correlations(lat, df)
    report_reconstruction(X, wave, lat, rec, dec_idx, df)
    qd = save_qiftdata(df, lat, sig)
    print("\n✓ KONIEC KROKU 2. Raporty w:", ASTRO_REPORTS_DIR.resolve())
    return dict(qiftdata=qd, latents=lat, signatures=sig, model=P)


if __name__ == "__main__":
    results_02 = run_pipeline()
