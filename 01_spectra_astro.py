# -*- coding: utf-8 -*-
"""
01_spectra_astro.py
===================
KROK 1 potoku QIFT-spectra: wczytanie widm gwiazd, część astrofizyczna,
pomiar przesunięcia ku czerwieni i deredshift.

Co robi ten plik
----------------
1. Wczytuje surowe widma (NPZ: X [n_gwiazd, n_pikseli] + wave_grid) oraz
   katalog parametrów (CSV: Teff, log g, [Fe/H], [alpha/M], z, S/N, ...).
2. Sprawdza spójność danych i tworzy raport jakości siatki długości fal
   i widm (krok siatki, efektywna rozdzielczość R, braki danych).
3. Tworzy raport parametrów katalogowych (rozkłady, diagram Kiela,
   diagram [Fe/H]–[alpha/M], mapa nieba, macierz korelacji).
4. Raportuje położenie linii referencyjnych (H, Fe, Mg, Ca, Na) na siatce.
5. Mierzy redshift z wielu linii widmowych (wersja odporna z odrzucaniem
   linii odstających) i porównuje z redshiftem katalogowym.
6. Przenosi widma do układu spoczynkowego (deredshift) wybranym źródłem z.
7. Waliduje deredshift (redshift resztkowy po korekcie) i rysuje wykresy.
8. Zapisuje widma deredshiftowane (NPZ) oraz katalog z kolumnami redshiftu
   (CSV) – są to dane wejściowe dla pliku 02.

Wszystkie raporty tekstowe, tabele i wykresy trafiają do folderu
ASTRO_REPORTS_DIR (domyślnie: astro_reports/).

Uruchomienie: python 01_spectra_astro.py   (albo F5 w Spyderze)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================================
# ======================= PARAMETRY URUCHOMIENIOWE ===========================
# ============================================================================

# --- Ścieżki wejściowe ---
RAW_NPZ_PATH = Path("warkocz_bereniki_dataset.npz")          # surowe widma
CATALOG_CSV_PATH = Path("warkocz_bereniki_full_clean.csv")   # katalog LAMOST

# --- Folder raportów i pliki wyjściowe ---
ASTRO_REPORTS_DIR = Path("astro_reports")
DERED_NPZ_PATH = ASTRO_REPORTS_DIR / "spectra_deredshifted.npz"
CATALOG_OUT_CSV = ASTRO_REPORTS_DIR / "catalog_with_redshift.csv"

# --- Kolumny katalogu ---
PARAM_COLS = ["teff", "logg", "feh", "alpha_m"]
PARAM_LABELS = {"teff": "Teff [K]", "logg": "log g [dex]",
                "feh": "[Fe/H] [dex]", "alpha_m": "[α/M] [dex]"}
REQUIRED_COLS = PARAM_COLS + ["z"]
ID_COLS = ["obsid", "designation", "subclass"]
SNR_COL = "snrg"
MISSING_VALUE_THRESHOLD = -900.0     # wartości <= progu (np. -9999) traktowane jako brak danych

# --- Deredshift ---
# Źródło redshiftu użytego do deredshiftu:
#   "catalog"                  – kolumna z z katalogu (jak w 01.redshift.py)
#   "measured"                 – mediana z wielu linii (jak w 06.redshift.py)
#   "measured_fallback_catalog"– pomiar, a gdy się nie uda – katalog
Z_SOURCE = "catalog"

# Linie do pomiaru redshiftu: (nazwa, lambda_spoczynkowa [Å], pół-okno [Å])
REDSHIFT_LINES = [
    ("Hdelta", 4101.74, 12.0),
    ("Hgamma", 4340.47, 12.0),
    ("Hbeta", 4861.33, 12.0),
    ("Halpha", 6562.80, 15.0),
    ("CaII_8498", 8498.02, 10.0),
    ("CaII_8542", 8542.09, 10.0),
    ("CaII_8662", 8662.14, 10.0),
]
REDSHIFT_OUTLIER_THRESHOLD = 1e-4   # odrzucenie linii |z_linia - mediana| > próg
REDSHIFT_SMOOTH_WINDOW = 5          # wygładzanie średnią ruchomą [piksele]
REDSHIFT_FIT_HALF_WIDTH = 4         # pół-szerokość dopasowania paraboli [piksele]

# --- Linie referencyjne raportowane na siatce (nazwa, lambda [Å]) ---
REFERENCE_LINES = [
    ("H I Halpha", 6562.80), ("H I Hbeta", 4861.33), ("H I Hgamma", 4340.47),
    ("H I Hdelta", 4101.74), ("Ca II K", 3933.66), ("Ca II H", 3968.47),
    ("Fe I 4045.81", 4045.81), ("Fe I 4383.55", 4383.55), ("Fe I 5216.75", 5216.75),
    ("Fe I 5269.54", 5269.54), ("Fe I 5328.04", 5328.04), ("Fe I 5615.64", 5615.64),
    ("Mg I b1 5183.60", 5183.60), ("Mg I b2 5172.68", 5172.68), ("Mg I b4 5167.32", 5167.32),
    ("Na I D2", 5889.95), ("Na I D1", 5895.92),
    ("Ca II 8498", 8498.02), ("Ca II 8542", 8542.09), ("Ca II 8662", 8662.14),
]

# --- Wykresy ---
N_RANDOM_SPECTRA = 10
EXAMPLES_XLIM = (6450.0, 6680.0)     # okno przykładów deredshiftu (wokół Hα)
SINGLE_STAR_ROW_IDS = [1433]         # gwiazdy do diagnostyki indywidualnej
DPI = 180
SEED = 42

C_KMS = 299_792.458                  # prędkość światła [km/s]


# ============================================================================
# ========================= FUNKCJE POMOCNICZE ================================
# ============================================================================

def ensure_dir(path: Path) -> Path:
    """Tworzy folder (rekurencyjnie), jeśli nie istnieje."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_report(path: Path, title: str, sections: list[tuple[str, list[str]]]) -> Path:
    """
    Zapisuje raport tekstowy złożony z sekcji (nagłówek, lista linii).
    Każdy raport ma nagłówek z datą wygenerowania.
    """
    lines = ["=" * 80, title, f"Wygenerowano: {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 80, ""]
    for heading, body in sections:
        lines += [heading, "-" * len(heading)] + list(body) + [""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ raport: {path}")
    return Path(path)


def save_fig(fig, path: Path, dpi: int = DPI) -> Path:
    """Zapisuje i zamyka figurę matplotlib."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ wykres: {path}")
    return Path(path)


def describe(values) -> dict:
    """Statystyki opisowe wektora (z pominięciem NaN)."""
    v = pd.Series(values, dtype=float).dropna()
    if len(v) == 0:
        return dict(n=0, mean=np.nan, std=np.nan, min=np.nan, p05=np.nan, q25=np.nan,
                    median=np.nan, q75=np.nan, p95=np.nan, max=np.nan)
    return dict(n=int(len(v)), mean=v.mean(), std=v.std(ddof=1) if len(v) > 1 else 0.0,
                min=v.min(), p05=v.quantile(.05), q25=v.quantile(.25), median=v.median(),
                q75=v.quantile(.75), p95=v.quantile(.95), max=v.max())


def describe_lines(name: str, values, fmt: str = ".6g") -> str:
    """Jedna linia raportu ze statystykami opisowymi."""
    d = describe(values)
    return (f"  {name:<22s} n={d['n']:<5d} mean={d['mean']:{fmt}}  std={d['std']:{fmt}}  "
            f"min={d['min']:{fmt}}  q25={d['q25']:{fmt}}  med={d['median']:{fmt}}  "
            f"q75={d['q75']:{fmt}}  max={d['max']:{fmt}}")


# ============================================================================
# ======================== 1. WCZYTYWANIE DANYCH ==============================
# ============================================================================

def load_raw_spectra(npz_path: Path = RAW_NPZ_PATH):
    """
    Wczytuje widma z pliku NPZ. Automatycznie rozpoznaje klucze siatki
    długości fal (wave_grid / wavelength / wave / lam / lambda / wavegrid)
    i macierzy widm (X / spectra / flux / spec / arr_0).

    Zwraca: wave_grid [n_pix], X [n_star, n_pix], keys (dict z nazwami kluczy).
    """
    print("=" * 80)
    print("KROK 1.1: WCZYTANIE WIDM")
    print("=" * 80)
    with np.load(npz_path, allow_pickle=True) as data:
        files = list(data.files)
        wave_key = next((k for k in ["wave_grid", "wavegrid", "wavelength", "wave", "lam", "lambda"]
                         if k in files), None)
        flux_key = next((k for k in ["X", "spectra", "flux", "spec", "arr_0"] if k in files), None)
        if wave_key is None or flux_key is None:
            raise KeyError(f"Nie rozpoznano kluczy siatki/widm w NPZ. Dostępne: {files}")
        wave = np.asarray(data[wave_key], dtype=np.float64)
        X = np.asarray(data[flux_key], dtype=np.float64)

    if X.ndim != 2 or wave.ndim != 1 or X.shape[1] != wave.shape[0]:
        raise ValueError(f"Niezgodne wymiary: X={X.shape}, wave_grid={wave.shape}")
    print(f"✓ {npz_path}: {X.shape[0]} widm × {X.shape[1]} pikseli "
          f"(klucze: {wave_key}, {flux_key})")
    return wave, X, {"wave_key": wave_key, "flux_key": flux_key, "all_keys": files}


def load_catalog(csv_path: Path = CATALOG_CSV_PATH, n_spectra: int | None = None) -> pd.DataFrame:
    """
    Wczytuje katalog parametrów gwiazd. Usuwa kolumnę indeksu z pandas
    ('Unnamed: 0'), konwertuje parametry na liczby i sprawdza, czy liczba
    wierszy zgadza się z liczbą widm (wiersz i <-> widmo i).
    """
    print("=" * 80)
    print("KROK 1.2: WCZYTANIE KATALOGU")
    print("=" * 80)
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Brak wymaganych kolumn w katalogu: {missing}")
    for c in REQUIRED_COLS + [SNR_COL, "ra", "dec", "gaia_g_mean_mag"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Wartości-wypełniacze braków danych (np. -9999 w LAMOST/Gaia) -> NaN
    for c in df.select_dtypes(include=[np.number]).columns:
        bad = df[c] <= MISSING_VALUE_THRESHOLD
        if bad.any():
            print(f"  ⚠ {c}: {int(bad.sum())} wartości ≤ {MISSING_VALUE_THRESHOLD} zamieniono na NaN")
            df.loc[bad, c] = np.nan
    if n_spectra is not None and len(df) != n_spectra:
        raise ValueError(f"Katalog ma {len(df)} wierszy, a NPZ {n_spectra} widm.")
    df = df.reset_index(drop=True)
    df.insert(0, "row_id", np.arange(len(df)))
    print(f"✓ {csv_path}: {len(df)} wierszy, {len(df.columns)} kolumn")
    return df


# ============================================================================
# =================== 2. RAPORT JAKOŚCI WIDM I SIATKI =========================
# ============================================================================

def report_spectra_quality(wave: np.ndarray, X: np.ndarray, df: pd.DataFrame,
                           out_dir: Path = ASTRO_REPORTS_DIR, tag: str = "raw") -> dict:
    """
    Raport jakości siatki i widm: zakres λ, krok siatki, efektywna
    rozdzielczość R = λ/Δλ, monotoniczność, braki danych, rozkład strumienia.
    Wykresy: losowe widma, widmo medianowe z pasmem 16–84%, krok siatki vs λ,
    mapa wszystkich widm (heatmapa posortowana wg Teff).
    """
    print("=" * 80)
    print(f"KROK 1.3: RAPORT JAKOŚCI WIDM [{tag}]")
    print("=" * 80)
    out_dir = ensure_dir(out_dir)
    steps = np.diff(wave)
    nan_frac_star = np.mean(~np.isfinite(X), axis=1)
    med_flux_star = np.nanmedian(X, axis=1)
    lam_ha = 6562.8
    i_ha = int(np.argmin(np.abs(wave - lam_ha)))
    step_ha = float(steps[min(i_ha, len(steps) - 1)])

    info = dict(n_stars=int(X.shape[0]), n_pixels=int(X.shape[1]),
                lambda_min=float(wave.min()), lambda_max=float(wave.max()),
                step_mean=float(steps.mean()), step_median=float(np.median(steps)),
                step_min=float(steps.min()), step_max=float(steps.max()),
                monotonic=bool(np.all(steps > 0)),
                R_at_Halpha=float(lam_ha / step_ha),
                log_lambda_step_median=float(np.median(np.diff(np.log10(wave)))),
                nan_fraction_total=float(np.mean(~np.isfinite(X))),
                n_stars_with_nan=int(np.sum(nan_frac_star > 0)))

    sec_grid = [
        f"  Liczba widm                : {info['n_stars']}",
        f"  Liczba pikseli             : {info['n_pixels']}",
        f"  Zakres λ                   : {info['lambda_min']:.2f} – {info['lambda_max']:.2f} Å",
        f"  Krok Δλ: średni / mediana  : {info['step_mean']:.4f} / {info['step_median']:.4f} Å",
        f"  Krok Δλ: min / max         : {info['step_min']:.4f} / {info['step_max']:.4f} Å",
        f"  Krok w log10(λ) (mediana)  : {info['log_lambda_step_median']:.3e}",
        f"  Siatka rosnąca monotonicznie: {info['monotonic']}",
        f"  Efektywne R=λ/Δλ przy Hα   : {info['R_at_Halpha']:.0f} (próbkowanie siatki, nie R instrumentu)",
        f"  Prędkość na 1 piksel przy Hα: {C_KMS * step_ha / lam_ha:.1f} km/s",
    ]
    sec_flux = [
        f"  Udział NaN w całej macierzy: {info['nan_fraction_total']:.4%}",
        f"  Gwiazd z co najmniej 1 NaN : {info['n_stars_with_nan']}",
        describe_lines("mediana strumienia", med_flux_star),
        describe_lines("min strumienia", np.nanmin(X, axis=1)),
        describe_lines("max strumienia", np.nanmax(X, axis=1)),
        describe_lines("std strumienia", np.nanstd(X, axis=1)),
    ]
    if SNR_COL in df.columns:
        sec_flux.append(describe_lines(f"S/N ({SNR_COL})", df[SNR_COL], fmt=".4g"))
    write_report(out_dir / f"01_spectra_quality_{tag}.txt",
                 f"RAPORT JAKOŚCI WIDM I SIATKI DŁUGOŚCI FAL [{tag}]",
                 [("Siatka długości fal", sec_grid), ("Strumień", sec_flux)])
    (out_dir / f"01_spectra_quality_{tag}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # --- wykres: losowe widma ---
    rng = np.random.default_rng(SEED)
    idx = rng.choice(X.shape[0], size=min(N_RANDOM_SPECTRA, X.shape[0]), replace=False)
    fig, ax = plt.subplots(figsize=(14, 5))
    for k, i in enumerate(idx):
        ax.plot(wave, X[i] + 0.0 * k, lw=0.6, alpha=0.7,
                label=f"#{i} Teff={df.loc[i, 'teff']:.0f}")
    ax.set_xlabel("λ [Å]"); ax.set_ylabel("strumień (znormalizowany)")
    ax.set_title(f"{len(idx)} losowych widm [{tag}]")
    ax.legend(fontsize=7, ncol=5); ax.grid(alpha=0.25)
    save_fig(fig, out_dir / f"01_random_spectra_{tag}.png")

    # --- wykres: widmo medianowe z pasmem percentyli ---
    p16, p50, p84 = np.nanpercentile(X, [16, 50, 84], axis=0)
    p02, p98 = np.nanpercentile(X, [2.5, 97.5], axis=0)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(wave, p02, p98, color="lightsteelblue", alpha=0.5, label="2.5–97.5%")
    ax.fill_between(wave, p16, p84, color="steelblue", alpha=0.5, label="16–84%")
    ax.plot(wave, p50, color="navy", lw=0.7, label="mediana")
    for name, lam in REFERENCE_LINES:
        if wave.min() <= lam <= wave.max() and name.startswith(("H I", "Ca II", "Mg I b2", "Na I D2")):
            ax.axvline(lam, color="gray", ls=":", lw=0.7)
            ax.text(lam, ax.get_ylim()[1] if False else np.nanmax(p98), name, rotation=90,
                    fontsize=6, va="top", ha="right", color="dimgray")
    ax.set_xlabel("λ [Å]"); ax.set_ylabel("strumień")
    ax.set_title(f"Widmo medianowe populacji i rozrzut [{tag}]")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    save_fig(fig, out_dir / f"01_median_spectrum_band_{tag}.png")

    # --- wykres: heatmapa widm posortowanych wg Teff ---
    order = np.argsort(df["teff"].to_numpy())
    fig, ax = plt.subplots(figsize=(14, 6))
    vmin, vmax = np.nanpercentile(X, [1, 99])
    im = ax.imshow(X[order], aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,
                   extent=[wave.min(), wave.max(), len(order), 0])
    ax.set_xlabel("λ [Å]"); ax.set_ylabel("gwiazdy posortowane wg Teff (rosnąco)")
    ax.set_title(f"Mapa wszystkich widm (posortowane wg Teff) [{tag}]")
    fig.colorbar(im, ax=ax, label="strumień")
    save_fig(fig, out_dir / f"01_spectra_heatmap_by_teff_{tag}.png")

    # --- wykres: krok siatki ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wave[1:], steps, lw=0.8)
    ax.set_xlabel("λ [Å]"); ax.set_ylabel("Δλ [Å]"); ax.set_title("Krok siatki długości fal")
    ax.grid(alpha=0.25)
    save_fig(fig, out_dir / f"01_grid_step_{tag}.png")
    return info


# ============================================================================
# ===================== 3. RAPORT PARAMETRÓW KATALOGU =========================
# ============================================================================

def report_catalog_parameters(df: pd.DataFrame, out_dir: Path = ASTRO_REPORTS_DIR) -> pd.DataFrame:
    """
    Raport parametrów fizycznych z katalogu: statystyki Teff, log g, [Fe/H],
    [α/M] (z błędami), typy widmowe (subclass), macierz korelacji Pearsona
    i Spearmana. Wykresy: histogramy, diagram Kiela, diagram chemiczny,
    mapa nieba, heatmapa korelacji.
    """
    print("=" * 80)
    print("KROK 1.4: RAPORT PARAMETRÓW KATALOGOWYCH")
    print("=" * 80)
    out_dir = ensure_dir(out_dir)
    sec_stats = [describe_lines(c, df[c], ".5g") for c in PARAM_COLS]
    sec_stats += [describe_lines(f"{c}_err", df[f"{c}_err"], ".4g")
                  for c in PARAM_COLS if f"{c}_err" in df.columns]
    sec_stats += [describe_lines("z (katalog)", df["z"], ".4e"),
                  describe_lines("RV = c·z [km/s]", df["z"] * C_KMS, ".4g")]
    for c in [SNR_COL, "gaia_g_mean_mag"]:
        if c in df.columns:
            sec_stats.append(describe_lines(c, df[c], ".4g"))

    sec_sub = []
    if "subclass" in df.columns:
        vc = df["subclass"].value_counts()
        sec_sub = [f"  {k:<6s}: {v:5d}  ({v / len(df):.1%})" for k, v in vc.items()]

    corr_p = df[PARAM_COLS].corr(method="pearson")
    corr_s = df[PARAM_COLS].corr(method="spearman")
    corr_p.to_csv(out_dir / "01_catalog_param_corr_pearson.csv")
    corr_s.to_csv(out_dir / "01_catalog_param_corr_spearman.csv")

    # Prosta klasyfikacja ewolucyjna (karły / podolbrzymy / olbrzymy) wg log g
    evo = pd.cut(df["logg"], [-np.inf, 3.5, 4.0, np.inf], labels=["olbrzymy (logg<3.5)",
                 "podolbrzymy (3.5–4.0)", "karły (logg≥4.0)"])
    sec_evo = [f"  {k:<24s}: {v:5d}" for k, v in evo.value_counts().sort_index().items()]
    # Metaliczność: grupy populacyjne
    met = pd.cut(df["feh"], [-np.inf, -1.0, -0.5, 0.0, np.inf],
                 labels=["[Fe/H]<-1 (halo-like)", "-1..-0.5 (gruby dysk)", "-0.5..0", "≥0 (bogate)"])
    sec_met = [f"  {k:<24s}: {v:5d}" for k, v in met.value_counts().sort_index().items()]

    write_report(out_dir / "01_catalog_parameters.txt", "RAPORT PARAMETRÓW KATALOGOWYCH GWIAZD", [
        ("Statystyki opisowe", sec_stats),
        ("Typy widmowe (subclass)", sec_sub or ["  brak kolumny subclass"]),
        ("Podział wg log g (stadium ewolucyjne)", sec_evo),
        ("Podział wg metaliczności", sec_met),
        ("Korelacje Pearsona", ["  " + l for l in corr_p.round(4).to_string().splitlines()]),
        ("Korelacje Spearmana", ["  " + l for l in corr_s.round(4).to_string().splitlines()]),
    ])

    # --- histogramy ---
    cols = PARAM_COLS + ["z"] + ([SNR_COL] if SNR_COL in df.columns else [])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.flat, cols):
        vals = df[c] * (C_KMS if c == "z" else 1)
        ax.hist(vals.dropna(), bins=40, color="steelblue", edgecolor="black", alpha=0.8)
        ax.axvline(vals.median(), color="red", ls="--", lw=1.5, label=f"mediana={vals.median():.3g}")
        ax.set_xlabel("RV = c·z [km/s]" if c == "z" else PARAM_LABELS.get(c, c))
        ax.set_ylabel("N"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    for ax in list(axes.flat)[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Rozkłady parametrów katalogowych", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir / "01_catalog_histograms.png")

    # --- diagram Kiela i diagram chemiczny ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sc = axes[0].scatter(df["teff"], df["logg"], c=df["feh"], s=8, cmap="viridis", alpha=0.8)
    axes[0].invert_xaxis(); axes[0].invert_yaxis()
    axes[0].set_xlabel(PARAM_LABELS["teff"]); axes[0].set_ylabel(PARAM_LABELS["logg"])
    axes[0].set_title("Diagram Kiela (kolor: [Fe/H])")
    fig.colorbar(sc, ax=axes[0], label="[Fe/H]")
    sc = axes[1].scatter(df["feh"], df["alpha_m"], c=df["teff"], s=8, cmap="coolwarm_r", alpha=0.8)
    axes[1].set_xlabel(PARAM_LABELS["feh"]); axes[1].set_ylabel(PARAM_LABELS["alpha_m"])
    axes[1].set_title("Diagram chemiczny [Fe/H]–[α/M] (kolor: Teff)")
    fig.colorbar(sc, ax=axes[1], label="Teff [K]")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    save_fig(fig, out_dir / "01_kiel_and_chemical_diagram.png")

    # --- mapa nieba ---
    if {"ra", "dec"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(df["ra"], df["dec"], c=df["z"] * C_KMS, s=10, cmap="RdBu_r",
                        vmin=-np.nanpercentile(np.abs(df["z"] * C_KMS), 98),
                        vmax=np.nanpercentile(np.abs(df["z"] * C_KMS), 98))
        ax.invert_xaxis()
        ax.set_xlabel("RA [°]"); ax.set_ylabel("Dec [°]")
        ax.set_title("Mapa nieba pola (kolor: prędkość radialna c·z)")
        fig.colorbar(sc, ax=ax, label="RV [km/s]"); ax.grid(alpha=0.25)
        save_fig(fig, out_dir / "01_sky_map_rv.png")

    # --- heatmapa korelacji ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cm, name in zip(axes, [corr_p, corr_s], ["Pearson", "Spearman"]):
        im = ax.imshow(cm.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(PARAM_COLS)), PARAM_COLS, rotation=45)
        ax.set_yticks(range(len(PARAM_COLS)), PARAM_COLS)
        for i in range(len(PARAM_COLS)):
            for j in range(len(PARAM_COLS)):
                ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_title(f"Korelacje parametrów ({name})")
    fig.colorbar(im, ax=axes, fraction=0.03)
    save_fig(fig, out_dir / "01_catalog_param_corr.png")
    return corr_p


# ============================================================================
# ===================== 4. LINIE REFERENCYJNE NA SIATCE =======================
# ============================================================================

def report_lines_on_grid(wave: np.ndarray, lines=REFERENCE_LINES,
                         out_dir: Path = ASTRO_REPORTS_DIR) -> pd.DataFrame:
    """
    Dla każdej linii referencyjnej znajduje najbliższy piksel siatki:
    indeks, λ siatki, odchyłkę Δλ i odchyłkę w pikselach. Pomaga zdefiniować
    okna sygnatur w pliku 02 (lambda_min, lambda_max).
    """
    print("=" * 80)
    print("KROK 1.5: LINIE REFERENCYJNE NA SIATCE")
    print("=" * 80)
    out_dir = ensure_dir(out_dir)
    steps = np.gradient(wave)
    rows = []
    for name, lam in lines:
        inside = wave.min() <= lam <= wave.max()
        i = int(np.argmin(np.abs(wave - lam)))
        rows.append(dict(line=name, lambda_rest_A=lam, in_range=inside, grid_index=i,
                         grid_lambda_A=float(wave[i]), delta_lambda_A=float(wave[i] - lam),
                         delta_pixels=float((wave[i] - lam) / steps[i]),
                         local_step_A=float(steps[i])))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "01_reference_lines_on_grid.csv", index=False)
    write_report(out_dir / "01_reference_lines_on_grid.txt", "LINIE REFERENCYJNE NA SIATCE",
                 [("Tabela", ["  " + l for l in out.round(4).to_string(index=False).splitlines()])])
    return out


# ============================================================================
# ======================= 5. POMIAR REDSHIFTU =================================
# ============================================================================

def moving_average(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Średnia ruchoma z odbiciem brzegów (stała długość wyjścia)."""
    if window < 2:
        return y.copy()
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, np.ones(window) / window, mode="valid")


def estimate_line_center_robust(wave, flux, lambda_rest, z_guess=0.0, window_rest=12.0,
                                smooth_window=REDSHIFT_SMOOTH_WINDOW,
                                fit_half_width_pts=REDSHIFT_FIT_HALF_WIDTH) -> float:
    """
    Wyznacza środek linii absorpcyjnej: wygładzenie, minimum, dopasowanie
    paraboli do otoczenia minimum (subpikselowa precyzja wierzchołka).
    Zwraca λ_obs [Å] lub NaN.
    """
    lam_guess = lambda_rest * (1.0 + z_guess)
    half = window_rest * (1.0 + z_guess)
    m = (wave >= lam_guess - half) & (wave <= lam_guess + half)
    if m.sum() < 9:
        return np.nan
    w, f = wave[m], flux[m]
    ok = np.isfinite(w) & np.isfinite(f)
    w, f = w[ok], f[ok]
    if len(w) < 9:
        return np.nan
    fs = moving_average(f, smooth_window)
    i0 = int(np.argmin(fs))
    lo, hi = max(0, i0 - fit_half_width_pts), min(len(w), i0 + fit_half_width_pts + 1)
    if hi - lo < 5:
        return np.nan
    try:
        a, b, _ = np.polyfit(w[lo:hi], fs[lo:hi], 2)
    except Exception:
        return float(w[i0])
    if a <= 0:            # parabola musi mieć minimum
        return float(w[i0])
    xv = -b / (2 * a)
    return float(xv) if w[lo] <= xv <= w[hi - 1] else float(w[i0])


def measure_multiline_redshift(wave, flux, z_guess=0.0, lines=REDSHIFT_LINES,
                               outlier_threshold=REDSHIFT_OUTLIER_THRESHOLD):
    """
    Mierzy redshift z wielu linii; odrzuca linie, których z różni się od
    mediany o więcej niż outlier_threshold. Zwraca (z_mean, z_median, z_std,
    per_line: dict).
    """
    wmin, wmax = np.nanmin(wave), np.nanmax(wave)
    per_line, recs = {}, []
    for name, lam, win in lines:
        if not (wmin <= lam <= wmax):
            continue
        lo = estimate_line_center_robust(wave, flux, lam, z_guess, win)
        zl = (lo - lam) / lam if np.isfinite(lo) else np.nan
        per_line[f"lambda_obs_{name}"] = lo
        per_line[f"z_{name}"] = zl
        if np.isfinite(zl):
            recs.append((name, zl))
    per_line["n_lines_all"] = len(recs)
    if not recs:
        per_line["n_lines_kept"] = 0
        return np.nan, np.nan, np.nan, per_line
    zall = np.array([z for _, z in recs])
    zmed_all = float(np.median(zall))
    kept = [(n, z) for n, z in recs if abs(z - zmed_all) <= outlier_threshold]
    kept_names = {n for n, _ in kept}
    for n, _ in recs:
        per_line[f"keep_{n}"] = int(n in kept_names)
    per_line["n_lines_kept"] = len(kept)
    per_line["z_median_all"] = zmed_all
    if not kept:
        return np.nan, np.nan, np.nan, per_line
    zk = np.array([z for _, z in kept])
    return float(zk.mean()), float(np.median(zk)), float(zk.std(ddof=1)) if len(zk) > 1 else 0.0, per_line


def measure_redshifts_all(wave, X, z_guess: np.ndarray, prefix: str = "") -> pd.DataFrame:
    """Pomiar redshiftu dla wszystkich widm; zwraca tabelę (jeden wiersz na gwiazdę)."""
    rows = []
    for i in range(X.shape[0]):
        zg = float(z_guess[i]) if np.isfinite(z_guess[i]) else 0.0
        zm, zmed, zs, per = measure_multiline_redshift(wave, X[i], zg)
        row = {"row_id": i, f"{prefix}z_mean": zm, f"{prefix}z_median": zmed, f"{prefix}z_std": zs}
        row.update({f"{prefix}{k}": v for k, v in per.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def choose_redshift(df: pd.DataFrame, meas: pd.DataFrame, source: str = Z_SOURCE):
    """
    Wybiera z użyte do deredshiftu zgodnie z Z_SOURCE. Zwraca
    (z_used, z_source_label) jako tablice numpy.
    """
    z_cat = df["z"].to_numpy(dtype=float)
    z_mes = meas["meas_z_median"].to_numpy(dtype=float)
    if source == "catalog":
        z_used = z_cat.copy(); lab = np.where(np.isfinite(z_cat), "catalog", "none")
    elif source == "measured":
        z_used = z_mes.copy(); lab = np.where(np.isfinite(z_mes), "measured", "none")
    elif source == "measured_fallback_catalog":
        z_used = np.where(np.isfinite(z_mes), z_mes, z_cat)
        lab = np.where(np.isfinite(z_mes), "measured", np.where(np.isfinite(z_cat), "catalog", "none"))
    else:
        raise ValueError(f"Nieznane Z_SOURCE={source}")
    z_used = np.where(np.isfinite(z_used), z_used, 0.0)
    return z_used, lab


# ============================================================================
# ========================== 6. DEREDSHIFT ====================================
# ============================================================================

def deredshift_spectrum(wave: np.ndarray, flux_obs: np.ndarray, z: float) -> np.ndarray:
    """
    Przenosi widmo do układu spoczynkowego na TEJ SAMEJ siatce:
        F_rest(λ) = F_obs(λ·(1+z)),
    interpolacja liniowa, brzegi wypełnione skrajnymi wartościami.
    """
    if not np.isfinite(z):
        return np.full_like(flux_obs, np.nan, dtype=float)
    ok = np.isfinite(flux_obs)
    if ok.sum() < 5:
        return np.full_like(flux_obs, np.nan, dtype=float)
    w, f = wave[ok], flux_obs[ok]
    return np.interp(wave * (1.0 + z), w, f, left=f[0], right=f[-1])


def deredshift_all(wave: np.ndarray, X: np.ndarray, z_used: np.ndarray) -> np.ndarray:
    """Deredshift całej macierzy widm."""
    print("=" * 80)
    print("KROK 1.7: DEREDSHIFT WIDM")
    print("=" * 80)
    Xr = np.empty_like(X, dtype=np.float64)
    for i in range(X.shape[0]):
        Xr[i] = deredshift_spectrum(wave, X[i], z_used[i])
    print(f"✓ Przeliczono {X.shape[0]} widm do układu spoczynkowego")
    return Xr


def report_redshift(df: pd.DataFrame, meas: pd.DataFrame, resid: pd.DataFrame,
                    z_used: np.ndarray, z_lab: np.ndarray, wave: np.ndarray,
                    X: np.ndarray, X_dered: np.ndarray, out_dir: Path = ASTRO_REPORTS_DIR) -> pd.DataFrame:
    """
    Raport redshiftu: porównanie z katalogowego z pomiarem wieloliniowym,
    statystyki Δz, prędkości radialne, liczba użytych linii, redshift
    resztkowy po deredshifcie, przesunięcie w pikselach. Wykresy: z_kat vs
    z_pomiar, histogram Δz, histogram RV, redshift przed/po, box-plot z per
    linia, 9 przykładów widm (najmniejsze / ~0 / największe z).
    """
    print("=" * 80)
    print("KROK 1.8: RAPORT REDSHIFTU I WALIDACJA DEREDSHIFTU")
    print("=" * 80)
    out_dir = ensure_dir(out_dir)
    z_cat = df["z"].to_numpy(float)
    z_mes = meas["meas_z_median"].to_numpy(float)
    dz = z_mes - z_cat
    z_res = resid["resid_z_median"].to_numpy(float)
    step_ha = float(np.interp(6562.8, wave[1:], np.diff(wave)))
    shift_pix = z_used * 6562.8 / step_ha

    ok = np.isfinite(z_cat) & np.isfinite(z_mes)
    r_p, p_p = stats.pearsonr(z_cat[ok], z_mes[ok]) if ok.sum() > 3 else (np.nan, np.nan)

    line_names = [n for n, lam, _ in REDSHIFT_LINES if wave.min() <= lam <= wave.max()]
    sec_lines = []
    for n in line_names:
        zc = meas.get(f"meas_z_{n}")
        kp = meas.get(f"meas_keep_{n}")
        if zc is None:
            continue
        frac_keep = np.nanmean(kp) if kp is not None else np.nan
        sec_lines.append(f"  {n:<10s} med(z)={np.nanmedian(zc):+.3e}  std={np.nanstd(zc):.3e}  "
                         f"pomiarów={np.isfinite(zc).sum():5d}  zachowano={frac_keep:.1%}")

    write_report(out_dir / "01_redshift_report.txt", "RAPORT REDSHIFTU I DEREDSHIFTU", [
        ("Ustawienia", [f"  Źródło z do deredshiftu (Z_SOURCE): {Z_SOURCE}",
                        f"  Próg odrzucania linii: {REDSHIFT_OUTLIER_THRESHOLD:g}",
                        f"  Linie: {', '.join(line_names)}",
                        "  Źródła faktycznie użyte: " + ", ".join(f"{k}={v}" for k, v in
                                                            pd.Series(z_lab).value_counts().items())]),
        ("Redshift katalogowy i zmierzony", [
            describe_lines("z katalog", z_cat, "+.4e"),
            describe_lines("z pomiar (mediana)", z_mes, "+.4e"),
            describe_lines("Δz = pomiar - katalog", dz, "+.4e"),
            describe_lines("|Δz|", np.abs(dz), ".4e"),
            describe_lines("Δv = c·Δz [km/s]", dz * C_KMS, "+.4g"),
            f"  Korelacja Pearsona z_kat vs z_pomiar: r={r_p:.4f}, p={p_p:.3g}, n={ok.sum()}",
            f"  Udział widm z udanym pomiarem: {np.isfinite(z_mes).mean():.1%}",
            describe_lines("liczba linii zachowanych", meas["meas_n_lines_kept"], ".3g"),
        ]),
        ("Prędkości radialne i przesunięcie w pikselach", [
            describe_lines("RV użyte = c·z [km/s]", z_used * C_KMS, "+.4g"),
            describe_lines("przesunięcie Hα [piksele]", shift_pix, "+.4g"),
            f"  Gwiazd z |przesunięciem| > 1 piksel: {(np.abs(shift_pix) > 1).sum()}",
            f"  Gwiazd z |RV| > 100 km/s (kandydaci halo / szybkie): {(np.abs(z_used * C_KMS) > 100).sum()}",
        ]),
        ("Walidacja: redshift resztkowy po deredshifcie (oczekiwane ≈ 0)", [
            describe_lines("z resztkowe", z_res, "+.4e"),
            describe_lines("|z resztkowe|", np.abs(z_res), ".4e"),
            f"  Mediana |z_res| / mediana |z_użyte| = "
            f"{np.nanmedian(np.abs(z_res)) / max(np.nanmedian(np.abs(z_used)), 1e-30):.3f}",
        ]),
        ("Pomiar per linia (przed deredshiftem)", sec_lines),
    ])

    # --- wykres: z katalog vs z pomiar + Δz + RV ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    ax.scatter(z_cat * C_KMS, z_mes * C_KMS, s=6, alpha=0.5)
    lim = np.nanpercentile(np.abs(np.r_[z_cat, z_mes]) * C_KMS, 99.5)
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="y = x")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("RV katalog [km/s]"); ax.set_ylabel("RV z pomiaru wieloliniowego [km/s]")
    ax.set_title(f"Katalog vs pomiar (r={r_p:.3f})"); ax.legend(); ax.grid(alpha=.25)
    axes[1].hist((dz * C_KMS)[np.isfinite(dz)], bins=60, color="slateblue", edgecolor="k")
    axes[1].set_xlabel("Δv = c·(z_pomiar − z_katalog) [km/s]"); axes[1].set_title("Różnica pomiar − katalog")
    axes[1].grid(alpha=.25)
    axes[2].hist(z_used * C_KMS, bins=60, color="tomato", edgecolor="k")
    axes[2].set_xlabel("RV użyte do deredshiftu [km/s]"); axes[2].set_title("Rozkład prędkości radialnych")
    axes[2].grid(alpha=.25)
    fig.tight_layout()
    save_fig(fig, out_dir / "01_redshift_catalog_vs_measured.png")

    # --- wykres: przed / po deredshifcie ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(z_used * C_KMS, z_res * C_KMS, s=6, alpha=.5)
    axes[0].axhline(0, color="r", ls="--")
    axes[0].set_xlabel("RV użyte [km/s]"); axes[0].set_ylabel("RV resztkowe po deredshifcie [km/s]")
    axes[0].set_title("Redshift resztkowy vs użyty"); axes[0].grid(alpha=.25)
    bins = np.linspace(-lim, lim, 70)
    axes[1].hist(z_mes[np.isfinite(z_mes)] * C_KMS, bins=bins, alpha=.6, label="przed", color="steelblue")
    axes[1].hist(z_res[np.isfinite(z_res)] * C_KMS, bins=bins, alpha=.6, label="po", color="tomato")
    axes[1].axvline(0, color="k", ls="--"); axes[1].legend()
    axes[1].set_xlabel("zmierzone RV [km/s]"); axes[1].set_title("Pomiar przed i po deredshifcie")
    axes[1].grid(alpha=.25)
    fig.tight_layout()
    save_fig(fig, out_dir / "01_redshift_before_after.png")

    # --- wykres: box-plot z per linia ---
    data_bp, labs = [], []
    for n in line_names:
        c = f"meas_z_{n}"
        if c in meas.columns:
            v = meas[c].to_numpy(float)
            v = v[np.isfinite(v)] * C_KMS
            if len(v):
                data_bp.append(v); labs.append(n)
    if data_bp:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(data_bp, tick_labels=labs, showfliers=False)
        ax.axhline(0, color="gray", ls=":")
        ax.set_ylabel("RV z pojedynczej linii [km/s]")
        ax.set_title("Rozrzut prędkości wyznaczonych z poszczególnych linii")
        ax.grid(alpha=.25)
        save_fig(fig, out_dir / "01_redshift_per_line_boxplot.png")

    # --- wykres: 9 przykładów (3 najmniejsze z, 3 ~0, 3 największe) ---
    order = np.argsort(z_used)
    sel = list(order[:3]) + list(np.argsort(np.abs(z_used))[:3]) + list(order[-3:])
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    xm = (wave >= EXAMPLES_XLIM[0]) & (wave <= EXAMPLES_XLIM[1])
    for ax, i in zip(axes.flat, sel):
        ax.plot(wave[xm], X[i, xm], color="black", lw=0.9, label="obserwowane")
        ax.plot(wave[xm], X_dered[i, xm], color="red", lw=0.9, label="deredshift")
        ax.axvline(6562.8, color="gray", ls="--", lw=0.8)
        ax.set_title(f"row={i} | RV={z_used[i] * C_KMS:+.1f} km/s | {df.loc[i, 'subclass'] if 'subclass' in df else ''}",
                     fontsize=9)
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("λ [Å]")
    fig.suptitle("Przykłady deredshiftu wokół Hα (rzędy: najmniejsze z, z≈0, największe z)",
                 fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir / "01_deredshift_examples_9.png")

    table = pd.concat([df[["row_id"] + [c for c in ID_COLS if c in df.columns] + ["z"]],
                       meas.drop(columns="row_id"), resid.drop(columns="row_id")], axis=1)
    table["z_used"] = z_used
    table["z_used_source"] = z_lab
    table["rv_used_kms"] = z_used * C_KMS
    table["shift_halpha_pixels"] = shift_pix
    table.to_csv(out_dir / "01_redshift_table.csv", index=False)
    print(f"  ✓ tabela: {out_dir / '01_redshift_table.csv'}")
    return table


def plot_single_star_diagnostic(row_id: int, wave, X, X_dered, table: pd.DataFrame,
                                df: pd.DataFrame, out_dir: Path = ASTRO_REPORTS_DIR) -> None:
    """
    Diagnostyka jednej gwiazdy: pełne widmo przed/po, zbliżenia na linie
    pomiarowe z zaznaczoną pozycją spoczynkową i obserwowaną, raport TXT.
    """
    if not (0 <= row_id < X.shape[0]):
        print(f"  ⚠ row_id={row_id} poza zakresem – pomijam")
        return
    out_dir = ensure_dir(out_dir / "single_stars")
    r = table.loc[row_id]
    used = [(n, lam, w) for n, lam, w in REDSHIFT_LINES if wave.min() <= lam <= wave.max()]
    fig = plt.figure(figsize=(16, 10))
    ax0 = fig.add_subplot(3, 1, 1)
    ax0.plot(wave, X[row_id], color="k", lw=.6, label="obserwowane")
    ax0.plot(wave, X_dered[row_id], color="r", lw=.6, alpha=.7, label="deredshift")
    ax0.set_title(f"row={row_id} | {df.loc[row_id, 'designation'] if 'designation' in df else ''} | "
                  f"Teff={df.loc[row_id, 'teff']:.0f} logg={df.loc[row_id, 'logg']:.2f} "
                  f"[Fe/H]={df.loc[row_id, 'feh']:+.2f} [α/M]={df.loc[row_id, 'alpha_m']:+.2f} | "
                  f"RV={r['rv_used_kms']:+.1f} km/s")
    ax0.legend(fontsize=8); ax0.grid(alpha=.25)
    for k, (n, lam, w) in enumerate(used[:8]):
        ax = fig.add_subplot(3, 4, 5 + k)
        m = (wave > lam - 2.5 * w) & (wave < lam + 2.5 * w)
        ax.plot(wave[m], X[row_id, m], "k-", lw=.9)
        ax.plot(wave[m], X_dered[row_id, m], "r-", lw=.9)
        ax.axvline(lam, color="gray", ls="--", lw=.8)
        lo = r.get(f"meas_lambda_obs_{n}", np.nan)
        if np.isfinite(lo):
            ax.axvline(lo, color="blue", ls=":", lw=1)
        ax.set_title(n, fontsize=9); ax.grid(alpha=.25)
    fig.tight_layout()
    save_fig(fig, out_dir / f"star_{row_id}_diagnostic.png")
    lines = [f"  {k:<32s} = {v}" for k, v in r.items()]
    lines += [f"  {c:<32s} = {df.loc[row_id, c]}" for c in df.columns if c not in r.index]
    write_report(out_dir / f"star_{row_id}_diagnostic.txt", f"DIAGNOSTYKA GWIAZDY row_id={row_id}",
                 [("Wszystkie wartości", lines)])


# ============================================================================
# ============================ 7. ZAPIS ======================================
# ============================================================================

def save_outputs(wave, X_dered, df: pd.DataFrame, table: pd.DataFrame,
                 npz_path: Path = DERED_NPZ_PATH, csv_path: Path = CATALOG_OUT_CSV) -> pd.DataFrame:
    """
    Zapisuje: (1) widma deredshiftowane NPZ (klucze X, wave_grid, z_used),
    (2) katalog z dołączonymi kolumnami redshiftu (CSV) – wejście pliku 02.
    """
    print("=" * 80)
    print("KROK 1.9: ZAPIS DANYCH DLA PLIKU 02")
    print("=" * 80)
    ensure_dir(Path(npz_path).parent)
    np.savez_compressed(npz_path, X=X_dered.astype(np.float32), wave_grid=wave.astype(np.float32),
                        z_used=table["z_used"].to_numpy(np.float64))
    cat = df.copy()
    for c in ["meas_z_median", "meas_z_std", "meas_n_lines_kept", "resid_z_median",
              "z_used", "z_used_source", "rv_used_kms", "shift_halpha_pixels"]:
        cat[c] = table[c].values
    cat.to_csv(csv_path, index=False)
    print(f"  ✓ {npz_path}")
    print(f"  ✓ {csv_path}")
    return cat


# ============================================================================
# ======================= URUCHOMIENIE PIPELINE'U =============================
# ============================================================================

def run_pipeline():
    ensure_dir(ASTRO_REPORTS_DIR)
    wave, X, keys = load_raw_spectra(RAW_NPZ_PATH)
    df = load_catalog(CATALOG_CSV_PATH, n_spectra=X.shape[0])
    report_spectra_quality(wave, X, df, tag="raw")
    report_catalog_parameters(df)
    report_lines_on_grid(wave)

    print("=" * 80)
    print("KROK 1.6: POMIAR REDSHIFTU Z WIELU LINII")
    print("=" * 80)
    meas = measure_redshifts_all(wave, X, df["z"].to_numpy(float), prefix="meas_")
    z_used, z_lab = choose_redshift(df, meas, Z_SOURCE)
    X_dered = deredshift_all(wave, X, z_used)
    resid = measure_redshifts_all(wave, X_dered, np.zeros(len(df)), prefix="resid_")

    table = report_redshift(df, meas, resid, z_used, z_lab, wave, X, X_dered)
    report_spectra_quality(wave, X_dered, df, tag="deredshifted")
    for rid in SINGLE_STAR_ROW_IDS:
        plot_single_star_diagnostic(rid, wave, X, X_dered, table, df)
    save_outputs(wave, X_dered, df, table)
    print("\n✓ KONIEC KROKU 1. Raporty w:", ASTRO_REPORTS_DIR.resolve())
    return dict(wave=wave, X=X, X_dered=X_dered, catalog=df, redshift_table=table)


if __name__ == "__main__":
    results_01 = run_pipeline()
