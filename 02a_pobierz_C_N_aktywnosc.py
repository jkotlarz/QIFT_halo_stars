# -*- coding: utf-8 -*-
"""
02a_pobierz_C_N_aktywnosc.py
============================
Dociąga brakujące dane do latentów węgla, azotu i aktywności chromosferycznej
i uzupełnia nimi plik referencyjny gwiazdy_qift_kinematyka.csv.

Uruchamiać PO kroku 01 i PRZED krokiem 02:
    python 02a_pobierz_C_N_aktywnosc.py

Co robi
-------
1. [C/Fe], [N/Fe] z katalogu DD-Payne dla LAMOST DR5 (Xiang i in. 2019, ApJS 245, 34;
   VizieR J/ApJS/245/34). Dopasowanie przez serwis CDS XMatch (promień 3″);
   jeśli w katalogu jest kolumna obsid, wybierany jest wiersz z tym samym obsid co nasze
   widmo, w przeciwnym razie najbliższy. Opcjonalnie: lokalne pliki z innymi katalogami
   (np. katalog wartości dodanych LAMOST DR8, Wang i in. 2022, ApJS 259, 51) – patrz
   LOCAL_CN_FILES. Nazwy kolumn są rozpoznawane automatycznie i wypisywane w raporcie.
2. Indeks aktywności S_L (Ca II H i K) z bazy LAMOST DR7 LRS (Zhang i in. 2022, ApJS 263, 12;
   Zenodo 10.5281/zenodo.7067044), dopasowanie po obsid. Plik CSV (~kilkaset MB) jest
   pobierany raz do katalogu dane_zewnetrzne/ i filtrowany fragmentami.
3. WŁASNE indeksy z ORYGINALNYCH plików FITS LAMOST (katalog widma/, natywne próbkowanie
   ~0,9–2 Å/piksel, strumień fizyczny, przesunięcie do układu spoczynkowego z_used z kroku 01),
   w oknach w PRÓŻNI uzgodnionych do latentów: S (Ca II H+K), rdzeń Hα, rdzenie Ca II IRT,
   pasmo G (CH), CN 3883, CN 4215. Działa bez sieci – to także zapas, gdy katalogi są
   niedostępne, i kontrola katalogów.
   UWAGA: widma w spectra_deredshifted.npz są standaryzowane (z-score każdego widma), więc
   ilorazy strumieni (indeksy) policzone na nich nie mają sensu fizycznego – dlatego FITS.
4. Nadwyżka aktywności ponad fotosferę: log10(indeks / poziom bazowy), gdzie poziom bazowy
   to 10. percentyl indeksu wśród 30 najbliższych karłów (log g ≥ 3.5) w przestrzeni (Teff, [Fe/H])
   – zależność od metaliczności usunięta (wersja z 24.10).
5. Kotwice dla kroku 02 (kotwica_C, kotwica_N, kotwica_akt) z kontrolą jakości.

Wyjście
-------
  astro_reports/kotwice_C_N_aktywnosc.csv   – jedna linia na WIDMO (obsid; 1729 wierszy),
                                              czyta to krok 02
  gwiazdy_qift_kinematyka.csv               – uzupełniony (jedna linia na GWIAZDĘ);
                                              kopia oryginału: *.przed_02a.csv
  astro_reports/02a_raport_C_N_aktywnosc.txt, 02a_*.png

Wymagania: numpy, pandas, matplotlib, astropy, requests (sieć tylko dla kroków 1–2).
"""

from __future__ import annotations

import io
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# ======================= PARAMETRY URUCHOMIENIOWE ===========================
# ============================================================================

REF_CSV = Path("gwiazdy_qift_kinematyka.csv")
ASTRO_REPORTS_DIR = Path("astro_reports")
CATALOG_CSV = ASTRO_REPORTS_DIR / "catalog_with_redshift.csv"
SPECTRA_DIR = Path("C:/Users/User/Documents/Publikacje/005_abundancje/widma")            # oryginalne pliki LAMOST spec-obsid-<obsid>.fits
FITS_NAME = "spec-obsid-{obsid}.fits"
ANCHORS_OUT = ASTRO_REPORTS_DIR / "kotwice_C_N_aktywnosc.csv"
CACHE_DIR = Path("dane_zewnetrzne")

# --- które źródła włączyć ---
USE_VIZIER_CN = True            # Xiang i in. 2019 przez CDS XMatch
USE_ZENODO_ACTIVITY = True      # Zhang i in. 2022 (S_L)
LOCAL_CN_FILES: list[Path] = [] # np. [Path("dane_zewnetrzne/LAMOST_DR8_VAC.csv")] – kolumny wykrywane automatycznie
LOCAL_ACTIVITY_FILE: Path | None = None  # jeśli plik Zhang i in. pobrano ręcznie

# --- katalogi zewnętrzne ---
VIZIER_CN_CATALOG = "J/ApJS/245/34"
VIZIER_CN_TABLE: str | None = "J/ApJS/245/34/catalog"   # tabela VizieR (None = próba wykrycia przez TAP)
TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
XMATCH_URL = "https://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync"
XMATCH_RADIUS_ARCSEC = 3.0
ZENODO_RECORD = "7067044"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
HTTP_TIMEOUT = 600

# --- kontrola jakości kotwic ---
CN_MAX_ERR = 0.20               # maks. błąd [C/Fe] i [N/Fe] [dex]
CN_TEFF_RANGE = (3800.0, 7000.0)
CN_FLAG_GOOD = "1"              # wartość flagi q_[C/Fe], q_[N/Fe] oznaczająca wartość wiarygodną (Xiang+2019)
CN_MULT_EXCLUDE = "NO"          # Mult = "NO" -> możliwa gwiazda podwójna/wielokrotna: bez kotwicy C, N
                                # (w katalogu 90% widm ma "YES"; jeśli znaczenie jest odwrotne, zmień na "YES" lub None)
ACT_ONLY_DWARFS = True          # kotwica aktywności tylko dla log g ≥ ACT_LOGG_MIN
ACT_LOGG_MIN = 3.5
BASAL_PERCENTILE = 10.0         # poziom bazowy aktywności: percentyl indeksu wśród podobnych karłów
BASAL_K_NEIGHBORS = 30          # liczba najbliższych karłów w przestrzeni (Teff, [Fe/H])
BASAL_TEFF_SCALE = 150.0        # [K]   – skala odległości w Teff
BASAL_FEH_SCALE = 0.20          # [dex] – skala odległości w [Fe/H] (poziom bazowy zależy od metaliczności)
BASAL_MIN_STARS = 15

# --- okna indeksów (PRÓŻNIA, Å), zgodne z oknami latentów w kroku 02 ---
INDEX_DEFS = {
    # aktywność
    "CaK":    dict(core=[(3933.28, 3936.28)]),
    "CaH":    dict(core=[(3968.09, 3971.09)]),   # blend z Hε 3971.2 – pomocniczo
    "HK_V":   dict(core=[(3892.17, 3912.17)]),
    "HK_R":   dict(core=[(3992.20, 4012.20)]),
    "Ha_core": dict(core=[(6563.11, 6566.11)], cont=[(6545.0, 6555.0), (6575.0, 6585.0)]),
    "IRT":    dict(core=[(8498.9, 8501.9), (8542.9, 8545.9), (8663.0, 8666.0)],
                   cont=[(8475.0, 8485.0), (8562.0, 8572.0), (8680.0, 8690.0)]),
    # węgiel i azot
    "G4300":  dict(band=(4282.6, 4317.6), cont=[(4267.6, 4283.8), (4320.1, 4336.3)]),
    "CN3883": dict(band=(3847.1, 3884.1), cont=[(3884.1, 3917.1)]),
    "CN4215": dict(band=(4143.3, 4178.3), cont=[(4085.0, 4097.5), (4245.3, 4285.3)]),
}

DPI = 160


# ============================================================================
# ========================= FUNKCJE POMOCNICZE ================================
# ============================================================================

def log(msg=""):
    print(msg, flush=True)


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_col(cols, want: str, err: bool = False):
    """Rozpoznaje kolumnę [X/Fe] lub jej błąd po znormalizowanej nazwie (np. '[C/Fe]', 'C_FE', 'cfe', 'e_[C/Fe]')."""
    for c in cols:
        n = norm_name(c)
        if not err and n == want:
            return c
        if err and n in (f"e{want}", f"{want}err", f"err{want}", f"{want}error", f"sig{want}"):
            return c
    return None


def find_obsid_col(cols):
    for c in cols:
        if norm_name(c) in ("obsid", "lamostobsid", "obsidlamost"):
            return c
    return None


def write_report(path: Path, title: str, sections):
    lines = ["=" * 80, title, f"Wygenerowano: {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 80, ""]
    for h, body in sections:
        lines += [h, "-" * len(h)] + list(body) + [""]
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"  ✓ raport: {path}")


def http():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "qift-halo-pipeline/02a"
    return s


# ============================================================================
# ===================== 1. WCZYTANIE DANYCH LOKALNYCH =========================
# ============================================================================

def load_local():
    log("=" * 80); log("KROK 2a.1: DANE LOKALNE"); log("=" * 80)
    ref = pd.read_csv(REF_CSV, dtype={"gaia_dr3_source_id": str})
    cat = pd.read_csv(CATALOG_CSV)
    log(f"  ✓ {REF_CSV}: {len(ref)} gwiazd; {CATALOG_CSV}: {len(cat)} widm")
    return ref, cat


# ============================================================================
# ===================== 2. WŁASNE INDEKSY Z WIDM ==============================
# ============================================================================

def read_lamost_fits(path: Path):
    """Zwraca (λ_próżnia, strumień, ivar) z pliku LAMOST LRS (format DR1–5: obraz; DR6+: tabela)."""
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        for hdu in h[1:]:
            if isinstance(hdu, fits.BinTableHDU) and "FLUX" in [c.upper() for c in hdu.columns.names]:
                d = hdu.data
                return (np.asarray(d["WAVELENGTH"][0], float), np.asarray(d["FLUX"][0], float),
                        np.asarray(d["IVAR"][0], float))
        data, hdr = h[0].data, h[0].header
        if data is None or data.ndim != 2:
            raise ValueError(f"nieznany format FITS: {path}")
        if data.shape[0] >= 3:
            wave = np.asarray(data[2], float)
        else:
            wave = 10 ** (hdr["COEFF0"] + hdr["COEFF1"] * np.arange(data.shape[1]))
        return wave, np.asarray(data[0], float), np.asarray(data[1], float)


def _mean_in(w, f, ranges):
    m = np.zeros(len(w), bool)
    for a, b in ranges:
        m |= (w >= a) & (w <= b)
    return (np.nanmean(f[m]) if m.sum() else np.nan), int(m.sum())


def indices_one(w, f) -> tuple[dict, dict]:
    """Indeksy jednej gwiazdy (λ w układzie spoczynkowym, próżnia)."""
    o, px = {}, {}
    v = {}
    for k in ["CaK", "CaH", "HK_V", "HK_R"]:
        v[k], px[k] = _mean_in(w, f, INDEX_DEFS[k]["core"])
    o["idx_S_wlasny"] = (v["CaK"] + v["CaH"]) / (v["HK_V"] + v["HK_R"])
    o["idx_K_wlasny"] = v["CaK"] / (0.5 * (v["HK_V"] + v["HK_R"]))
    for k in ["Ha_core", "IRT"]:
        core, px[k] = _mean_in(w, f, INDEX_DEFS[k]["core"])
        cont, _ = _mean_in(w, f, INDEX_DEFS[k]["cont"])
        o[f"idx_{k}"] = core / cont
    for k in ["G4300", "CN4215"]:
        a, b = INDEX_DEFS[k]["band"]; m = (w >= a) & (w <= b); px[k] = int(m.sum())
        pts = [(np.mean(c), _mean_in(w, f, [c])[0]) for c in INDEX_DEFS[k]["cont"]]
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        fc = y1 + (y2 - y1) * (w[m] - x1) / (x2 - x1)
        ratio = f[m] / fc
        if k == "G4300":
            o["idx_G4300_EW"] = float(np.nansum((1 - ratio) * np.gradient(w)[m]))      # [Å]
        else:
            o["idx_CN4215_mag"] = -2.5 * np.log10(max(np.nanmean(ratio), 1e-6))
    band, px["CN3883"] = _mean_in(w, f, [INDEX_DEFS["CN3883"]["band"]])
    cont, _ = _mean_in(w, f, INDEX_DEFS["CN3883"]["cont"])
    o["idx_CN3883_S3839"] = -2.5 * np.log10(max(band / cont, 1e-6)) if np.isfinite(band / cont) else np.nan
    return o, px


IDX_COLS = ["idx_S_wlasny", "idx_K_wlasny", "idx_Ha_core", "idx_IRT", "idx_G4300_EW",
            "idx_CN4215_mag", "idx_CN3883_S3839"]


def measure_indices(cat: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log("=" * 80); log("KROK 2a.2: WŁASNE INDEKSY Z PLIKÓW FITS (okna w próżni)"); log("=" * 80)
    rows, npx, missing, bad = [], [], 0, 0
    z = cat["z_used"].to_numpy(float) if "z_used" in cat else cat["z"].to_numpy(float)
    for i, obsid in enumerate(cat["obsid"].astype(np.int64)):
        path = SPECTRA_DIR / FITS_NAME.format(obsid=obsid)
        if not path.exists():
            missing += 1; rows.append({k: np.nan for k in IDX_COLS}); continue
        try:
            w, f, iv = read_lamost_fits(path)
            f = np.where((iv > 0) & np.isfinite(f), f, np.nan)
            o, px = indices_one(w / (1.0 + z[i]), f)
            rows.append(o); npx.append(px)
        except Exception as e:
            bad += 1; rows.append({k: np.nan for k in IDX_COLS})
            if bad <= 5:
                log(f"  ⚠ {path.name}: {e}")
        if (i + 1) % 300 == 0:
            log(f"    {i + 1}/{len(cat)}")
    df = pd.DataFrame(rows, columns=IDX_COLS)
    notes = [f"  katalog widm: {SPECTRA_DIR.resolve()}",
             f"  zmierzono {len(cat) - missing - bad} widm, brak pliku {missing}, błąd odczytu {bad}"]
    if npx:
        med = pd.DataFrame(npx).median()
        notes += [f"  {k:<8s}: mediana {v:.0f} pikseli w rdzeniu/paśmie" + ("   ⚠ < 2 piksele" if v < 2 else "")
                  for k, v in med.items()]
    if missing == len(cat):
        notes.append("  ⚠ nie znaleziono żadnego pliku FITS – indeksy własne pominięte (ustaw SPECTRA_DIR)")
    log(f"  ✓ indeksy: {len(cat) - missing - bad} widm (brak plików: {missing}, błędy: {bad})")
    return df, notes


def basal_excess(index: np.ndarray, teff: np.ndarray, logg: np.ndarray, feh: np.ndarray):
    """
    log10(indeks / poziom bazowy), gdzie poziom bazowy = BASAL_PERCENTILE-ty percentyl indeksu wśród
    BASAL_K_NEIGHBORS najbliższych karłów w przestrzeni (Teff/150 K, [Fe/H]/0,2 dex).
    Uwzględnienie [Fe/H] usuwa fotosferyczny wpływ metaliczności (słabsza absorpcja Ca II H, K
    u gwiazd ubogich w metale wygląda jak „wypełnienie” rdzenia).
    """
    from scipy.spatial import cKDTree
    ok = np.isfinite(index) & np.isfinite(teff) & np.isfinite(feh) & (index > 0)
    dw = ok & (logg >= ACT_LOGG_MIN)
    if dw.sum() < max(3 * BASAL_MIN_STARS, BASAL_K_NEIGHBORS):
        return np.full(len(index), np.nan), np.nan
    Z = np.c_[teff / BASAL_TEFF_SCALE, feh / BASAL_FEH_SCALE]
    tree = cKDTree(Z[dw]); ref = index[dw]
    base = np.full(len(index), np.nan)
    _, nn = tree.query(Z[ok], k=BASAL_K_NEIGHBORS)
    base[ok] = np.percentile(ref[nn], BASAL_PERCENTILE, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        exc = np.where(ok & (base > 0), np.log10(index / base), np.nan)
    o = np.argsort(teff[dw])
    return exc, (teff[dw][o], base[dw][o])


# ============================================================================
# ============== 3. [C/Fe], [N/Fe]: VizieR (XMatch) i pliki lokalne ===========
# ============================================================================

def tap_query(sess, adql: str) -> pd.DataFrame:
    r = sess.post(TAP_URL, data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=adql),
                  timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def discover_vizier_table(sess) -> tuple[str, list[str]]:
    if VIZIER_CN_TABLE:                      # nazwa podana – kolumny rozpoznawane po wyniku XMatch
        return VIZIER_CN_TABLE, []
    tabs = tap_query(sess, f"SELECT table_name FROM tap_schema.tables WHERE table_name LIKE '%{VIZIER_CN_CATALOG}%'")
    log(f"    tabele TAP: {tabs.iloc[:, 0].tolist()}")
    best = None
    for t in tabs.iloc[:, 0]:
        cols = tap_query(sess, f"SELECT column_name FROM tap_schema.columns WHERE table_name = '{t}'").iloc[:, 0].tolist()
        log(f"    tabela {t}: {len(cols)} kolumn")
        if find_col(cols, "cfe") and find_col(cols, "nfe"):
            best = (t, cols)
    if best is None:
        raise RuntimeError(f"W {VIZIER_CN_CATALOG} nie znaleziono tabeli z [C/Fe] i [N/Fe] – ustaw VIZIER_CN_TABLE")
    return best


def xmatch_vizier(sess, pos: pd.DataFrame, table: str) -> pd.DataFrame:
    buf = io.StringIO(); pos[["key", "ra", "dec"]].to_csv(buf, index=False)
    r = sess.post(XMATCH_URL, data=dict(request="xmatch", distMaxArcsec=XMATCH_RADIUS_ARCSEC,
                                        RESPONSEFORMAT="csv", cat2=f"vizier:{table}",
                                        colRA1="ra", colDec1="dec"),
                  files=dict(cat1=("pozycje.csv", buf.getvalue())), timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def standardize_cn(raw: pd.DataFrame, source: str) -> tuple[pd.DataFrame, list[str]]:
    """Ujednolica kolumny: obsid_kat, c_fe, c_fe_err, n_fe, n_fe_err, flagi (tekst)."""
    cols = list(raw.columns)
    m = dict(c_fe=find_col(cols, "cfe"), c_fe_err=find_col(cols, "cfe", err=True),
             n_fe=find_col(cols, "nfe"), n_fe_err=find_col(cols, "nfe", err=True),
             obsid_kat=find_obsid_col(cols))
    flags = [c for c in cols if "flag" in c.lower() or norm_name(c) in ("qcfe", "qnfe", "qchi2", "mult", "uq")]
    notes = [f"  {source}: rozpoznane kolumny " + ", ".join(f"{k}←{v}" for k, v in m.items()),
             f"  {source}: kolumny flag: {flags or 'brak'}"]
    if m["c_fe"] is None or m["n_fe"] is None:
        raise RuntimeError(f"{source}: brak kolumn [C/Fe] lub [N/Fe]; dostępne: {cols[:60]}")
    out = pd.DataFrame({k: pd.to_numeric(raw[v], errors="coerce") if v else np.nan for k, v in m.items()})
    out["cn_flagi"] = raw[flags].astype(str).agg(";".join, axis=1) if flags else ""
    for c in flags:                                   # flagi także jako osobne kolumny: flaga_qcfe, flaga_qnfe, flaga_mult
        out[f"flaga_{norm_name(c)}"] = raw[c].astype(str).str.strip().to_numpy()
    for c in ["c_fe", "n_fe", "c_fe_err", "n_fe_err"]:
        out.loc[out[c] <= -9, c] = np.nan            # wartości brakujące typu -9999
    return out, notes


def cn_from_vizier(sess, cat: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log("  VizieR: wykrywanie tabeli ...")
    table, cols = discover_vizier_table(sess)
    log(f"  ✓ tabela {table}")
    pos = pd.DataFrame(dict(key=cat["obsid"].astype(np.int64), ra=cat["ra"], dec=cat["dec"]))
    log(f"  CDS XMatch: {len(pos)} pozycji, promień {XMATCH_RADIUS_ARCSEC}″ ...")
    xm = xmatch_vizier(sess, pos, table)
    log(f"  ✓ {len(xm)} dopasowań")
    std, notes = standardize_cn(xm, f"VizieR {table}")
    std["obsid"] = pd.to_numeric(xm["key"], errors="coerce").astype("Int64")
    std["cn_sep_arcsec"] = pd.to_numeric(xm.get("angDist"), errors="coerce")
    std["cn_zrodlo"] = f"Xiang+2019 ({table})"
    return std, [f"  tabela VizieR: {table}"] + notes


def cn_from_local(path: Path, cat: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Plik lokalny (CSV lub FITS) z obsid i [C/Fe], [N/Fe]; dopasowanie po obsid."""
    if path.suffix.lower() in (".fits", ".fit"):
        from astropy.table import Table
        raw = Table.read(path).to_pandas()
    else:
        raw = pd.read_csv(path, low_memory=False)
    std, notes = standardize_cn(raw, str(path))
    if std["obsid_kat"].isna().all():
        raise RuntimeError(f"{path}: brak kolumny obsid – dopasowanie pozycyjne dla plików lokalnych nie jest obsługiwane")
    std = std[std["obsid_kat"].isin(cat["obsid"])].copy()
    std["obsid"] = std["obsid_kat"].astype("Int64"); std["cn_sep_arcsec"] = 0.0
    std["cn_zrodlo"] = path.name
    return std, notes


def pick_best_cn(cands: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """Jeden wiersz na nasze widmo: najpierw zgodny obsid, potem najmniejszy błąd, potem najbliższy."""
    if cands.empty:
        return pd.DataFrame(columns=["obsid"])
    c = cands.copy()
    c["_same"] = (pd.to_numeric(c["obsid_kat"], errors="coerce") == pd.to_numeric(c["obsid"], errors="coerce")).fillna(False).astype(int)
    c["_err"] = c[["c_fe_err", "n_fe_err"]].max(axis=1).fillna(9.0)
    # ta sama gwiazda = dopasowania nie dalej niż 0,5″ od najbliższego (inne widma tej samej gwiazdy w DR5);
    # spośród nich: zgodny obsid, potem najmniejszy błąd
    sep = c["cn_sep_arcsec"].fillna(0.0)
    c = c[sep <= sep.groupby(c["obsid"]).transform("min") + 0.5]
    c = c.sort_values(["obsid", "_same", "_err", "cn_sep_arcsec"], ascending=[True, False, True, True])
    best = c.drop_duplicates("obsid", keep="first").copy()
    best["cn_dopasowanie"] = np.where(best["_same"] == 1, "obsid", "pozycja")
    return best.drop(columns=["_same", "_err"])


# ============================================================================
# ================ 4. AKTYWNOŚĆ: baza Zhang i in. (Zenodo) ====================
# ============================================================================

def zenodo_activity_file(sess) -> Path:
    if LOCAL_ACTIVITY_FILE is not None:
        return Path(LOCAL_ACTIVITY_FILE)
    CACHE_DIR.mkdir(exist_ok=True)
    rec = sess.get(ZENODO_API, timeout=HTTP_TIMEOUT).json()
    files = rec.get("files", [])
    cand = [f for f in files if str(f.get("key", "")).lower().endswith((".csv", ".csv.gz", ".zip"))]
    if not cand:
        raise RuntimeError(f"Zenodo {ZENODO_RECORD}: brak pliku CSV; pliki: {[f.get('key') for f in files]}")
    f = sorted(cand, key=lambda f: ("sindex" not in f["key"].lower(), f["key"]))[0]
    dest = CACHE_DIR / f["key"]
    if dest.exists() and dest.stat().st_size == f.get("size", dest.stat().st_size):
        log(f"  ✓ plik w pamięci podręcznej: {dest}")
        return dest
    url = f["links"].get("self") or f["links"].get("download")
    log(f"  pobieram {f['key']} ({f.get('size', 0) / 1e6:.0f} MB) ...")
    with sess.get(url, stream=True, timeout=HTTP_TIMEOUT) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.rename(dest)
    log(f"  ✓ {dest}")
    return dest


def activity_from_file(path: Path, obsids: set[int]) -> tuple[pd.DataFrame, list[str]]:
    rows, cols_used = [], None
    reader = pd.read_csv(path, chunksize=200_000, low_memory=False)
    for chunk in reader:
        if cols_used is None:
            oc = find_obsid_col(chunk.columns)
            want = {norm_name(c): c for c in chunk.columns}
            pick = {k: want.get(k) for k in ["sl", "slerr", "stri", "strierr", "srec", "srecerr", "snrg"]}
            cols_used = (oc, pick)
            if oc is None or pick["sl"] is None:
                raise RuntimeError(f"{path}: nie rozpoznano kolumn obsid / S_L; kolumny: {list(chunk.columns)}")
        oc, pick = cols_used
        sel = chunk[pd.to_numeric(chunk[oc], errors="coerce").isin(obsids)]
        if len(sel):
            rows.append(sel)
    if not rows:
        return pd.DataFrame(columns=["obsid"]), [f"  {path.name}: brak wspólnych obsid"]
    raw = pd.concat(rows)
    oc, pick = cols_used
    out = pd.DataFrame({"obsid": pd.to_numeric(raw[oc], errors="coerce").astype("Int64")})
    names = dict(sl="S_L", slerr="S_L_err", stri="S_tri", strierr="S_tri_err", srec="S_rec", srecerr="S_rec_err", snrg="akt_snrg")
    for k, v in pick.items():
        if v is not None:
            out[names[k]] = pd.to_numeric(raw[v], errors="coerce").to_numpy()
    out = out.sort_values("akt_snrg" if "akt_snrg" in out else "obsid", ascending=False).drop_duplicates("obsid", keep="first")
    out["akt_zrodlo"] = f"Zhang+2022 ({path.name})"
    return out, [f"  {path.name}: kolumny " + ", ".join(f"{names[k]}←{v}" for k, v in pick.items() if v),
                 f"  {path.name}: {len(out)} wspólnych widm"]


# ============================================================================
# ======================== 5. KOTWICE I ZAPIS =================================
# ============================================================================

def build_anchors(cat, idx, cn, act) -> tuple[pd.DataFrame, dict]:
    a = pd.DataFrame({"obsid": cat["obsid"].astype(np.int64), "teff": cat["teff"], "logg": cat["logg"],
                      "feh": cat["feh"], "snrg": cat.get("snrg")})
    a = pd.concat([a, idx.reset_index(drop=True)], axis=1)
    if len(cn):
        cn = cn.copy(); cn["obsid"] = cn["obsid"].astype(np.int64)
        a = a.merge(cn.drop(columns=["obsid_kat"], errors="ignore"), on="obsid", how="left")
    for c in ["c_fe", "c_fe_err", "n_fe", "n_fe_err", "cn_sep_arcsec"]:
        if c not in a:
            a[c] = np.nan
    for c in ["cn_zrodlo", "cn_dopasowanie", "cn_flagi"]:
        if c not in a:
            a[c] = ""
    if len(act):
        act = act.copy(); act["obsid"] = act["obsid"].astype(np.int64)
        a = a.merge(act, on="obsid", how="left")
    for c in ["S_L", "S_L_err", "akt_zrodlo"]:
        if c not in a:
            a[c] = np.nan if c != "akt_zrodlo" else ""
    a["c_n"] = a["c_fe"] - a["n_fe"]

    teff, logg = a["teff"].to_numpy(float), a["logg"].to_numpy(float)
    basal = {}
    for col, name in [("idx_S_wlasny", "nadw_S_wlasny"), ("idx_K_wlasny", "nadw_K_wlasny"),
                      ("idx_Ha_core", "nadw_Ha_core"), ("idx_IRT", "nadw_IRT"), ("S_L", "nadw_S_L")]:
        a[name], basal[name] = basal_excess(a[col].to_numpy(float), teff, logg, a["feh"].to_numpy(float))

    # kotwice z kontrolą jakości
    base_ok = a["teff"].between(*CN_TEFF_RANGE)
    if CN_MULT_EXCLUDE is not None and "flaga_mult" in a:
        base_ok &= a["flaga_mult"].astype(str) != CN_MULT_EXCLUDE
    ok_c = base_ok & (a["c_fe_err"].fillna(0) <= CN_MAX_ERR)
    ok_n = base_ok & (a["n_fe_err"].fillna(0) <= CN_MAX_ERR)
    if "flaga_qcfe" in a:
        ok_c &= a["flaga_qcfe"].astype(str) == CN_FLAG_GOOD
    if "flaga_qnfe" in a:
        ok_n &= a["flaga_qnfe"].astype(str) == CN_FLAG_GOOD
    a["kotwica_C"] = np.where(ok_c, a["c_fe"], np.nan)
    a["kotwica_N"] = np.where(ok_n, a["n_fe"], np.nan)
    akt = a["nadw_S_L"].where(a["nadw_S_L"].notna(), a["nadw_S_wlasny"])
    a["kotwica_akt_zrodlo"] = np.where(a["nadw_S_L"].notna(), "S_L (Zhang+2022)",
                                       np.where(a["nadw_S_wlasny"].notna(), "S własny", ""))
    if ACT_ONLY_DWARFS:
        akt = akt.where(a["logg"] >= ACT_LOGG_MIN)
        a.loc[a["logg"] < ACT_LOGG_MIN, "kotwica_akt_zrodlo"] = "pominięto (olbrzym)"
    a["kotwica_akt"] = akt
    return a, basal


def update_reference(ref: pd.DataFrame, anch: pd.DataFrame) -> pd.DataFrame:
    """Jedna linia na gwiazdę: wartości z widma użytego (obsid_uzyte); braki uzupełnione z innych widm gwiazdy."""
    keep = ["c_fe", "c_fe_err", "n_fe", "n_fe_err", "c_n", "cn_zrodlo", "cn_dopasowanie", "cn_sep_arcsec",
            "S_L", "S_L_err", "akt_zrodlo", "nadw_S_L",
            "idx_S_wlasny", "idx_K_wlasny", "idx_Ha_core", "idx_IRT", "idx_G4300_EW", "idx_CN3883_S3839",
            "idx_CN4215_mag", "nadw_S_wlasny", "nadw_K_wlasny", "nadw_Ha_core", "nadw_IRT",
            "kotwica_C", "kotwica_N", "kotwica_akt", "kotwica_akt_zrodlo"]
    A = anch.set_index("obsid")
    out = ref.drop(columns=[c for c in keep + ["cn_obsid"] if c in ref.columns]).copy()
    rows = []
    for _, r in ref.iterrows():
        used = int(r["obsid_uzyte"])
        allids = [int(x) for x in str(r.get("wszystkie_obsid", used)).split(";") if x.strip().isdigit()]
        rec = A.loc[used, keep].to_dict() if used in A.index else {k: np.nan for k in keep}
        if (pd.isna(rec.get("c_fe")) or pd.isna(rec.get("S_L"))) and len(allids) > 1:
            for o in allids:
                if o == used or o not in A.index:
                    continue
                alt = A.loc[o]
                if pd.isna(rec.get("c_fe")) and pd.notna(alt["c_fe"]):
                    for k in ["c_fe", "c_fe_err", "n_fe", "n_fe_err", "c_n", "cn_zrodlo", "cn_dopasowanie",
                              "cn_sep_arcsec", "kotwica_C", "kotwica_N"]:
                        rec[k] = alt[k]
                    rec["cn_zrodlo"] = f"{alt['cn_zrodlo']} [obsid {o}]"
                if pd.isna(rec.get("S_L")) and pd.notna(alt["S_L"]):
                    for k in ["S_L", "S_L_err", "akt_zrodlo", "nadw_S_L"]:
                        rec[k] = alt[k]
        rows.append(rec)
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def plots_and_report(anch, ref_new, basal, notes):
    out = ASTRO_REPORTS_DIR
    pop = ref_new.get("populacja_toomre", pd.Series("", index=ref_new.index)).astype(str)
    halo = pop.str.contains("halo"); thick = pop.str.contains("gruby")
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    for m, c, l in [(~halo & ~thick, "#9a9993", "dysk (cienki/przejściowy)"), (thick, "#eb6834", "gruby dysk"), (halo, "#2a78d6", "halo")]:
        ax[0, 0].scatter(ref_new.feh[m], ref_new.c_fe[m], s=6, c=c, label=l)
        ax[0, 1].scatter(ref_new.feh[m], ref_new.n_fe[m], s=6, c=c)
        ax[0, 2].scatter(ref_new.logg[m], ref_new.c_n[m], s=6, c=c)
    ax[0, 0].set(xlabel="[Fe/H]", ylabel="[C/Fe]"); ax[0, 0].legend(fontsize=7)
    ax[0, 1].set(xlabel="[Fe/H]", ylabel="[N/Fe]"); ax[0, 2].set(xlabel="log g", ylabel="[C/N]")
    ax[0, 2].invert_xaxis()
    sc = ax[1, 0].scatter(anch.teff, anch.idx_S_wlasny, s=4, c=anch.logg, cmap="viridis")
    b = basal.get("nadw_S_wlasny")
    if isinstance(b, tuple):
        ax[1, 0].plot(b[0], b[1], "r.", ms=2, label=f"poziom bazowy karłów ({BASAL_PERCENTILE:.0f}. percentyl sąsiadów w Teff, [Fe/H])")
        ax[1, 0].legend(fontsize=7)
    ax[1, 0].set(xlabel="Teff [K]", ylabel="S własny (Ca II H+K)"); fig.colorbar(sc, ax=ax[1, 0], label="log g")
    ax[1, 1].scatter(anch.S_L, anch.idx_S_wlasny, s=4, c="k")
    ax[1, 1].set(xlabel="S_L (Zhang+2022)", ylabel="S własny", title="kontrola: indeks własny vs katalog")
    ax[1, 2].scatter(anch.c_fe, anch.idx_G4300_EW, s=4, c=anch.teff, cmap="coolwarm_r")
    ax[1, 2].set(xlabel="[C/Fe] (katalog)", ylabel="EW pasma G [Å]", title="kontrola: pasmo G vs [C/Fe]")
    for a_ in ax.flat:
        a_.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out / "02a_C_N_aktywnosc.png", dpi=DPI); plt.close(fig)
    log(f"  ✓ wykres: {out / '02a_C_N_aktywnosc.png'}")

    def cov(df, col):
        return f"{df[col].notna().sum():5d}/{len(df)}"
    lines = [f"  widma: [C/Fe] {cov(anch, 'c_fe')}, [N/Fe] {cov(anch, 'n_fe')}, S_L {cov(anch, 'S_L')}, "
             f"S własny {cov(anch, 'idx_S_wlasny')}",
             f"  kotwice (po kontroli jakości): C {cov(anch, 'kotwica_C')}, N {cov(anch, 'kotwica_N')}, "
             f"aktywność {cov(anch, 'kotwica_akt')}",
             f"  gwiazdy: [C/Fe] {cov(ref_new, 'c_fe')}, S_L {cov(ref_new, 'S_L')}"]
    if "populacja_toomre" in ref_new:
        g = ref_new.groupby("populacja_toomre")
        lines.append("  pokrycie wg populacji Toomre'a (gwiazdy z [C/Fe] / z S_L / razem):")
        for k, d in g:
            lines.append(f"    {k:<22s} {d.c_fe.notna().sum():4d} / {d.S_L.notna().sum():4d} / {len(d):4d}")
    corr = []
    for a_, b_ in [("idx_S_wlasny", "S_L"), ("idx_G4300_EW", "c_fe"), ("idx_CN3883_S3839", "n_fe"), ("idx_CN4215_mag", "n_fe")]:
        ok = anch[[a_, b_]].dropna()
        if len(ok) > 10:
            corr.append(f"  Spearman({a_}, {b_}) = {ok.corr('spearman').iloc[0, 1]:+.3f}  (n={len(ok)})")
    write_report(out / "02a_raport_C_N_aktywnosc.txt", "DOCIĄGNIĘCIE [C/Fe], [N/Fe] I AKTYWNOŚCI CHROMOSFERYCZNEJ", [
        ("Źródła i rozpoznane kolumny", notes or ["  brak"]),
        ("Pokrycie", lines),
        ("Kontrola: indeksy własne vs katalogi", corr or ["  za mało wspólnych danych"]),
        ("Kontrola jakości kotwic", [f"  błąd [C/Fe], [N/Fe] ≤ {CN_MAX_ERR} dex (jeśli katalog podaje błędy), Teff w {CN_TEFF_RANGE},",
                                     f"  flaga q_[X/Fe] = {CN_FLAG_GOOD}, Mult ≠ {CN_MULT_EXCLUDE}",
                                     f"  aktywność: log10(S / poziom bazowy(Teff, [Fe/H])); tylko log g ≥ {ACT_LOGG_MIN}" if ACT_ONLY_DWARFS
                                     else "  aktywność: wszystkie gwiazdy",
                                     "  [C/N] u olbrzymów ubogich w metale zależy także od dodatkowego mieszania",
                                     "  (powyżej 'bump' na RGB), nie tylko od masy i wieku."]),
        ("Okna indeksów własnych (próżnia, Å)", [f"  {k:<8s} {v}" for k, v in INDEX_DEFS.items()]),
    ])


# ============================================================================
# ============================ URUCHOMIENIE ===================================
# ============================================================================

def run():
    ref, cat = load_local()
    idx, notes = measure_indices(cat)
    notes = ["Indeksy własne:"] + notes

    log("=" * 80); log("KROK 2a.3: [C/Fe] i [N/Fe]"); log("=" * 80)
    sess = None
    cands = []
    if USE_VIZIER_CN or USE_ZENODO_ACTIVITY:
        try:
            sess = http()
        except ImportError:
            log("  ⚠ brak pakietu requests – pomijam źródła sieciowe")
    if USE_VIZIER_CN and sess is not None:
        try:
            c, n = cn_from_vizier(sess, cat); cands.append(c); notes += n
        except Exception as e:
            log(f"  ⚠ VizieR/XMatch niedostępny: {e}"); notes.append(f"  ⚠ VizieR/XMatch: {e}")
    for p in LOCAL_CN_FILES:
        try:
            c, n = cn_from_local(Path(p), cat); cands.append(c); notes += n
        except Exception as e:
            log(f"  ⚠ {p}: {e}"); notes.append(f"  ⚠ {p}: {e}")
    cn = pick_best_cn(pd.concat(cands, ignore_index=True), cat) if cands else pd.DataFrame(columns=["obsid"])
    log(f"  ✓ [C/Fe]/[N/Fe] dla {cn['c_fe'].notna().sum() if 'c_fe' in cn else 0} widm")

    log("=" * 80); log("KROK 2a.4: AKTYWNOŚĆ (S_L)"); log("=" * 80)
    act = pd.DataFrame(columns=["obsid"])
    if USE_ZENODO_ACTIVITY and (sess is not None or LOCAL_ACTIVITY_FILE):
        try:
            path = zenodo_activity_file(sess)
            act, n = activity_from_file(path, set(cat["obsid"].astype(np.int64))); notes += n
        except Exception as e:
            log(f"  ⚠ baza aktywności niedostępna: {e}"); notes.append(f"  ⚠ Zenodo/S_L: {e}")

    log("=" * 80); log("KROK 2a.5: KOTWICE I ZAPIS"); log("=" * 80)
    anch, basal = build_anchors(cat, idx, cn, act)
    anch.to_csv(ANCHORS_OUT, index=False)
    log(f"  ✓ {ANCHORS_OUT}  ({len(anch)} widm)")
    backup = REF_CSV.with_name(REF_CSV.stem + ".przed_02a.csv")
    if not backup.exists():
        shutil.copy(REF_CSV, backup); log(f"  ✓ kopia oryginału: {backup}")
    ref_new = update_reference(ref, anch)
    ref_new.to_csv(REF_CSV, index=False)
    log(f"  ✓ {REF_CSV} uzupełniony: {ref_new.shape[1] - ref.shape[1]} nowych kolumn")
    plots_and_report(anch, ref_new, basal, notes)
    log("\n✓ KONIEC KROKU 2a")
    return anch, ref_new


if __name__ == "__main__":
    run()