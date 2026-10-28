"""
KROK 01b: GAIA DR3 – KINEMATYKA I ZBIORY REFERENCYJNE
====================================================

Tworzy gwiazdy_qift_kinematyka.csv (1 wiersz = 1 gwiazda), którego używają kroki 02a, 03 i 04.

1. Z astro_reports/catalog_with_redshift.csv (krok 01, 1 wiersz = 1 widmo) wybiera dla każdej
   gwiazdy (designation) widmo o najwyższym S/N w paśmie g.
2. Pobiera Gaia DR3 (VizieR I/355/gaiadr3) przez CDS XMatch w promieniu 2.2″; jeśli katalog LAMOST
   podaje gaia_source_id, wybiera ten obiekt, w przeciwnym razie najbliższy.
3. Prędkość radialna: Gaia, a gdy jej brak – LAMOST poprawiona o medianowy offset LAMOST − Gaia.
   Odległość: 1/paralaksa dla plx/σ > 5, inaczej GSP-Phot (kolumna Dist).
4. U, V, W względem LSR (ruch Słońca: Schönrich i in. 2010), v_tot, niepewność v_tot metodą
   Monte Carlo, wysokość z nad płaszczyzną, prawdopodobieństwa populacji (Bensby i in. 2014).
5. Etykiety: dobra astrometria (RUWE < 1.4 i plx/σ > 5), populacja Toomre’a, zbiór referencyjny
   d-QSA: dysk (dobra astrometria, v_tot < 70 km/s), halo (dobra astrometria, v_tot > 220 km/s).

Uruchomienie: python 01b_gaia_kinematyka.py
  (pobrana tabela Gaia jest zapisywana w dane_zewnetrzne/gaia_dr3_xmatch.csv i używana ponownie;
   można też podać własny plik w GAIA_CSV_LOCAL).
Uwaga: kroku 02a nie trzeba powtarzać przed 01b – 02a dopisuje kolumny do tego pliku.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# ======================= PARAMETRY URUCHOMIENIOWE ===========================
# ============================================================================
CATALOG_CSV = Path("astro_reports") / "catalog_with_redshift.csv"
QIFTDATA_CSV = Path("astro_reports") / "qiftdata.csv"      # opcjonalnie: latenty (jeśli istnieją)
OUT_CSV = Path("gwiazdy_qift_kinematyka.csv")
REPORT_TXT = Path("astro_reports") / "01b_gaia_kinematyka_raport.txt"
CACHE_DIR = Path("dane_zewnetrzne")
GAIA_CSV_LOCAL: Path | None = None        # np. Path("dane_zewnetrzne/gaia_dr3_astrometria_pelna.csv")
SPECTRA_DIR = Path("widma")               # tylko do kolumny sciezka_widma
FITS_NAME = "spec-obsid-{obsid}.fits"

XMATCH_URL = "https://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync"
GAIA_TABLE = "vizier:I/355/gaiadr3"
XMATCH_RADIUS_ARCSEC = 2.2
HTTP_TIMEOUT = 600

GOOD_RUWE = 1.4
GOOD_PLX_SNR = 5.0
E_RV_LAMOST = 5.0                          # przyjęta niepewność RV z LAMOST [km/s]
N_MONTE_CARLO = 300
SUN_UVW = np.array([11.1, 12.24, 7.25])    # Schönrich, Binney & Dehnen 2010
Z_SUN_KPC = 0.0208                          # wysokość Słońca nad płaszczyzną
NGP_RA, NGP_DEC = 192.85948, 27.12825       # biegun galaktyczny (J2000)
DISK_VMAX, HALO_VMIN = 70.0, 220.0
GIANT_LOGG, DWARF_LOGG, GIANT_TEFF_MAX = 3.5, 4.0, 5300.0
SEED = 0

# macierz ICRS -> galaktyczne (Hipparcos)
T_GAL = np.array([[-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
                  [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
                  [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669]])
# Bensby, Feltzing & Oey 2014, Tab. A.1: X, σU, σV, σW, V_asym
BENSBY = {"cienki_dysk": (0.85, 35, 20, 16, -15), "gruby_dysk": (0.09, 67, 38, 35, -46),
          "halo": (0.0015, 160, 90, 90, -220)}


def log(msg=""):
    print(msg, flush=True)


# ============================================================================
# ============================== 1. WIDMA ====================================
# ============================================================================
def load_stars() -> pd.DataFrame:
    cat = pd.read_csv(CATALOG_CSV, dtype={"gaia_source_id": str, "obsid": str}, low_memory=False)
    log(f"Katalog widm: {len(cat)} widm, {cat.designation.nunique()} gwiazd")
    cat = cat.sort_values("snrg", ascending=False)
    grp = cat.groupby("designation", sort=False)
    allobs = grp["obsid"].apply(lambda s: ";".join(s.astype(str))).rename("wszystkie_obsid")
    nsp = grp.size().rename("n_widm_gwiazdy")
    best = cat.drop_duplicates("designation").set_index("designation")
    best = best.join(allobs).join(nsp).reset_index()
    best["obsid_uzyte"] = best["obsid"].astype(str)
    best["plik_widma"] = [FITS_NAME.format(obsid=o) for o in best.obsid_uzyte]
    best["sciezka_widma"] = [str(SPECTRA_DIR / p) for p in best.plik_widma]
    best["wszystkie_pliki_widm"] = [";".join(FITS_NAME.format(obsid=o) for o in s.split(";")) for s in best.wszystkie_obsid]
    # w katalogu LAMOST source_id bywa zapisany jako liczba zmiennoprzecinkowa (np. 3.9447005539597783e+18),
    # więc porównujemy go z Gaia jako float (dokładność ~1e-16 względnie wystarcza do jednoznaczności)
    best["gaia_id_kat"] = pd.to_numeric(best["gaia_source_id"], errors="coerce")
    return best


# ============================================================================
# ============================== 2. GAIA =====================================
# ============================================================================
def xmatch_gaia(stars: pd.DataFrame) -> pd.DataFrame:
    cache = CACHE_DIR / "gaia_dr3_xmatch.csv"
    if GAIA_CSV_LOCAL is not None:
        log(f"Gaia: plik lokalny {GAIA_CSV_LOCAL}")
        return pd.read_csv(GAIA_CSV_LOCAL, dtype={"Source": str}, low_memory=False)
    if cache.exists():
        log(f"Gaia: używam zapisanego wyniku {cache}")
        return pd.read_csv(cache, dtype={"Source": str}, low_memory=False)
    import requests
    CACHE_DIR.mkdir(exist_ok=True)
    buf = io.StringIO(); stars[["designation", "ra", "dec"]].to_csv(buf, index=False)
    log(f"Gaia: CDS XMatch {GAIA_TABLE}, {len(stars)} pozycji, promień {XMATCH_RADIUS_ARCSEC}″ ...")
    r = requests.post(XMATCH_URL, data=dict(request="xmatch", distMaxArcsec=XMATCH_RADIUS_ARCSEC,
                                            RESPONSEFORMAT="csv", cat2=GAIA_TABLE, colRA1="ra", colDec1="dec"),
                      files=dict(cat1=("pozycje.csv", buf.getvalue())), timeout=HTTP_TIMEOUT,
                      headers={"User-Agent": "qift-halo-pipeline/01b"})
    r.raise_for_status()
    g = pd.read_csv(io.StringIO(r.text), dtype={"Source": str}, low_memory=False)
    g.to_csv(cache, index=False)
    log(f"  ✓ {len(g)} dopasowań zapisano w {cache}")
    return g


def pick_gaia(stars: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy(); g["Source"] = g["Source"].astype(str).str.replace(r"\.0$", "", regex=True)
    g = g.sort_values("angDist")
    rows = []
    for _, s in stars[["designation", "gaia_id_kat"]].iterrows():
        c = g[g.designation == s.designation]
        if len(c) == 0:
            rows.append(dict(designation=s.designation)); continue
        same = np.isclose(pd.to_numeric(c.Source, errors="coerce").to_numpy(float), s.gaia_id_kat, rtol=1e-15, atol=0) \
            if np.isfinite(s.gaia_id_kat) else np.zeros(len(c), bool)
        if same.any():
            r = c[same].iloc[0]; how = "source_id"
        else:
            r = c.iloc[0]; how = "pozycja"
        rows.append(dict(designation=s.designation, gaia_dr3_source_id=r.Source, dopasowanie_gaia=how,
                         sep_gaia_arcsec=r.angDist, plx_mas=r.Plx, e_plx_mas=r.e_Plx, pmra_masyr=r.pmRA,
                         e_pmra_masyr=r.e_pmRA, pmdec_masyr=r.pmDE, e_pmdec_masyr=r.e_pmDE, ruwe=r.RUWE,
                         gaia_G=r.Gmag, gaia_bp_rp=r["BP-RP"], rv_gaia_kms=r.RV, e_rv_gaia_kms=r.e_RV,
                         dist_gspphot_pc=r.Dist))
    return pd.DataFrame(rows)


# ============================================================================
# ============================ 3. KINEMATYKA =================================
# ============================================================================
def uvw(ra, dec, d_pc, pmra, pmde, rv):
    a, dl = np.deg2rad(ra), np.deg2rad(dec); k = 4.740470446 * d_pc / 1000.0
    r = np.stack([np.cos(dl) * np.cos(a), np.cos(dl) * np.sin(a), np.sin(dl)])
    ea = np.stack([-np.sin(a), np.cos(a), np.zeros_like(a)])
    ed = np.stack([-np.sin(dl) * np.cos(a), -np.sin(dl) * np.sin(a), np.cos(dl)])
    return T_GAL @ (rv * r + k * pmra * ea + k * pmde * ed) + SUN_UVW[:, None]


def kinematics(D: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes = []
    D["rv_lamost_kms"] = D["rv_used_kms"]
    m = D.rv_gaia_kms.notna() & D.rv_lamost_kms.notna()
    d = D.rv_lamost_kms[m] - D.rv_gaia_kms[m]; off = float(np.median(d))
    mad = 1.4826 * float(np.median(np.abs(d - off)))
    notes.append(f"  RV LAMOST − Gaia: mediana {off:+.2f} km/s, σ(MAD) {mad:.2f} km/s, n = {m.sum()}")
    hg = D.rv_gaia_kms.notna()
    D["rv_uzyte_kms"] = np.where(hg, D.rv_gaia_kms, D.rv_lamost_kms - off)
    e_rv = np.where(hg, D.e_rv_gaia_kms, E_RV_LAMOST)
    D["rv_zrodlo"] = np.where(hg, "Gaia", f"LAMOST{-off:+.2f}")
    po = (D.plx_mas / D.e_plx_mas).to_numpy()
    D["odleglosc_pc"] = np.where(po > GOOD_PLX_SNR, 1000.0 / D.plx_mas, D.dist_gspphot_pc)
    D["odleglosc_zrodlo"] = np.where(po > GOOD_PLX_SNR, "1/plx", np.where(D.dist_gspphot_pc.notna(), "GSP-Phot", "brak"))
    ra, de = D.ra.to_numpy(float), D.dec.to_numpy(float)
    U, V, W = uvw(ra, de, D.odleglosc_pc.to_numpy(float), D.pmra_masyr.to_numpy(float),
                  D.pmdec_masyr.to_numpy(float), D.rv_uzyte_kms.to_numpy(float))
    D["U_lsr_kms"], D["V_lsr_kms"], D["W_lsr_kms"] = U, V, W
    D["v_tot_kms"] = np.sqrt(U ** 2 + V ** 2 + W ** 2)
    rng = np.random.default_rng(SEED); n = len(D); vt = []
    for _ in range(N_MONTE_CARLO):
        pl = D.plx_mas.to_numpy(float) + rng.normal(size=n) * D.e_plx_mas.to_numpy(float)
        dd = np.where(po > GOOD_PLX_SNR, 1000.0 / np.clip(pl, 1e-3, None),
                      D.odleglosc_pc.to_numpy(float) * (1 + 0.1 * rng.normal(size=n)))
        u, v, w = uvw(ra, de, dd,
                      D.pmra_masyr.to_numpy(float) + rng.normal(size=n) * D.e_pmra_masyr.to_numpy(float),
                      D.pmdec_masyr.to_numpy(float) + rng.normal(size=n) * D.e_pmdec_masyr.to_numpy(float),
                      D.rv_uzyte_kms.to_numpy(float) + rng.normal(size=n) * e_rv)
        vt.append(np.sqrt(u ** 2 + v ** 2 + w ** 2))
    D["e_v_tot_kms"] = np.nanstd(np.array(vt), axis=0)
    a0, d0 = np.deg2rad(NGP_RA), np.deg2rad(NGP_DEC); a, dl = np.deg2rad(ra), np.deg2rad(de)
    b = np.arcsin(np.sin(dl) * np.sin(d0) + np.cos(dl) * np.cos(d0) * np.cos(a - a0))
    D["z_kpc"] = D.odleglosc_pc * np.sin(b) / 1000.0 + Z_SUN_KPC
    f = {}
    for k, (X, su, sv, sw, va) in BENSBY.items():
        kk = 1.0 / ((2 * np.pi) ** 1.5 * su * sv * sw)
        f[k] = X * kk * np.exp(-U ** 2 / (2 * su ** 2) - (V - va) ** 2 / (2 * sv ** 2) - W ** 2 / (2 * sw ** 2))
    S = sum(f.values())
    for k in f:
        D["P_" + k] = f[k] / S
    D["dobra_astrometria"] = (D.ruwe < GOOD_RUWE) & (po > GOOD_PLX_SNR)
    v = D.v_tot_kms
    D["populacja_toomre"] = np.select(
        [v.isna(), v < 50, v < DISK_VMAX, v <= 180, v <= HALO_VMIN, v > HALO_VMIN],
        ["brak", "<50 cienki_dysk", "50-70 przejsciowa", "70-180 gruby_dysk", "180-220 przejsciowa", ">220 halo"], "brak")
    D["zbior_ref_dqsa"] = np.where(D.dobra_astrometria & (v < DISK_VMAX), "dysk",
                                   np.where(D.dobra_astrometria & (v > HALO_VMIN), "halo", "-"))
    notes += [f"  odległości: 1/plx {(D.odleglosc_zrodlo == '1/plx').sum()}, GSP-Phot "
              f"{(D.odleglosc_zrodlo == 'GSP-Phot').sum()}, brak {(D.odleglosc_zrodlo == 'brak').sum()}",
              f"  dobra astrometria: {int(D.dobra_astrometria.sum())}",
              f"  mediana niepewności v_tot: {np.nanmedian(D.e_v_tot_kms):.2f} km/s",
              "  populacje Toomre’a: " + ", ".join(f"{k}: {n}" for k, n in D.populacja_toomre.value_counts().items()),
              "  zbiory referencyjne: " + ", ".join(f"{k}: {n}" for k, n in D.zbior_ref_dqsa.value_counts().items())]
    return D, notes


def kiel_class(teff, logg):
    return np.select([logg >= DWARF_LOGG, logg >= GIANT_LOGG, teff < GIANT_TEFF_MAX],
                     ["ciag_glowny", "podolbrzym", "olbrzym"], "reszta")


# ============================================================================
# ================================ RUN =======================================
# ============================================================================
COLS = ["designation", "obsid_uzyte", "plik_widma", "sciezka_widma", "n_widm_gwiazdy", "wszystkie_obsid",
        "wszystkie_pliki_widm", "ra", "dec", "gaia_dr3_source_id", "dopasowanie_gaia", "sep_gaia_arcsec",
        "w_atm_1", "w_atm_2", "z_Fe", "z_alpha", "teff", "teff_err", "logg", "logg_err", "feh", "feh_err",
        "alpha_m", "alpha_m_err", "typ_widmowy", "snrg", "klasa_kiel", "rv_lamost_kms", "rv_gaia_kms",
        "e_rv_gaia_kms", "rv_uzyte_kms", "rv_zrodlo", "plx_mas", "e_plx_mas", "pmra_masyr", "e_pmra_masyr",
        "pmdec_masyr", "e_pmdec_masyr", "ruwe", "gaia_G", "gaia_bp_rp", "odleglosc_pc", "odleglosc_zrodlo",
        "z_kpc", "U_lsr_kms", "V_lsr_kms", "W_lsr_kms", "v_tot_kms", "e_v_tot_kms", "P_cienki_dysk",
        "P_gruby_dysk", "P_halo", "dobra_astrometria", "populacja_toomre", "zbior_ref_dqsa"]


def run():
    stars = load_stars()
    g = xmatch_gaia(stars)
    gp = pick_gaia(stars, g)
    D = stars.merge(gp, on="designation", how="left")
    n_miss = D.gaia_dr3_source_id.isna().sum()
    log(f"Gaia: dopasowano {len(D) - n_miss} gwiazd (source_id: {(D.dopasowanie_gaia == 'source_id').sum()}, "
        f"pozycja: {(D.dopasowanie_gaia == 'pozycja').sum()}), brak: {n_miss}")
    D, notes = kinematics(D)
    D["typ_widmowy"] = D.get("subclass")
    D["klasa_kiel"] = kiel_class(D.teff.to_numpy(float), D.logg.to_numpy(float))
    if QIFTDATA_CSV.exists():   # latenty z kroku 02 dla użytego widma (03/04 i tak czytają qiftdata.csv)
        q = pd.read_csv(QIFTDATA_CSV, dtype={"obsid": str}, low_memory=False)
        lat = [c for c in ["w_atm_1", "w_atm_2", "z_Fe", "z_alpha"] if c in q]
        D = D.merge(q[["obsid"] + lat].rename(columns={"obsid": "obsid_uzyte"}), on="obsid_uzyte", how="left")
    for c in COLS:
        if c not in D:
            D[c] = np.nan
    out = D[COLS].sort_values(["ra", "dec"]).reset_index(drop=True)
    if OUT_CSV.exists():
        bak = OUT_CSV.with_suffix(".przed_01b.csv"); OUT_CSV.replace(bak); log(f"  stara wersja -> {bak}")
    out.to_csv(OUT_CSV, index=False)
    REPORT_TXT.parent.mkdir(exist_ok=True)
    REPORT_TXT.write_text("\n".join(["KROK 01b – Gaia DR3 i kinematyka", "=" * 40,
                                     f"gwiazd: {len(out)}; dopasowane do Gaia: {out.gaia_dr3_source_id.notna().sum()}",
                                     *notes]), encoding="utf-8")
    log("\n".join(notes)); log(f"✓ zapisano {OUT_CSV} ({len(out)} gwiazd) i {REPORT_TXT}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        GAIA_CSV_LOCAL = Path(sys.argv[1])
    run()
