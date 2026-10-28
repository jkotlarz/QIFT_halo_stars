# -*- coding: utf-8 -*-
"""
04_qift_eigenstates_p2_2klasy_7q.py
======================================================================
KROK 4 potoku QIFT-spectra – PODEJŚCIE 2: 7 latentów (atmosfera + 5 chemicznych), dwie klasy: dysk i halo
  kubity: w_atm_1, w_atm_2, z_Fe, z_alpha, z_C, z_N, z_act
  referencja: dwie klasy – dysk (v_tot < 70 km/s) i halo (v_tot > 220 km/s)

Co robi ten plik
----------------
1. Wczytuje stany, topologię i dane gwiazd z kroku 03 tego podejścia.
2. Liczy miary nietypowości z operatorowej typologii (notatka S. Zająca):
     jedna klasa : (1) a0 = 1 − F0,  E = ⟨ψ|H|ψ⟩,  (5) Q – reszta poza modami,
                   (6) T² – skrajność w zachowanych modach (odchylenia Hilberta–Schmidta),
                   wersje „kierunkowe” (tylko gwiazdy o z_Fe poniżej mediany referencji);
     dwie klasy  : (2) s = F_halo − F_dysk,  (4) margines m = |s|,
                   (3) a_open = 1 − max F,  (10) waga mieszaniny w i reszta a_rec.
   F_c = Tr(P_x Π_c,k) z projektorem rangi PROJECTOR_RANK.
3. Uczciwa ocena: gdy referencją jest dysk lub dwie klasy, wyniki gwiazd
   referencyjnych są liczone poza foldem (CV_FOLDS × CV_REPEATS), a topologia
   (przerwa masowa) jest uczona ponownie w każdym foldzie. Progi decyzji
   kalibrowane na wynikach poza foldem (fałszywe alarmy ALPHA na referencji).
4. Ewaluacja wobec halo kinematycznego (Gaia DR3, v_tot > 220 km/s):
   AUC, AP, precyzja@N; dla decyzji: precyzja, czułość, F1, MCC; bootstrap 95%.
5. Wpływ topologii: kąty wyuczone (przerwa masowa) vs θ = 0 vs oryginalne
   – sparowany bootstrap ΔAUC, ΔAP.
6. Metody klasyczne na tych samych latentach i tych samych foldach:
   jedna klasa: Mahalanobis (MCD), PCA-T²/Q, Isolation Forest, LOF; dwie klasy:
   regresja logistyczna, QDA; zawsze: sam −z_Fe.
7. (tylko wersje z kubitami atmosfery) Splątanie atmosfera (q0,q1) – chemia (pozostałe kubity) w grupach,
   testy różnic (Mann–Whitney + delta Cliffa, test permutacyjny dla I(A:B)),
   kontrole: θ = 0, przetasowanie chemii, klasyczna MI (gauss + KSG),
   porównania wewnątrz stadiów ewolucyjnych (karły / olbrzymy).

Wersja z nowymi latentami chemicznymi (z_C, z_N, z_act): miary T² i Q liczone są przez jądro
wierności K_ij = |⟨ψ_i|ψ_j⟩|² (centrowanie jądra = odchylenia Hilberta–Schmidta), co daje te same
wartości co jawne wektory odchyleń, ale bez macierzy N × 2·4^n (dla 7 kubitów ~350 MB).

Uruchomienie: python 04_qift_eigenstates_p2_2klasy_7q.py
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================================
# ======================= PARAMETRY URUCHOMIENIOWE ===========================
# ============================================================================

VARIANT = "p2_2klasy_7q"
MODE = "dwie_klasy"                 # "jedna_klasa_cala" | "jedna_klasa_dysk" | "dwie_klasy"
HAS_ATM = True                # czy kubity q0,q1 to atmosfera (analiza splątania atmosfera–chemia)
FEATURES = ["w_atm_1", "w_atm_2", "z_Fe", "z_alpha", "z_C", "z_N", "z_act"]

# --- Wejście / wyjście ---
HILBERT_DIR = Path("hilbertspace_topology") / VARIANT
STATES_NPZ = HILBERT_DIR / "qift_states.npz"
DATA_CSV = HILBERT_DIR / "qiftdata_scaled.csv"
SPECTRA_NPZ = Path("astro_reports") / "spectra_deredshifted.npz"
MODEL_META_JSON = Path("astro_reports") / "model" / "physically_structured_vae_meta.json"
QIFT_REPORTS_DIR = Path("qift_reports") / VARIANT

# --- Miary ---
PROJECTOR_RANK = 1                # ranga projektora Π_c,k
K_MODES = 4                  # liczba zachowanych modów odchyleń (T², Q)
HS_ETA = 1e-6                     # η w T²
RHO_EPS = 1e-6
ENERGY_EPS = 1e-12

# --- Walidacja krzyżowa i ponowne uczenie topologii ---
CV_FOLDS = 5
CV_REPEATS = 3
CV_SEED = 0
RELEARN_TOPOLOGY_IN_CV = True     # przerwa masowa uczona tylko na foldzie treningowym
INITIAL_ANGLE_DEG = 20.0
TOPOLOGY_BOUNDS_DEG = (0.0, 90.0)
TOPOLOGY_REG_LAMBDA = 1e-4
ANGLE_PENALTY_THRESHOLD_DEG = 80.0
ANGLE_PENALTY_LAMBDA = 1.5
STEP_SCHEDULE_DEG = [12.0, 6.0, 2.0, 0.5]
TRIALS_PER_STAGE = 60
BATCH_PER_TRIAL = 4
PATIENCE = 8

# --- Decyzje ---
ALPHA = 0.05                      # dopuszczalny odsetek fałszywych alarmów na referencji
ANOMALY_F_QUANTILE = 0.05         # reguła oryginalnego potoku: F0 ≤ q ORAZ E ≥ q'
ANOMALY_E_QUANTILE = 0.95
MARGIN_QUANTILE = 0.10            # próg „małego marginesu” (dwie klasy)

# --- Ewaluacja ---
HALO_VTOT = 220.0                 # etykieta prawdy (z dobrą astrometrią)
BOOTSTRAP = 1000
TOP_N = 50                        # precyzja@N

# --- Splątanie ---
ENT_BOOT = 200
ENT_PERM = 500
ENT_SHUFFLE = 100
KSG_K = 5
KSG_SUBSAMPLE = 30           # KSG na podpróbach równej liczności (obciążenie zależy od n)
MIN_GROUP = 5

PARAM_COLS = ["teff", "logg", "feh", "alpha_m"]
FALLBACK_WINDOWS = [("Fe_5315", 5305, 5331), ("MgI_b", 5160, 5190), ("Halpha", 6545, 6580),
                    ("CaII_HK", 3925, 3980), ("Fe_5710_5750", 5705, 5755), ("TiII_4369", 4360, 4382)]
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


def basis_labels(n: int) -> list[str]:
    return [format(b, f"0{n}b") for b in range(2 ** n)]


def cohen_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def compare_groups(df: pd.DataFrame, mask_a, mask_b, cols, name_a="A", name_b="B") -> pd.DataFrame:
    """Statystyki i testy (Welch t, Mann-Whitney U, KS, d Cohena) dla grupy A vs B."""
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        a = pd.to_numeric(df.loc[mask_a, c], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[mask_b, c], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        t_p = stats.ttest_ind(a, b, equal_var=False).pvalue
        u_p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        rows.append({"feature": c, f"median_{name_a}": a.median(), f"median_{name_b}": b.median(),
                     f"mean_{name_a}": a.mean(), f"mean_{name_b}": b.mean(), "cohen_d": cohen_d(a, b),
                     "p_welch": t_p, "p_mannwhitney": u_p, "p_ks": stats.ks_2samp(a, b).pvalue,
                     "signif_0.01": "TAK" if min(t_p, u_p) < 0.01 else "nie"})
    return pd.DataFrame(rows)


def load_states(npz_path: Path, csv_path: Path):
    print("=" * 80)
    print("KROK 4.1: WCZYTANIE STANÓW KWANTOWYCH I DANYCH REFERENCYJNYCH")
    print("=" * 80)
    d = np.load(npz_path, allow_pickle=True)
    st = {k: d[k] for k in d.files}
    st["features"] = [str(f) for f in st["features"]]
    st["pairs"] = [tuple(map(int, p)) for p in st["pairs"]]
    df = pd.read_csv(csv_path, dtype={"gaia_dr3_source_id": str})
    if len(df) != st["Psi"].shape[0]:
        raise ValueError("Liczba wierszy danych ≠ liczba stanów – uruchom ponownie krok 03 tego wariantu")
    df["dobra_astrometria"] = df["dobra_astrometria"].astype(str).str.lower().isin(["true", "1"])
    df["halo_kin"] = df["dobra_astrometria"] & (df["v_tot_kms"] > HALO_VTOT)
    df["eval_mask"] = df["dobra_astrometria"] & df["v_tot_kms"].notna()
    print(f"✓ Psi={st['Psi'].shape}, cechy={st['features']}, kąty RZZ [°]="
          f"{np.round(np.rad2deg(st['angles']), 2).tolist()} | halo kin.: {int(df.halo_kin.sum())}")
    return st, df


# ---------------- kodowanie (identyczne jak w kroku 03) ----------------
def _apply_1q(Psi, U, q, n):
    N = Psi.shape[0]
    T = Psi.reshape(N, 2 ** (n - 1 - q), 2, 2 ** q)
    return np.einsum("nab,nxby->nxay", U, T).reshape(N, 2 ** n)


def _ry_mats(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def encode_states(S, pairs, angles):
    N, n = S.shape
    ang = np.clip(S, 0, 1) * (np.pi / 2)
    Psi = np.zeros((N, 2 ** n), complex); Psi[:, 0] = 1.0
    for q in range(n):
        Psi = _apply_1q(Psi, _ry_mats(ang[:, q]).astype(complex), q, n)
    idx = np.arange(2 ** n); z = 1 - 2 * ((idx[:, None] >> np.arange(n)[None, :]) & 1)
    phase = np.zeros(2 ** n)
    for (i, j), th in zip(pairs, angles):
        phase += -0.5 * th * z[:, i] * z[:, j]
    Psi = Psi * np.exp(1j * phase)[None, :]
    for q in range(n):
        Psi = _apply_1q(Psi, _ry_mats(ang[:, q]).astype(complex), q, n)
    return Psi


def density_matrix(Psi):
    rho = (Psi.T @ Psi.conj()) / Psi.shape[0]
    rho = 0.5 * (rho + rho.conj().T)
    mn = np.linalg.eigvalsh(rho).min()
    if mn <= 0:
        rho = rho + (RHO_EPS - mn if mn < 0 else RHO_EPS) * np.eye(rho.shape[0])
    return rho / np.trace(rho).real


# ---------------- metryki (bez zależności od sklearn) ----------------
def auc_score(y, s) -> float:
    y = np.asarray(y, bool); r = stats.rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def ap_score(y, s) -> float:
    """Average precision = średnia precyzja w pozycjach kolejnych gwiazd pozytywnych."""
    y = np.asarray(y, bool); o = np.argsort(-np.asarray(s, float), kind="mergesort"); yy = y[o]
    tp = np.cumsum(yy); prec = tp / np.arange(1, len(yy) + 1)
    return float(prec[yy].mean()) if yy.any() else np.nan


def metrics_ci(y, s, B=BOOTSTRAP, seed=0) -> dict:
    y, s = np.asarray(y, bool), np.asarray(s, float)
    rng = np.random.default_rng(seed); ab, pb = [], []
    for _ in range(B):
        j = rng.integers(0, len(y), len(y))
        if 0 < y[j].sum() < len(j):
            ab.append(auc_score(y[j], s[j])); pb.append(ap_score(y[j], s[j]))
    return dict(AUC=auc_score(y, s), AUC_lo=np.percentile(ab, 2.5), AUC_hi=np.percentile(ab, 97.5),
                AP=ap_score(y, s), AP_lo=np.percentile(pb, 2.5), AP_hi=np.percentile(pb, 97.5),
                n=int(len(y)), n_pos=int(y.sum()))


def curves(y, s):
    y = np.asarray(y, bool); o = np.argsort(-np.asarray(s, float), kind="mergesort"); yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(~yy)
    return dict(fpr=np.r_[0, fp / max((~y).sum(), 1)], tpr=np.r_[0, tp / max(y.sum(), 1)],
                rec=tp / max(y.sum(), 1), prec=tp / np.arange(1, len(yy) + 1))


def metric_line(name, m) -> str:
    return (f"  {name:34s} AUC {m['AUC']:.3f} [{m['AUC_lo']:.3f}; {m['AUC_hi']:.3f}]   "
            f"AP {m['AP']:.3f} [{m['AP_lo']:.3f}; {m['AP_hi']:.3f}]   (n={m['n']}, halo={m['n_pos']})")


POP_ORDER = ["<50 cienki_dysk", "50-70 przejsciowa", "70-180 gruby_dysk", "180-220 przejsciowa", ">220 halo", "brak"]


def population_table(df, flag) -> pd.DataFrame:
    t = pd.crosstab(df["populacja_toomre"], np.where(flag, "oznaczone", "pozostałe"), colnames=["decyzja"])
    t = t.reindex([p for p in POP_ORDER if p in t.index]).fillna(0).astype(int)
    t["frakcja_oznaczonych"] = t.get("oznaczone", 0) / t.sum(axis=1)
    return t


def plot_toomre(df, flag, label_flag, out_path, title, ref_mask=None, ref_label=None):
    d = df[df["v_tot_kms"].notna()]; f = np.asarray(flag)[df["v_tot_kms"].notna().to_numpy()]
    x, y = d["V_lsr_kms"].to_numpy(), np.sqrt(d["U_lsr_kms"] ** 2 + d["W_lsr_kms"] ** 2).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax in axes:
        for r, ls in [(70, ":"), (180, "--"), (220, "-")]:
            t = np.linspace(0, np.pi, 300); ax.plot(r * np.cos(t), r * np.sin(t), ls=ls, c="gray", lw=1)
            ax.text(-r - 5, 8, str(r), color="gray", fontsize=8, ha="right")
        ax.set_xlabel("V względem LSR [km/s]"); ax.set_ylabel("√(U² + W²) [km/s]")
        ax.set_xlim(-530, 130); ax.set_ylim(0, 450); ax.grid(alpha=.25)
    o = np.argsort(-d["feh"].to_numpy())
    sc = axes[0].scatter(x[o], y[o], c=d["feh"].to_numpy()[o], s=8, cmap="viridis", vmin=-2.3, vmax=0.4)
    fig.colorbar(sc, ax=axes[0], label="[Fe/H]"); axes[0].set_title("Diagram Toomre'a – kolor: [Fe/H]")
    axes[1].scatter(x[~f], y[~f], s=6, c="steelblue", alpha=.4, label="pozostałe")
    axes[1].scatter(x[f], y[f], s=18, c="orangered", edgecolors="k", lw=.3, label=label_flag)
    if ref_mask is not None:
        rm = np.asarray(ref_mask)[df["v_tot_kms"].notna().to_numpy()]
        axes[1].scatter(x[rm], y[rm], s=60, facecolors="none", edgecolors="k", lw=.8, label=ref_label)
    axes[1].legend(fontsize=8, loc="upper right"); axes[1].set_title(title)
    fig.tight_layout(); save_fig(fig, out_path)


def plot_roc_pr(y, scores: dict, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, (name, s) in enumerate(scores.items()):
        c = curves(y, s); ls = "-" if i == 0 else ["--", ":", "-."][(i - 1) % 3]; lw = 2.2 if i == 0 else 1.3
        axes[0].plot(c["fpr"], c["tpr"], ls=ls, lw=lw, label=f"{name} (AUC {auc_score(y, s):.3f})")
        axes[1].plot(c["rec"], c["prec"], ls=ls, lw=lw, label=f"{name} (AP {ap_score(y, s):.3f})")
    axes[0].plot([0, 1], [0, 1], c="lightgray"); axes[1].axhline(np.mean(y), c="lightgray")
    axes[0].set(xlabel="odsetek fałszywie dodatnich", ylabel="czułość", title=f"ROC – {title}")
    axes[1].set(xlabel="czułość", ylabel="precyzja", title="precyzja–czułość", ylim=(0, 1.02))
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); save_fig(fig, out_path)


# ---------------- splątanie atmosfera (q0,q1) | chemia (q2,q3) ----------------
def S_vn(rho) -> float:
    l = np.linalg.eigvalsh(rho); l = l[l > 1e-12]
    return float(-(l * np.log2(l)).sum())


def per_star_entropy(P) -> np.ndarray:
    """Entropia splątania A|B stanu czystego [bit]; A = q0,q1 (atmosfera), B = pozostałe kubity (chemia);
    indeks bazy = 4·chemia + atmosfera."""
    s = np.linalg.svd(P.reshape(P.shape[0], -1, 4), compute_uv=False) ** 2
    s = np.clip(s, 0, 1)
    return -np.where(s > 1e-12, s * np.log2(np.where(s > 1e-12, s, 1)), 0.0).sum(1)


def ensemble_measures(P) -> dict:
    rho = P.T @ P.conj() / len(P)
    dB = rho.shape[0] // 4
    R = rho.reshape(dB, 4, dB, 4)                   # [c, a, c', a']
    rA, rB = np.einsum("cacb->ab", R), np.einsum("cada->cd", R)
    pt = R.transpose(0, 3, 2, 1).reshape(4 * dB, 4 * dB)   # transpozycja częściowa po atmosferze
    ln = float(np.log2(np.abs(np.linalg.eigvalsh((pt + pt.conj().T) / 2)).sum()))
    SA, SB, SAB = S_vn(rA), S_vn(rB), S_vn(rho)
    return dict(S_atm=SA, S_chem=SB, S_AB=SAB, MI=SA + SB - SAB, logneg=max(ln, 0.0),
                purity=float(np.real(np.trace(rho @ rho))))


def gauss_mi(Z) -> float:
    C = np.corrcoef(Z.T)
    return float(0.5 * np.log2(np.linalg.det(C[:2, :2]) * np.linalg.det(C[2:, 2:]) / np.linalg.det(C)))


def entanglement_analysis(st, df, extra_vectors: dict, out_dir: Path):
    """
    Splątanie atmosfera–chemia w populacjach halo i dysku (zbior_ref_dqsa) przy kątach tego
    wariantu i przy θ = 0. Kontrola: przetasowanie chemii względem atmosfery w obrębie grupy.
    """
    print("=" * 80)
    print("KROK 4.x: SPLĄTANIE ATMOSFERA–CHEMIA (halo vs dysk)")
    print("=" * 80)
    S, pairs, ang = st["S"], st["pairs"], st["angles"]
    X = df[st["features"]].to_numpy(float)
    grp = {"halo": (df["zbior_ref_dqsa"] == "halo").to_numpy(), "dysk": (df["zbior_ref_dqsa"] == "dysk").to_numpy()}
    rng = np.random.default_rng(0); rows, per_star = [], {}
    for enc_name, th in [("kąty wariantu", np.asarray(ang)), ("θ = 0", np.zeros(len(pairs)))]:
        P = encode_states(S, pairs, th); sA = per_star_entropy(P); per_star[enc_name] = sA
        p_mw = stats.mannwhitneyu(sA[grp["halo"]], sA[grp["dysk"]]).pvalue
        for g, m in grp.items():
            idx = np.where(m)[0]; e = ensemble_measures(P[idx])
            bs = [ensemble_measures(P[rng.choice(idx, len(idx))])["MI"] for _ in range(ENT_BOOT)]
            sh = []
            for _ in range(ENT_SHUFFLE):
                Z = S[idx].copy(); Z[:, 2:] = Z[rng.permutation(len(idx)), 2:]
                sh.append(ensemble_measures(encode_states(Z, pairs, th))["MI"])
            rows.append(dict(kodowanie=enc_name, grupa=g, n=len(idx), S_gwiazdy_mediana=np.median(sA[idx]),
                             S_q16=np.quantile(sA[idx], .16), S_q84=np.quantile(sA[idx], .84),
                             MI=e["MI"], MI_lo=np.percentile(bs, 2.5), MI_hi=np.percentile(bs, 97.5),
                             MI_przetasowane=np.mean(sh), logneg=e["logneg"], S_atm=e["S_atm"], S_chem=e["S_chem"],
                             czystosc=e["purity"], p_MW_S_halo_vs_dysk=p_mw))
    T = pd.DataFrame(rows); T.to_csv(out_dir / "entanglement_halo_disk.csv", index=False)
    cl = []
    for g, m in grp.items():
        idx = np.where(m)[0]; v = [gauss_mi(X[rng.choice(idx, len(idx))]) for _ in range(500)]
        cl.append(dict(grupa=g, n=len(idx), MI_klasyczna=gauss_mi(X[idx]), lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
    CL = pd.DataFrame(cl)
    vec_lines = [f"  S({k}) = {per_star_entropy(np.asarray(v)[None, :])[0]:.3f} bit" for k, v in extra_vectors.items()]
    write_report(out_dir / "entanglement_report.txt", f"SPLĄTANIE ATMOSFERA–CHEMIA – {VARIANT}", [
        ("Definicje", ["  A = kubity q0,q1 (w_atm_1, w_atm_2), B = kubity q2,q3 (z_Fe, z_alpha)",
                       "  S gwiazdy: entropia splątania A|B stanu czystego |ψ⟩ [bit, maks. 2]",
                       "  I(A:B): informacja wzajemna stanu zespołu ρ_grupy; logneg: logarytmiczna negatywność",
                       "  MI_przetasowane: I(A:B) po losowym sparowaniu chemii z atmosferą w obrębie grupy",
                       "  (jeśli ≈ I(A:B), miara odzwierciedla kodowanie, a nie korelacje w danych)"]),
        ("Wyniki", df_lines(T.drop(columns=["S_atm", "S_chem"]), 3, index=False)),
        ("Klasyczna informacja wzajemna atmosfera–chemia (gaussowska, latenty) [bit]", df_lines(CL, 3, index=False)),
        ("Splątanie wektorów stanu (kąty wariantu)", vec_lines or ["  —"]),
    ])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, enc_name in enumerate(per_star):
        for j, (g, col) in enumerate([("halo", "crimson"), ("dysk", "steelblue")]):
            v = per_star[enc_name][grp[g]]
            axes[0].boxplot(v, positions=[i + (j - .5) * .3], widths=.25, patch_artist=True, showfliers=False,
                            boxprops=dict(facecolor=col, alpha=.5), medianprops=dict(color="k"))
            r = T[(T.kodowanie == enc_name) & (T.grupa == g)].iloc[0]
            axes[1].errorbar(i + (j - .5) * .3, r.MI, yerr=[[r.MI - r.MI_lo], [r.MI_hi - r.MI]], fmt="o", c=col,
                             label=g if i == 0 else None)
            axes[1].scatter(i + (j - .5) * .3, r.MI_przetasowane, marker="x", c="k",
                            label="chemia przetasowana" if (i == 0 and j == 0) else None)
    for g, col, ls in [("halo", "crimson", "--"), ("dysk", "steelblue", ":")]:
        axes[1].axhline(CL.set_index("grupa").loc[g, "MI_klasyczna"], c=col, ls=ls, lw=1)
    for ax in axes:
        ax.set_xticks(range(len(per_star)), list(per_star)); ax.grid(alpha=.25, axis="y")
    axes[0].set_ylabel("S gwiazdy [bit]"); axes[0].set_title("Splątanie pojedynczych gwiazd (czerwony: halo)")
    axes[1].set_ylabel("I(A:B) [bit]"); axes[1].set_title("Informacja wzajemna zespołu (linie: klasyczna MI)")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); save_fig(fig, out_dir / "entanglement_halo_disk.png")
    return T, CL


def load_windows():
    if MODEL_META_JSON.exists():
        meta = json.loads(MODEL_META_JSON.read_text(encoding="utf-8"))
        regs = meta.get("signature_regions", [])
        if regs:
            return [(r["name"], r["lambda_min"] - 6, r["lambda_max"] + 6) for r in regs]
    return FALLBACK_WINDOWS


def plot_mean_spectra(res, groups: list, windows, wave, X, out_dir: Path, fname_prefix: str, diff_pair=None):
    """Średnie widma grup w oknach sygnatur (indeks widma = row_id z kroku 02)."""
    if wave is None or X is None:
        return
    ncol = 3; nrow = math.ceil(len(windows) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.8 * nrow), squeeze=False)
    idx = res["row_id"].to_numpy(int)
    for ax, (name, lo, hi) in zip(axes.flat, windows):
        m = (wave >= lo) & (wave <= hi)
        for lab, mask, c in groups:
            sel = idx[np.asarray(mask)]
            if len(sel):
                ax.plot(wave[m], X[sel][:, m].mean(axis=0), color=c, lw=1.6, label=f"{lab} (n={len(sel)})")
        ax.set_title(name, fontsize=9); ax.grid(alpha=.25)
    for ax in list(axes.flat)[len(windows):]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Średnie widma grup w oknach sygnatur – {VARIANT}", fontweight="bold")
    fig.tight_layout(); save_fig(fig, out_dir / f"{fname_prefix}_mean_spectra_groups.png")
    if diff_pair is not None:
        (la, ma), (lb, mb) = diff_pair
        a, b = idx[np.asarray(ma)], idx[np.asarray(mb)]
        if len(a) and len(b):
            fig, ax = plt.subplots(figsize=(16, 4.5))
            ax.plot(wave, X[a].mean(0) - X[b].mean(0), "k-", lw=.6)
            for name, lo, hi in windows:
                ax.axvspan(lo, hi, color="orange", alpha=.25)
            ax.axhline(0, color="gray"); ax.set_xlabel("λ [Å]"); ax.set_ylabel("ΔF"); ax.grid(alpha=.25)
            ax.set_title(f"Różnica widm średnich: {la} − {lb} (pomarańcz: okna sygnatur)")
            save_fig(fig, out_dir / f"{fname_prefix}_mean_spectrum_difference.png")



# ============================================================================
# ====================== 2. TOPOLOGIA (PRZERWA MASOWA) ========================
# ============================================================================

def spectral_full(rho):
    l, V = np.linalg.eigh(rho)
    o = np.argsort(l)[::-1]
    return l[o].real, V[:, o]


def mass_gap_of(P_ref) -> float:
    l, _ = spectral_full(density_matrix(P_ref))
    return float(-np.log(l[1] + ENERGY_EPS) + np.log(l[0] + ENERGY_EPS))


def gap_cost(th, S, pairs, ref_sets):
    P = encode_states(S, pairs, th)
    g = np.mean([mass_gap_of(P[idx]) for idx in ref_sets])
    ex = np.maximum(np.abs(th) - np.deg2rad(ANGLE_PENALTY_THRESHOLD_DEG), 0.0)
    return -g + TOPOLOGY_REG_LAMBDA * float(np.sum(th ** 2)) + ANGLE_PENALTY_LAMBDA * float(np.sum(ex ** 2))


def learn_gap_topology(S, pairs, ref_sets, starts, seed=0):
    """Maksymalizacja przerwy masowej (hill-climbing z kilku startów) – jak w kroku 03."""
    lo, hi = np.deg2rad(TOPOLOGY_BOUNDS_DEG); best_x, best_c = None, np.inf
    for k, x0 in enumerate(starts):
        rng = np.random.default_rng(seed + k)
        x = np.clip(np.asarray(x0, float), lo, hi); c = gap_cost(x, S, pairs, ref_sets)
        for step in STEP_SCHEDULE_DEG:
            no_imp = 0
            for _ in range(TRIALS_PER_STAGE):
                imp = False
                for _ in range(BATCH_PER_TRIAL):
                    d = rng.normal(size=len(x)); d /= max(np.linalg.norm(d), 1e-12)
                    xt = np.clip(x + rng.uniform(.3, 1) * np.deg2rad(step) * d, lo, hi); ct = gap_cost(xt, S, pairs, ref_sets)
                    if ct < c - 1e-12:
                        x, c, imp = xt, ct, True
                no_imp = 0 if imp else no_imp + 1
                if no_imp >= PATIENCE:
                    break
        if c < best_c:
            best_x, best_c = x, c
    return best_x


# ============================================================================
# ========================== 3. MIARY NIETYPOWOŚCI ============================
# ============================================================================

def topk_overlap(P, V, k):
    """F = Tr(P_x Π_k) = Σ_{j<k} |⟨e_j|ψ⟩|²."""
    return (np.abs(P @ V[:, :k].conj()) ** 2).sum(1)


def hs_deviation_measures(P, train, k_modes=K_MODES, eta=HS_ETA):
    """
    T² i Q odchyleń Z_x = |ψ_x⟩⟨ψ_x| − ρ_ref w przestrzeni Hilberta–Schmidta, liczone przez jądro
    K(x, y) = |⟨ψ_x|ψ_y⟩|² (⟨Z_x, Z_y⟩_HS = K(x,y) − a_x − a_y + c, a_x = ⟨ψ_x|ρ|ψ_x⟩, c = Tr ρ²).
    Mody = główne kierunki odchyleń referencji (PCA jądrowe); wynik identyczny z jawnymi wektorami.
    """
    Pt = P[train]
    Kt = np.abs(Pt @ Pt.conj().T) ** 2                 # [n_ref, n_ref]
    Ka = np.abs(P @ Pt.conj().T) ** 2                  # [N, n_ref]
    a_t, c = Kt.mean(1), Kt.mean()
    a_x = Ka.mean(1)
    G = Kt - a_t[:, None] - a_t[None, :] + c           # macierz Grama odchyleń referencji
    lam, U = np.linalg.eigh(G); o = np.argsort(lam)[::-1]; lam, U = np.clip(lam[o], 0, None), U[:, o]
    nu = lam / len(train); k = min(k_modes, int((nu > 1e-12).sum()))
    Gx = Ka - a_x[:, None] - a_t[None, :] + c
    q = Gx @ U[:, :k] / np.sqrt(lam[:k])[None, :]
    z2 = 1.0 - 2.0 * a_x + c                           # ||Z_x||²_HS
    return (q ** 2 / (nu[:k] + eta)).sum(1), z2 - (q ** 2).sum(1)


def one_class_measures(P, train):
    rho = density_matrix(P[train]); l, V = spectral_full(rho)
    F0 = topk_overlap(P, V, PROJECTOR_RANK)
    H = V @ np.diag(-np.log(l + ENERGY_EPS)) @ V.conj().T
    E = np.real(np.einsum("ni,ij,nj->n", P.conj(), H, P))
    T2, Q = hs_deviation_measures(P, train)
    return dict(F0=F0, a0=1 - F0, E=E, T2=T2, Q=Q, lam0=l[0], gap=float(np.log(l[0]) - np.log(l[1])))


def two_class_measures(P, idx_d, idx_h):
    rd, rh = density_matrix(P[idx_d]), density_matrix(P[idx_h])
    (ld, Vd), (lh, Vh) = spectral_full(rd), spectral_full(rh)
    Fd, Fh = topk_overlap(P, Vd, PROJECTOR_RANK), topk_overlap(P, Vh, PROJECTOR_RANK)
    bd = np.real(np.einsum("ni,ij,nj->n", P.conj(), rd, P)); bh = np.real(np.einsum("ni,ij,nj->n", P.conj(), rh, P))
    Ahh, Add, Ahd = [float(np.real(np.trace(a @ b))) for a, b in [(rh, rh), (rd, rd), (rh, rd)]]
    a2 = Ahh + Add - 2 * Ahd; b1 = -2 * (bh - bd) - 2 * Add + 2 * Ahd
    w = np.clip(-b1 / (2 * a2), 0, 1) if a2 > 1e-12 else np.zeros(len(P))
    a_rec = 1 - 2 * (w * bh + (1 - w) * bd) + w ** 2 * Ahh + (1 - w) ** 2 * Add + 2 * w * (1 - w) * Ahd
    return dict(F_dysk=Fd, F_halo=Fh, s=Fh - Fd, m=np.abs(Fh - Fd), a_open=1 - np.maximum(Fh, Fd), w_halo=w, a_rec=a_rec,
                overlap=float(abs(np.vdot(Vd[:, 0], Vh[:, 0])) ** 2), gap=0.5 * (np.log(ld[0] / ld[1]) + np.log(lh[0] / lh[1])))


# ============================================================================
# ==================== 4. FOLDY, CV I METODY KLASYCZNE ========================
# ============================================================================

def make_folds(df):
    """Lista foldów: słownik {klasa: indeksy treningowe}. Dla 'jedna_klasa_cala' – jeden 'fold' = cała próba."""
    disk = np.where(df["klasa_kin"] == "dysk")[0]; halo = np.where(df["klasa_kin"] == "halo")[0]
    if MODE == "jedna_klasa_cala":
        return [{"ref": np.arange(len(df))}], np.arange(len(df))
    folds = []
    for rep in range(CV_REPEATS):
        rng = np.random.default_rng(CV_SEED + rep)
        pd_ = np.array_split(rng.permutation(disk), CV_FOLDS)
        ph_ = np.array_split(rng.permutation(halo), CV_FOLDS)
        for k in range(CV_FOLDS):
            tr_d = np.setdiff1d(disk, pd_[k]); tr_h = np.setdiff1d(halo, ph_[k])
            folds.append({"ref": tr_d} if MODE == "jedna_klasa_dysk" else {"dysk": tr_d, "halo": tr_h})
    refs = disk if MODE == "jedna_klasa_dysk" else np.r_[disk, halo]
    return folds, refs


def run_qift(S, pairs, setting, angles_saved, folds, df):
    """Uśrednia miary dla każdej gwiazdy po foldach, w których nie była w treningu."""
    acc, cnt, fold_angles = {}, np.zeros(len(S)), []
    orig_start = [np.full(len(pairs), np.deg2rad(INITIAL_ANGLE_DEG))]
    for f in folds:
        train_all = np.unique(np.concatenate(list(f.values())))
        if setting == "θ = 0":
            th = np.zeros(len(pairs))
        elif setting == "oryginalna":
            th = ANGLES_ORIG
        elif MODE == "jedna_klasa_cala" or not RELEARN_TOPOLOGY_IN_CV:
            th = angles_saved
        else:
            starts = orig_start + ([ANGLES_ORIG] if ANGLES_ORIG is not None else [])
            th = learn_gap_topology(S, pairs, list(f.values()), starts)
        fold_angles.append(np.rad2deg(th))
        P = encode_states(S, pairs, th)
        m = one_class_measures(P, f["ref"]) if MODE != "dwie_klasy" else two_class_measures(P, f["dysk"], f["halo"])
        mask = np.ones(len(S), bool)
        if MODE != "jedna_klasa_cala":
            mask[train_all] = False
        for k, v in m.items():
            if np.ndim(v) == 1:
                acc.setdefault(k, np.zeros(len(S)))[mask] += v[mask]
        cnt[mask] += 1
    out = {k: v / np.maximum(cnt, 1) for k, v in acc.items()}
    return out, np.array(fold_angles)


def run_classical(X, folds):
    try:
        from sklearn.covariance import MinCovDet
        from sklearn.ensemble import IsolationForest
        from sklearn.linear_model import LogisticRegression
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        from sklearn.neighbors import LocalOutlierFactor
    except Exception:
        print("  ⚠ brak scikit-learn – metody klasyczne pominięte (zostaje −z_Fe)")
        return {}
    acc, cnt = {}, np.zeros(len(X))
    for f in folds:
        train_all = np.unique(np.concatenate(list(f.values())))
        mask = np.ones(len(X), bool)
        if MODE != "jedna_klasa_cala":
            mask[train_all] = False
        out = {}
        if MODE != "dwie_klasy":
            Xt = X[f["ref"]]
            out["Mahalanobis (MCD)"] = MinCovDet(random_state=0).fit(Xt).mahalanobis(X)
            mu, sd = Xt.mean(0), Xt.std(0); Zt = (Xt - mu) / sd; Z = (X - mu) / sd
            _, s, Vt = np.linalg.svd(Zt - Zt.mean(0), full_matrices=False); kc = max(1, X.shape[1] // 2)
            var = s ** 2 / len(Zt); t = (Z - Zt.mean(0)) @ Vt[:kc].T
            out["PCA-T²"] = (t ** 2 / var[:kc]).sum(1)
            out["PCA-Q"] = ((Z - Zt.mean(0)) ** 2).sum(1) - (t ** 2).sum(1)
            out["Isolation Forest"] = -IsolationForest(n_estimators=300, random_state=0).fit(Xt).score_samples(X)
            Zs = (X - mu) / sd                         # LOF: anomalie lokalne / relacyjne (gęstość sąsiedztwa)
            out["LOF"] = -LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Zt).score_samples(Zs)
        else:
            tr = np.r_[f["dysk"], f["halo"]]; yt = np.r_[np.zeros(len(f["dysk"]), bool), np.ones(len(f["halo"]), bool)]
            out["regresja logistyczna"] = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[tr], yt).decision_function(X)
            out["QDA"] = QuadraticDiscriminantAnalysis(reg_param=0.05, priors=[0.5, 0.5]).fit(X[tr], yt).decision_function(X)
        for k, v in out.items():
            acc.setdefault(k, np.zeros(len(X)))[mask] += v[mask]
        cnt[mask] += 1
    return {k: v / np.maximum(cnt, 1) for k, v in acc.items()}


# ============================================================================
# ======================= 5. DECYZJE I METRYKI ================================
# ============================================================================

def calib_threshold(score, refs, q):
    return float(np.quantile(score[refs], q))


def decision_metrics(y, dec) -> dict:
    tp = int((dec & y).sum()); fp = int((dec & ~y).sum()); fn = int((~dec & y).sum()); tn = int((~dec & ~y).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan; rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else np.nan
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(n_oznaczonych=int(dec.sum()), TP=tp, FP=fp, FN=fn, precyzja=prec, czulosc=rec, F1=f1,
                MCC=(tp * tn - fp * fn) / den if den else np.nan)


def decision_metrics_ci(y, dec, B=BOOTSTRAP, seed=1) -> dict:
    m = decision_metrics(y, dec); rng = np.random.default_rng(seed); mc = []
    for _ in range(B):
        j = rng.integers(0, len(y), len(y)); mc.append(decision_metrics(y[j], dec[j])["MCC"])
    mc = np.array(mc, float)
    m["MCC_lo"], m["MCC_hi"] = np.nanpercentile(mc, [2.5, 97.5]) if np.isfinite(mc).any() else (np.nan, np.nan)
    return m


def precision_at(y, s, n=TOP_N) -> float:
    return float(y[np.argsort(-s)[:n]].mean())


def paired_delta(y, s1, s2, B=BOOTSTRAP, seed=2) -> dict:
    rng = np.random.default_rng(seed); da, dp = [], []
    for _ in range(B):
        j = rng.integers(0, len(y), len(y))
        if 0 < y[j].sum() < len(j):
            da.append(auc_score(y[j], s1[j]) - auc_score(y[j], s2[j])); dp.append(ap_score(y[j], s1[j]) - ap_score(y[j], s2[j]))
    da, dp = np.array(da), np.array(dp)
    return dict(dAUC=auc_score(y, s1) - auc_score(y, s2), dAUC_lo=np.percentile(da, 2.5), dAUC_hi=np.percentile(da, 97.5),
                dAP=ap_score(y, s1) - ap_score(y, s2), dAP_lo=np.percentile(dp, 2.5), dAP_hi=np.percentile(dp, 97.5),
                P_dAUC_le0=float(np.mean(da <= 0)))


# ============================================================================
# =============== 6. SPLĄTANIE (TYLKO WERSJE Z KUBITAMI ATMOSFERY) ==============
# ============================================================================

def cliffs_delta(a, b) -> float:
    if len(a) == 0 or len(b) == 0 or np.ptp(np.r_[a, b]) < 1e-9:
        return np.nan
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def ksg_mi(A, B, k=KSG_K) -> float:
    """Estymator Kraskova–Stögbauera–Grassbergera (I) informacji wzajemnej [bit]."""
    from scipy.spatial import cKDTree
    from scipy.special import digamma
    A = (A - A.mean(0)) / A.std(0); B = (B - B.mean(0)) / B.std(0); J = np.c_[A, B]; n = len(J)
    if n <= k + 1:
        return np.nan
    eps = cKDTree(J).query(J, k + 1, p=np.inf)[0][:, -1]
    ta, tb = cKDTree(A), cKDTree(B)
    na = np.array([len(ta.query_ball_point(A[i], eps[i] - 1e-12, p=np.inf)) - 1 for i in range(n)])
    nb = np.array([len(tb.query_ball_point(B[i], eps[i] - 1e-12, p=np.inf)) - 1 for i in range(n)])
    return float((digamma(k) + digamma(n) - np.mean(digamma(na + 1) + digamma(nb + 1))) / np.log(2))


def entanglement_study(S, pairs, encodings: dict, groups: dict, comparisons: list, df, X, out_dir: Path):
    print("=" * 80)
    print("KROK 4.6: SPLĄTANIE ATMOSFERA–CHEMIA")
    print("=" * 80)
    rng = np.random.default_rng(0); rows_g, rows_c = [], []
    logg = df["logg"].to_numpy()
    for enc, th in encodings.items():
        P = encode_states(S, pairs, th); sA = per_star_entropy(P)
        for g, msk in groups.items():
            idx = np.where(msk)[0]
            if len(idx) < MIN_GROUP:
                continue
            e = ensemble_measures(P[idx]); bs = [ensemble_measures(P[rng.choice(idx, len(idx))])["MI"] for _ in range(ENT_BOOT)]
            sh = []
            for _ in range(ENT_SHUFFLE):
                Z = S[idx].copy(); Z[:, 2:] = Z[rng.permutation(len(idx)), 2:]; sh.append(ensemble_measures(encode_states(Z, pairs, th))["MI"])
            rows_g.append(dict(kodowanie=enc, grupa=g, n=len(idx), S_mediana=np.median(sA[idx]), S_q16=np.quantile(sA[idx], .16),
                               S_q84=np.quantile(sA[idx], .84), MI=e["MI"], MI_lo=np.percentile(bs, 2.5), MI_hi=np.percentile(bs, 97.5),
                               MI_przetasowane=np.mean(sh), logneg=e["logneg"], olbrzymy=float((logg[idx] < 3.5).mean())))
        for g1, g2 in comparisons:
            i1, i2 = np.where(groups[g1])[0], np.where(groups[g2])[0]
            if len(i1) < MIN_GROUP or len(i2) < MIN_GROUP:
                continue
            mi1, mi2 = ensemble_measures(P[i1])["MI"], ensemble_measures(P[i2])["MI"]
            pool = np.r_[i1, i2]; perm = []
            for _ in range(ENT_PERM):
                pp = rng.permutation(pool); perm.append(ensemble_measures(P[pp[:len(i1)]])["MI"] - ensemble_measures(P[pp[len(i1):]])["MI"])
            r = dict(kodowanie=enc, porownanie=f"{g1} vs {g2}", n1=len(i1), n2=len(i2),
                     S_med1=np.median(sA[i1]), S_med2=np.median(sA[i2]),
                     p_MW_S=stats.mannwhitneyu(sA[i1], sA[i2]).pvalue if np.ptp(np.r_[sA[i1], sA[i2]]) > 1e-9 else np.nan,
                     delta_Cliffa_S=cliffs_delta(sA[i1], sA[i2]), dMI=mi1 - mi2,
                     p_perm_dMI=float(np.mean(np.abs(perm) >= abs(mi1 - mi2))))
            for nm, st in [("karly", logg >= 4.0), ("olbrzymy", logg < 3.5)]:
                a, b = sA[np.intersect1d(i1, np.where(st)[0])], sA[np.intersect1d(i2, np.where(st)[0])]
                r[f"n_{nm}"] = f"{len(a)}/{len(b)}"
                r[f"p_MW_S_{nm}"] = (stats.mannwhitneyu(a, b).pvalue if len(a) >= MIN_GROUP and len(b) >= MIN_GROUP
                                     and np.ptp(np.r_[a, b]) > 1e-9 else np.nan)
                r[f"delta_Cliffa_{nm}"] = cliffs_delta(a, b) if len(a) >= MIN_GROUP and len(b) >= MIN_GROUP else np.nan
            rows_c.append(r)
    G, C = pd.DataFrame(rows_g), pd.DataFrame(rows_c)
    cl = []
    for g, msk in groups.items():
        idx = np.where(msk)[0]
        if len(idx) >= max(MIN_GROUP, KSG_K + 3):
            n_sub = KSG_SUBSAMPLE
            if len(idx) > n_sub:
                sub = [ksg_mi(X[j, :2], X[j, 2:]) for j in (rng.choice(idx, n_sub, replace=False) for _ in range(30))]
                ksg_eq = float(np.nanmean(sub))
            else:
                ksg_eq = ksg_mi(X[idx, :2], X[idx, 2:]) if len(idx) == n_sub else np.nan
            cl.append(dict(grupa=g, n=len(idx), MI_gauss=gauss_mi(X[idx]), MI_KSG=ksg_mi(X[idx, :2], X[idx, 2:]),
                           **{f"MI_KSG_n{n_sub}": ksg_eq}))
    CL = pd.DataFrame(cl)
    G.to_csv(out_dir / "splatanie_grupy.csv", index=False); C.to_csv(out_dir / "splatanie_porownania.csv", index=False)
    CL.to_csv(out_dir / "splatanie_klasyczna_MI.csv", index=False)
    write_report(out_dir / "splatanie_raport.txt", f"SPLĄTANIE ATMOSFERA–CHEMIA – {VARIANT}", [
        ("Definicje", [f"  A = q0,q1 ({FEATURES[0]}, {FEATURES[1]}), B = q2..q{len(FEATURES) - 1} ({', '.join(FEATURES[2:])})",
                       "  S: entropia splątania A|B stanu gwiazdy [bit, maks. 2]; MI: informacja wzajemna zespołu ρ_grupy;",
                       "  MI_przetasowane: po losowym sparowaniu chemii z atmosferą w grupie (≈ MI -> efekt kodowania);",
                       "  p_perm_dMI: test permutacyjny różnicy MI między grupami; delta Cliffa ∈ [−1, 1];",
                       "  porównania w obrębie karłów (log g ≥ 4) i olbrzymów (log g < 3.5) kontrolują stadium ewolucyjne.",
                       "  Przy θ = 0 stany są produktowe: S ≡ 0 i log-negatywność ≡ 0 z definicji."]),
        ("Grupy", df_lines(G, 3, index=False)),
        ("Porównania", df_lines(C, 3, index=False)),
        ("Klasyczna informacja wzajemna atmosfera–chemia (latenty) [bit]", df_lines(CL, 3, index=False) + [
            f"  MI_KSG_n{KSG_SUBSAMPLE}: średnia z 30 podprób po {KSG_SUBSAMPLE} gwiazd – porównywalna między grupami",
            "  (estymator KSG jest obciążony zależnie od liczności; MI_gauss mierzy tylko zależność liniową)"]),
    ])
    encs = list(encodings); fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    gnames = [g for g in groups if groups[g].sum() >= MIN_GROUP]; cols = plt.cm.tab10(np.arange(len(gnames)))
    for i, enc in enumerate(encs):
        P = encode_states(S, pairs, encodings[enc]); sA = per_star_entropy(P)
        for j, g in enumerate(gnames):
            pos = i + (j - (len(gnames) - 1) / 2) * 0.8 / len(gnames)
            axes[0].boxplot(sA[groups[g]], positions=[pos], widths=0.6 / len(gnames), patch_artist=True, showfliers=False,
                            boxprops=dict(facecolor=cols[j], alpha=.55), medianprops=dict(color="k"))
            r = G[(G.kodowanie == enc) & (G.grupa == g)]
            if len(r):
                r = r.iloc[0]
                axes[1].errorbar(pos, r.MI, yerr=[[r.MI - r.MI_lo], [r.MI_hi - r.MI]], fmt="o", c=cols[j], label=g if i == 0 else None)
                axes[1].scatter(pos, r.MI_przetasowane, marker="x", c="k", s=20, label="chemia przetasowana" if i == 0 and j == 0 else None)
    for ax in axes:
        ax.set_xticks(range(len(encs)), encs); ax.grid(alpha=.25, axis="y")
    axes[0].set_ylabel("S(A|B) gwiazdy [bit]"); axes[0].set_title("Splątanie pojedynczych gwiazd w grupach")
    axes[1].set_ylabel("I(A:B) zespołu [bit]"); axes[1].set_title("Informacja wzajemna zespołów"); axes[1].legend(fontsize=7)
    fig.tight_layout(); save_fig(fig, out_dir / "splatanie_grupy.png")
    return G, C, CL


# ============================================================================
# ======================= URUCHOMIENIE PIPELINE'U =============================
# ============================================================================

ANGLES_ORIG = None


def run_pipeline():
    global ANGLES_ORIG
    ensure_dir(QIFT_REPORTS_DIR)
    st, df = load_states(STATES_NPZ, DATA_CSV)
    if str(st["mode"]) != MODE:
        raise ValueError(f"Stany z kroku 03 mają tryb {st['mode']}, oczekiwano {MODE} – uruchom 03_hilbert_topology_{VARIANT}.py")
    df["row_id"] = df["row_id"].astype(int)
    S, pairs, feats = st["S"], st["pairs"], st["features"]
    X = df[feats].to_numpy(float)
    ANGLES_ORIG = st["angles_original"] if np.size(st["angles_original"]) == len(pairs) else None
    ev = df["eval_mask"].to_numpy(bool); y = df["halo_kin"].to_numpy(bool)
    folds, refs = make_folds(df)
    settings = ["wyuczona (przerwa masowa)", "θ = 0"] + (["oryginalna"] if ANGLES_ORIG is not None else [])
    print("=" * 80)
    print(f"KROK 4.2: MIARY QIFT ({MODE}; foldów: {len(folds)}; ponowne uczenie topologii: {RELEARN_TOPOLOGY_IN_CV})")
    print("=" * 80)
    Q, FA = {}, {}
    for s_ in settings:
        t0 = time.time(); Q[s_], FA[s_] = run_qift(S, pairs, s_, np.asarray(st["angles"]), folds, df)
        print(f"  ✓ {s_}: {time.time() - t0:.0f} s")
    C = run_classical(X, folds); C["−z_Fe"] = -df["z_Fe"].to_numpy(float)

    # ----- wyniki ciągłe i decyzje -----
    res = df.copy(); main = settings[0]
    ref_med_fe = float(np.median(X[refs, feats.index("z_Fe")]))
    low_fe = df["z_Fe"].to_numpy() < ref_med_fe
    scores, decisions = {}, {}
    for s_ in settings:
        q = Q[s_]; tag = {"wyuczona (przerwa masowa)": "wyucz", "θ = 0": "th0", "oryginalna": "oryg"}[s_]
        for k, v in q.items():
            res[f"{k}__{tag}"] = v
        if MODE != "dwie_klasy":
            scores[f"QIFT a0 [{s_}]"] = q["a0"]; scores[f"QIFT T² [{s_}]"] = q["T2"]
            scores[f"QIFT Q [{s_}]"] = q["Q"]; scores[f"QIFT E [{s_}]"] = q["E"]
            scores[f"QIFT a0 kierunkowe [{s_}]"] = q["a0"] + low_fe
            if s_ == main:
                decisions["QIFT reguła oryginalna (F0≤q, E≥q)"] = (q["F0"] <= np.quantile(q["F0"], ANOMALY_F_QUANTILE)) & (q["E"] >= np.quantile(q["E"], ANOMALY_E_QUANTILE))
                for nm in ["a0", "T2"]:
                    thr = calib_threshold(q[nm], refs, 1 - ALPHA)
                    decisions[f"QIFT {nm} > próg (α={ALPHA})"] = q[nm] > thr
                    decisions[f"QIFT {nm} > próg, kierunkowe"] = (q[nm] > thr) & low_fe
        else:
            scores[f"QIFT s = F_halo − F_dysk [{s_}]"] = q["s"]; scores[f"QIFT waga halo w [{s_}]"] = q["w_halo"]
            if s_ == main:
                thr_open = calib_threshold(q["a_open"], refs, 1 - ALPHA); thr_m = calib_threshold(q["m"], refs, MARGIN_QUANTILE)
                novelty = q["a_open"] > thr_open; inter = ~novelty & (q["m"] < thr_m)
                cat = np.where(novelty, "nowość", np.where(inter, "pośrednia", np.where(q["s"] > 0, "halo", "dysk")))
                res["kategoria_QIFT"] = cat
                decisions["QIFT s > 0 (F_halo > F_dysk)"] = q["s"] > 0
                decisions["QIFT kategoria = halo (4 grupy)"] = cat == "halo"
    for k, v in C.items():
        scores[k] = v; res[f"klas__{k}"] = v
        if k != "−z_Fe":
            decisions[f"{k} > próg" if MODE != "dwie_klasy" else f"{k} > 0"] = (v > calib_threshold(v, refs, 1 - ALPHA)) if MODE != "dwie_klasy" else (v > 0)
    for k, v in decisions.items():
        res[f"dec__{k}"] = v

    print("=" * 80)
    print("KROK 4.3: METRYKI")
    print("=" * 80)
    MR = pd.DataFrame([dict(metoda=k, **metrics_ci(y[ev], s[ev]), **{f"precyzja@{TOP_N}": precision_at(y[ev], s[ev])})
                       for k, s in scores.items()])
    MD = pd.DataFrame([dict(decyzja=k, **decision_metrics_ci(y[ev], d[ev]),
                            olbrzymy=float((df.logg[d] < 3.5).mean()) if d.any() else np.nan,
                            FeH_lt_m1=float((df.feh[d] < -1).mean()) if d.any() else np.nan) for k, d in decisions.items()])
    key = "QIFT a0" if MODE != "dwie_klasy" else "QIFT s = F_halo − F_dysk"
    comps = [(f"{key} [{main}]", f"{key} [θ = 0]")]
    if ANGLES_ORIG is not None:
        comps += [(f"{key} [{main}]", f"{key} [oryginalna]"), (f"{key} [oryginalna]", f"{key} [θ = 0]")]
    if MODE != "dwie_klasy":
        comps += [(f"QIFT T² [{main}]", f"QIFT T² [θ = 0]"), (f"QIFT T² [{main}]", f"QIFT a0 [{main}]")]
        comps += [(f"{key} [{main}]", c) for c in ["Mahalanobis (MCD)", "PCA-T²", "Isolation Forest"] if c in scores]
        comps += [(f"QIFT T² [{main}]", c) for c in ["Mahalanobis (MCD)", "PCA-T²", "Isolation Forest", "LOF", "−z_Fe"] if c in scores]
    else:
        comps += [(f"{key} [{main}]", c) for c in ["regresja logistyczna", "QDA"] if c in scores]
    comps += [(f"{key} [{main}]", "−z_Fe")]
    PD = pd.DataFrame([dict(porownanie=f"{a}  −  {b}", **paired_delta(y[ev], scores[a][ev], scores[b][ev])) for a, b in comps])
    MR.to_csv(QIFT_REPORTS_DIR / "metryki_ranking.csv", index=False); MD.to_csv(QIFT_REPORTS_DIR / "metryki_decyzje.csv", index=False)
    PD.to_csv(QIFT_REPORTS_DIR / "porownania_parowane.csv", index=False)

    ang_lines = [f"  kąty z kroku 03 (cała referencja) [°]: {np.round(np.rad2deg(st['angles']), 1).tolist()}"]
    if ANGLES_ORIG is not None:
        ang_lines.append(f"  kąty oryginalne [°]: {np.round(np.rad2deg(ANGLES_ORIG), 1).tolist()}")
    if len(FA[main]) > 1:
        fa = FA[main]; ang_lines += [f"  kąty uczone w foldach [°]: średnia {np.round(fa.mean(0), 1).tolist()}",
                                     f"                            odch. std {np.round(fa.std(0), 1).tolist()}"]
    comp_lines = []
    for p in POP_ORDER:
        sel = df["populacja_toomre"] == p
        if sel.any():
            comp_lines.append(f"  {p:20s} n={int(sel.sum()):4d} | " + " | ".join(
                f"{k[:28]}: {d[sel].mean():.2f}" for k, d in list(decisions.items())[:3]))
    extra = []
    if MODE == "dwie_klasy":
        ct = pd.crosstab(res["populacja_toomre"], res["kategoria_QIFT"]).reindex([p for p in POP_ORDER if p in set(res.populacja_toomre)])
        ct.to_csv(QIFT_REPORTS_DIR / "kategorie_wg_populacji.csv"); extra = [("Kategorie QIFT wg populacji Toomre'a", df_lines(ct, 3))]
    write_report(QIFT_REPORTS_DIR / "raport_glowny.txt", f"QIFT – PODEJŚCIE {VARIANT} ({MODE})", [
        ("Ustawienia", [f"  kubity: {feats}", f"  referencja: {MODE}; gwiazd referencyjnych: {len(refs)}; foldów: {len(folds)}",
                        f"  ranga projektora k = {PROJECTOR_RANK}; mody odchyleń (T², Q): {K_MODES}; α = {ALPHA}",
                        f"  ewaluacja: {int(ev.sum())} gwiazd z dobrą astrometrią, halo kin. (v_tot > {HALO_VTOT:.0f}): {int((y & ev).sum())}",
                        f"  AP losowego rankingu = {y[ev].mean():.3f}",
                        f"  kierunkowe: tylko gwiazdy z z_Fe < mediany referencji ({ref_med_fe:+.3f})"] + ang_lines),
        ("Ranking (AUC, AP z bootstrap 95%, precyzja@N)", df_lines(MR, 3, index=False)),
        ("Decyzje (precyzja, czułość, F1, MCC z bootstrap 95%)", df_lines(MD, 3, index=False)),
        ("Porównania parowane (bootstrap; a − b)", df_lines(PD, 3, index=False)),
        ("Odsetek oznaczonych wg populacji Toomre'a (pierwsze 3 decyzje)", comp_lines)] + extra)

    # ----- wykresy -----
    print("=" * 80)
    print("KROK 4.4: WYKRESY")
    print("=" * 80)
    mains = [f"{key} [{s_}]" for s_ in settings] + [k for k in C if k in scores]
    plot_roc_pr(y[ev], {k: scores[k][ev] for k in mains}, QIFT_REPORTS_DIR / "01_roc_pr.png", f"halo kinematyczne, {VARIANT}")
    fig, axes = plt.subplots(1, 2, figsize=(15, max(4, 0.45 * len(MR) + 1.5)), sharey=True)
    for ax, met in zip(axes, ["AUC", "AP"]):
        yy = np.arange(len(MR))
        ax.errorbar(MR[met], yy, xerr=[MR[met] - MR[f"{met}_lo"], MR[f"{met}_hi"] - MR[met]], fmt="o", c="k", ms=4)
        ax.set_xlabel(met); ax.grid(alpha=.25, axis="x")
    axes[0].set_yticks(np.arange(len(MR)), MR.metoda, fontsize=7); axes[0].invert_yaxis()
    axes[1].axvline(y[ev].mean(), c="gray", ls=":", label="losowo"); axes[1].legend(fontsize=8)
    fig.suptitle(f"Wyniki wszystkich miar – {VARIANT}", fontweight="bold"); fig.tight_layout()
    save_fig(fig, QIFT_REPORTS_DIR / "02_metryki_wszystkie.png")
    dkey = f"QIFT T2 > próg (α={ALPHA})" if MODE != "dwie_klasy" else "QIFT s > 0 (F_halo > F_dysk)"
    dmain = decisions[dkey]
    plot_toomre(res, dmain, "oznaczone QIFT", QIFT_REPORTS_DIR / "03_toomre.png", dkey,
                ref_mask=(res.klasa_kin == "halo").to_numpy(), ref_label="halo kin.")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    if MODE == "dwie_klasy":
        colmap = {"dysk": "steelblue", "halo": "crimson", "pośrednia": "orange", "nowość": "k"}
        for c_, col in colmap.items():
            mm = res.kategoria_QIFT == c_
            axes[0].scatter(res.teff[mm], res.logg[mm], s=8 if c_ == "dysk" else 18, c=col, label=f"{c_} ({mm.sum()})", alpha=.6 if c_ == "dysk" else .9)
            axes[1].scatter(res.feh[mm], res.alpha_m[mm], s=8 if c_ == "dysk" else 18, c=col, alpha=.6 if c_ == "dysk" else .9)
    else:
        axes[0].scatter(res.teff, res.logg, s=6, c="lightgray")
        axes[0].scatter(res.teff[dmain], res.logg[dmain], s=16, c="crimson", label=f"oznaczone ({dmain.sum()})")
        axes[1].scatter(res.feh, res.alpha_m, s=6, c="lightgray"); axes[1].scatter(res.feh[dmain], res.alpha_m[dmain], s=16, c="crimson")
    hk = (res.klasa_kin == "halo").to_numpy()
    axes[0].scatter(res.teff[hk], res.logg[hk], s=45, facecolors="none", edgecolors="k", lw=.7, label="halo kin.")
    axes[1].scatter(res.feh[hk], res.alpha_m[hk], s=45, facecolors="none", edgecolors="k", lw=.7)
    axes[0].invert_xaxis(); axes[0].invert_yaxis(); axes[0].set(xlabel="Teff [K]", ylabel="log g", title="Diagram Kiela")
    axes[1].set(xlabel="[Fe/H]", ylabel="[α/M]", title="Chemia"); axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout(); save_fig(fig, QIFT_REPORTS_DIR / "04_kiel_chemia.png")
    if MODE != "dwie_klasy":
        q = Q[main]; fig, ax = plt.subplots(figsize=(8, 6.5))
        for p, col in [("<50 cienki_dysk", "steelblue"), ("70-180 gruby_dysk", "orange"), (">220 halo", "crimson")]:
            mm = (res.populacja_toomre == p).to_numpy()
            ax.scatter(q["T2"][mm], q["Q"][mm], s=10 if p != ">220 halo" else 25, c=col, alpha=.6, label=p)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("T² (skrajność w modach)"); ax.set_ylabel("Q (reszta poza modami)")
        ax.legend(); ax.grid(alpha=.25); ax.set_title("Typ nietypowości: T² vs Q (miary 5–6)")
        save_fig(fig, QIFT_REPORTS_DIR / "05_T2_vs_Q.png")
    else:
        q = Q[main]; e = np.where(ev)[0]; order = e[np.argsort(-q["m"][e])]
        cov, mcc, prec = [], [], []
        for c_ in np.linspace(0.2, 1, 33):
            sel = order[:max(10, int(c_ * len(order)))]; dm = decision_metrics(y[sel], q["s"][sel] > 0)
            cov.append(c_); mcc.append(dm["MCC"]); prec.append(dm["precyzja"])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(cov, mcc, "o-", label="MCC"); ax.plot(cov, prec, "s-", label="precyzja (halo)")
        ax.set_xlabel("pokrycie (odsetek gwiazd z największym marginesem)"); ax.set_ylabel("wartość"); ax.legend(); ax.grid(alpha=.25)
        ax.set_title("Ryzyko–pokrycie: odkładanie gwiazd o małym marginesie"); save_fig(fig, QIFT_REPORTS_DIR / "05_ryzyko_pokrycie.png")
    fig, ax = plt.subplots(figsize=(10, 5)); sm = scores[f"{key} [{main}]"]
    bins = np.linspace(np.percentile(sm, 1), np.percentile(sm, 99.5), 50)
    for p in POP_ORDER[:-1]:
        v = sm[(res.populacja_toomre == p).to_numpy()]
        if len(v):
            ax.hist(v, bins=bins, histtype="step", lw=1.6, density=True, label=f"{p} (n={len(v)})")
    ax.set_xlabel(f"{key} [{main}]"); ax.set_ylabel("gęstość"); ax.legend(fontsize=8); ax.grid(alpha=.25)
    ax.set_title("Rozkład głównej miary w populacjach Toomre'a"); save_fig(fig, QIFT_REPORTS_DIR / "06_miara_wg_populacji.png")

    # ----- splątanie -----
    if HAS_ATM:
        D_ = (res.klasa_kin == "dysk").to_numpy(); H_ = (res.klasa_kin == "halo").to_numpy()
        if MODE != "dwie_klasy":
            dd = decisions[f"QIFT T2 > próg (α={ALPHA})"]          # główna decyzja (w v1 skryptów: a0)
            groups = {"dysk-normalne": D_ & ~dd, "dysk-anomalie": D_ & dd, "halo-wykryte": H_ & dd,
                      "halo-przeoczone": H_ & ~dd, "dysk (kin.)": D_, "halo (kin.)": H_}
            comparisons = [("dysk-anomalie", "dysk-normalne"), ("halo (kin.)", "dysk (kin.)"), ("halo-wykryte", "dysk-anomalie")]
        else:
            cat = res["kategoria_QIFT"].to_numpy()
            groups = {"QIFT: dysk": cat == "dysk", "QIFT: halo": cat == "halo", "QIFT: pośrednia": cat == "pośrednia",
                      "QIFT: nowość": cat == "nowość", "dysk (kin.)": D_, "halo (kin.)": H_}
            comparisons = [("QIFT: halo", "QIFT: dysk"), ("QIFT: pośrednia", "QIFT: dysk"), ("halo (kin.)", "dysk (kin.)")]
        enc = {"wyuczona": np.asarray(st["angles"]), "θ = 0": np.zeros(len(pairs))}
        if ANGLES_ORIG is not None:
            enc["oryginalna"] = ANGLES_ORIG
        entanglement_study(S, pairs, enc, groups, comparisons, res, X, QIFT_REPORTS_DIR)

    res.to_csv(QIFT_REPORTS_DIR / "wyniki_gwiazd.csv", index=False)
    print(f"  ✓ {QIFT_REPORTS_DIR / 'wyniki_gwiazd.csv'}")
    print(f"\n✓ KONIEC KROKU 4 ({VARIANT}). Raporty w:", QIFT_REPORTS_DIR.resolve())
    return dict(results=res, ranking=MR, decisions=MD, paired=PD)


if __name__ == "__main__":
    results_04 = run_pipeline()
