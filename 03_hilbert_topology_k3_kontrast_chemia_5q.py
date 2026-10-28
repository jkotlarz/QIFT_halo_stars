# -*- coding: utf-8 -*-
"""
03_hilbert_topology_k3_kontrast_chemia_5q.py
======================================================================
KROK 3 potoku QIFT-spectra – WARIANT KONTRASTOWY K3: pełna chemia (Fe, α, C, N, aktywność)
  kubity: z_Fe, z_alpha, z_C, z_N, z_act

Idea
----
Dwie populacje: dysk (v_tot < 70 km/s) i halo (v_tot > 220 km/s), dobra astrometria Gaia.
Topologia RZZ jest uczona tak, by DYSK tworzył stan podstawowy (ρ_dysk), a gwiazdy HALO
były jak najsilniej nietypowe względem niego:
    maksymalizujemy gładkie AUC( a0_halo > a0_dysk ),  a0 = 1 − |⟨E0_dysk|ψ⟩|²,
z karą KONTRAST_REG·Σθ² (kąt rośnie tylko, jeśli poprawia rozdzielenie).

Co robi ten plik
----------------
1. Wczytuje dane (REF_CSV + qiftdata.csv z kroku 02), skaluje cechy min–max, klasy kinematyczne.
2. Porównuje każdą zmienną latentną w dysku i halo (delta Cliffa, d Cohena, testy MW i KS)
   i dzieli zmienne na „podobne” (|δ| < DIFF_DELTA) i „różne” (|δ| ≥ DIFF_DELTA).
3. Uczy topologię kryterium kontrastowym (starty: θ = 0, θ = 20°, losowe).
4. Sprawdza teorię: pary zmiennych podobnych → duże sprzężenie, pary z różną zmienną → małe
   (kąty wg typu pary, Spearman kąt ~ max|δ|, Mann–Whitney), ablacja pojedynczych par.
5. Zapisuje stany i topologię dla kroku 04 (tam topologia jest uczona od nowa w foldach CV).

UWAGA: kryterium używa etykiet halo – to wariant NADZOROWANY w uczeniu topologii
(miary nietypowości pozostają jednoklasowe względem dysku).

Uruchomienie: python 03_hilbert_topology_k3_kontrast_chemia_5q.py
"""

from __future__ import annotations

import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ============================================================================
# ======================= PARAMETRY URUCHOMIENIOWE ===========================
# ============================================================================

VARIANT = "k3_kontrast_chemia_5q"
MODE = "kontrast"

# --- Wejście ---
REF_CSV = Path("gwiazdy_qift_kinematyka.csv")
QIFTDATA_PATH = Path("astro_reports") / "qiftdata.csv"
LATENT_SOURCE = "qiftdata"

# --- Wyjście ---
HILBERT_DIR = Path("hilbertspace_topology") / VARIANT
TOPOLOGY_FILE = HILBERT_DIR / "best_topology.csv"
STATES_NPZ = HILBERT_DIR / "qift_states.npz"

# --- Cechy kodowane na kubity ---
FEATURES = ["z_Fe", "z_alpha", "z_C", "z_N", "z_act"]

# --- Klasy kinematyczne ---
DISK_VTOT_MAX = 70.0
HALO_VTOT_MIN = 220.0

# --- Podział zmiennych na podobne / różne ---
DIFF_DELTA = 0.40                 # |δ Cliffa| ≥ próg -> zmienna „różna” w halo i dysku

# --- Uczenie topologii: kryterium kontrastowe ---
TOPOLOGY_BOUNDS_DEG = (0.0, 90.0)
KONTRAST_REG = 2e-3               # kara λ·Σθ² [rad²]
KONTRAST_TAU_FRAC = 0.10          # szerokość sigmoidy gładkiego AUC (ułamek odch. std a0)
INITIAL_ANGLE_DEG = 20.0
N_RANDOM_STARTS = 2
STEP_SCHEDULE_DEG = [15.0, 8.0, 3.0, 1.0]
TRIALS_PER_STAGE = 80
BATCH_PER_TRIAL = 6
PATIENCE = 10
TOPOLOGY_SEED = 48

# --- ρ ---
RHO_EPS = 1e-6
ENERGY_EPS = 1e-12

# --- Diagnostyka ---
VERIFY_WITH_QISKIT = True
N_QISKIT_CHECK = 5
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


def basis_labels(n: int) -> list[str]:
    """Etykiety stanów bazy w konwencji Qiskit: |q_{n-1} ... q_1 q_0⟩."""
    return [format(b, f"0{n}b") for b in range(2 ** n)]


# ============================================================================
# ================ 1. DANE REFERENCYJNE + SYGNATURY I SKALOWANIE ==============
# ============================================================================

def load_reference_data(ref_csv: Path = REF_CSV, qiftdata_csv: Path = QIFTDATA_PATH,
                        latent_source: str = LATENT_SOURCE) -> pd.DataFrame:
    """
    Łączy dane referencyjne (REF_CSV, 1 wiersz = 1 gwiazda) z wynikami kroku 02
    (qiftdata.csv: latenty, sygnatury sig_*, recon_mse_*, row_id = indeks widma w NPZ).
    Klucz: REF.obsid_uzyte == qiftdata.obsid. Zwraca tabelę z nazwami kolumn
    zgodnymi z oryginalnym potokiem (obsid, subclass, rv_used_kms, ...).
    """
    print("=" * 80)
    print("KROK 3.1: DANE REFERENCYJNE (CSV) + SYGNATURY (KROK 02)")
    print("=" * 80)
    ref = pd.read_csv(ref_csv, dtype={"gaia_dr3_source_id": str})
    q = pd.read_csv(qiftdata_csv)
    lat = ["w_atm_1", "w_atm_2", "z_Fe", "z_alpha", "z_C", "z_N", "z_act"]
    sig = [c for c in q.columns if c.startswith("sig_") or c.startswith("recon_mse")]
    q_keep = q[["obsid", "row_id"] + [c for c in lat if c in q.columns] + sig].rename(
        columns={c: f"{c}__q" for c in lat})
    df = ref.merge(q_keep, left_on="obsid_uzyte", right_on="obsid", how="inner")
    lost = len(ref) - len(df)
    print(f"  REF: {len(ref)} gwiazd | qiftdata: {len(q)} widm | połączono: {len(df)}"
          + (f"  ⚠ {lost} gwiazd REF bez widma w qiftdata" if lost else ""))
    for c in lat:
        if f"{c}__q" in df.columns:
            if c not in df.columns:                      # nowe latenty są tylko w qiftdata
                df[c] = df[f"{c}__q"]
            elif latent_source == "qiftdata":
                d = (df[f"{c}__q"] - df[c]).abs().max()
                if d > 1e-3:
                    print(f"  ⚠ {c}: latenty z qiftdata różnią się od REF_CSV (max |Δ| = {d:.3g}) – używam qiftdata")
                df[c] = df[f"{c}__q"]
            df = df.drop(columns=f"{c}__q")
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Brak latentów {missing} – uruchom nowy krok 02 (02_latent_signatures.py, 7 latentów), "
                         f"który zapisuje je w {qiftdata_csv}")
    # nazwy zgodne z oryginalnym potokiem
    df["subclass"] = df["typ_widmowy"]
    df["rv_used_kms"] = df["rv_lamost_kms"]
    df["dobra_astrometria"] = df["dobra_astrometria"].astype(str).str.lower().isin(["true", "1"])
    df["halo_kin"] = df["dobra_astrometria"] & (df["v_tot_kms"] > 220)
    df = df.reset_index(drop=True)
    df.insert(0, "star_id", np.arange(len(df)))
    print(f"  latenty z: {latent_source} | sygnatur sig_*: {sum(c.startswith('sig_') for c in df.columns)}"
          f" | halo kinematyczne (v_tot>220, dobra astrometria): {int(df.halo_kin.sum())}")
    return df


def scale_features(df: pd.DataFrame, features=FEATURES):
    """Skalowanie min–max do [0, 1] (kolumny s_<cecha>)."""
    ok = df[features].notna().all(axis=1)
    if (~ok).any():
        print(f"  ⚠ usuwam {(~ok).sum()} wierszy z NaN w cechach")
    df = df[ok].reset_index(drop=True)
    fmin = df[features].min().to_numpy(float)
    fmax = df[features].max().to_numpy(float)
    S = (df[features].to_numpy(float) - fmin) / np.where(fmax - fmin == 0, 1, fmax - fmin)
    for k, f in enumerate(features):
        df[f"s_{f}"] = S[:, k]
    print(f"✓ {len(df)} gwiazd, {len(features)} kubitów, wymiar przestrzeni Hilberta = {2 ** len(features)}")
    return df, S, fmin, fmax


# ============================================================================
# =================== 2. KODOWANIE STANÓW (numpy / Qiskit) ====================
# ============================================================================

def make_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _apply_1q(Psi: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
    N = Psi.shape[0]
    T = Psi.reshape(N, 2 ** (n - 1 - q), 2, 2 ** q)
    T = np.einsum("nab,nxby->nxay", U, T)
    return T.reshape(N, 2 ** n)


def _ry_mats(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def zz_phase_diag(n: int, pairs, angles) -> np.ndarray:
    """Diagonala Π RZZ(θ_ij) = exp(−i θ/2 · z_i z_j), z = +1 dla bitu 0, −1 dla bitu 1."""
    idx = np.arange(2 ** n)
    z = 1 - 2 * ((idx[:, None] >> np.arange(n)[None, :]) & 1)
    phase = np.zeros(2 ** n)
    for (i, j), th in zip(pairs, angles):
        phase += -0.5 * th * z[:, i] * z[:, j]
    return np.exp(1j * phase)


def encode_states(S: np.ndarray, pairs, angles) -> np.ndarray:
    """Wsadowe kodowanie: Ry(x·π/2) – RZZ – Ry(x·π/2). Zwraca Psi [N, 2^n]."""
    N, n = S.shape
    ang = np.clip(S, 0, 1) * (np.pi / 2)
    Psi = np.zeros((N, 2 ** n), complex)
    Psi[:, 0] = 1.0
    for q in range(n):
        Psi = _apply_1q(Psi, _ry_mats(ang[:, q]).astype(complex), q, n)
    Psi = Psi * zz_phase_diag(n, pairs, angles)[None, :]
    for q in range(n):
        Psi = _apply_1q(Psi, _ry_mats(ang[:, q]).astype(complex), q, n)
    return Psi


def verify_with_qiskit(S, pairs, angles, n_check=N_QISKIT_CHECK) -> list[str]:
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except Exception as e:
        return [f"  Qiskit niedostępny ({e}) – weryfikacja pominięta"]
    Psi = encode_states(S[:n_check], pairs, angles)
    out = []
    for k in range(min(n_check, len(S))):
        n = S.shape[1]
        a = np.clip(S[k], 0, 1) * np.pi / 2
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.ry(float(a[q]), q)
        for (i, j), th in zip(pairs, angles):
            qc.rzz(float(th), i, j)
        for q in range(n):
            qc.ry(float(a[q]), q)
        sv = Statevector.from_instruction(qc).data
        out.append(f"  gwiazda {k}: |⟨ψ_qiskit|ψ_numpy⟩|² = {abs(np.vdot(sv, Psi[k])) ** 2:.12f}")
    return out



def density_matrix(Psi: np.ndarray) -> np.ndarray:
    rho = (Psi.T @ Psi.conj()) / Psi.shape[0]
    rho = 0.5 * (rho + rho.conj().T)
    mn = np.linalg.eigvalsh(rho).min()
    if mn <= 0:
        rho = rho + (RHO_EPS - mn if mn < 0 else RHO_EPS) * np.eye(rho.shape[0])
    return rho / np.trace(rho).real


def spectral(rho: np.ndarray) -> dict:
    lam, V = np.linalg.eigh(rho)
    o = np.argsort(lam)[::-1]
    lam, V = lam[o].real, V[:, o]
    E = -np.log(lam + ENERGY_EPS)
    return dict(eigvals=lam, eigvecs=V, energies=E, mass_gap=float(E[1] - E[0]),
                purity=float(np.real(np.trace(rho @ rho))),
                entropy=float(-np.sum(lam * np.log(lam + ENERGY_EPS))))



def plot_topology_graph(pairs, angles, features, path: Path, title="Topologia połączeń RZZ"):
    n = len(features)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    pos = {i: (np.cos(t), np.sin(t)) for i, t in enumerate(th)}

    def col(a):
        a = float(np.clip(a, 0, 90))
        c1, c2, t = (("black", "navy", a / 45) if a <= 45 else ("navy", "red", (a - 45) / 45))
        return tuple((1 - t) * np.array(to_rgb(c1)) + t * np.array(to_rgb(c2)))

    fig, ax = plt.subplots(figsize=(8, 8))
    for (i, j), a in zip(pairs, np.rad2deg(angles)):
        (x1, y1), (x2, y2) = pos[i], pos[j]
        ax.plot([x1, x2], [y1, y2], color=col(a), lw=0.8 + 9 * a / 90, alpha=0.9, zorder=1)
        t_lab = 0.30 if n > 2 else 0.5
        ax.text(x1 + t_lab * (x2 - x1), y1 + t_lab * (y2 - y1), f"{a:.1f}°", fontsize=9 if n <= 4 else 6.5, color=col(a), ha="center",
                va="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    for i, f in enumerate(features):
        ax.scatter(*pos[i], s=2200 if n <= 4 else 1500, color="white", edgecolors="black", lw=1.5, zorder=3)
        ax.text(*pos[i], f"q{i}\n{f}", ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)
    ax.set_title(title, fontweight="bold"); ax.axis("off"); ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    save_fig(fig, path)



# ============================================================================
# ======== KRYTERIUM KONTRASTOWE: DYSK = STAN PODSTAWOWY, HALO = ANOMALIE =====
# ============================================================================

def ground_state_vector(P_ref):
    """Wektor własny ρ_ref o największej wartości własnej (stan podstawowy)."""
    rho = density_matrix(P_ref)
    l, V = np.linalg.eigh(rho)
    return V[:, np.argmax(l)]


def soft_auc(neg, pos, tau):
    """Gładkie AUC: średnia sigmoid((pos_j − neg_i)/τ) po wszystkich parach (różniczkowalne przybliżenie AUC)."""
    z = (pos[None, :] - neg[:, None]) / max(tau, 1e-12)
    return float(np.mean(1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))))


def contrast_objective(S, pairs, th, idx_d, idx_h):
    """
    Kryterium kontrastowe: stan podstawowy ρ_dysk, nietypowość a0 = 1 − |⟨E0_dysk|ψ⟩|².
    Zwraca gładkie AUC(a0_halo > a0_dysk) – im wyższe, tym lepiej halo „odstaje” od dysku.
    """
    idx = np.r_[idx_d, idx_h]
    P = encode_states(S[idx], pairs, th)
    v = ground_state_vector(P[:len(idx_d)])
    a0 = 1.0 - np.abs(P @ v.conj()) ** 2
    ad, ah = a0[:len(idx_d)], a0[len(idx_d):]
    tau = KONTRAST_TAU_FRAC * float(np.std(np.r_[ad, ah]) + 1e-12)
    return soft_auc(ad, ah, tau)


def contrast_cost(th, S, pairs, idx_d, idx_h):
    lo, hi = np.deg2rad(TOPOLOGY_BOUNDS_DEG)
    th = np.clip(np.asarray(th, float), lo, hi)
    return -contrast_objective(S, pairs, th, idx_d, idx_h) + KONTRAST_REG * float(np.sum(th ** 2))


def learn_contrast_topology(S, pairs, idx_d, idx_h, starts, schedule, trials, batch, patience, seed=0,
                            verbose=False, names=None):
    """Hill-climbing kryterium kontrastowego z kilku startów. Zwraca (kąty, koszt, lista wyników startów)."""
    lo, hi = np.deg2rad(TOPOLOGY_BOUNDS_DEG)
    out = []
    for k, x0 in enumerate(starts):
        rng = np.random.default_rng(seed + k)
        x = np.clip(np.asarray(x0, float), lo, hi); c = contrast_cost(x, S, pairs, idx_d, idx_h)
        for step in schedule:
            no_imp = 0
            for _ in range(trials):
                imp = False
                for _ in range(batch):
                    d = rng.normal(size=len(x)); d /= max(np.linalg.norm(d), 1e-12)
                    xt = np.clip(x + rng.uniform(.3, 1) * np.deg2rad(step) * d, lo, hi)
                    ct = contrast_cost(xt, S, pairs, idx_d, idx_h)
                    if ct < c - 1e-12:
                        x, c, imp = xt, ct, True
                no_imp = 0 if imp else no_imp + 1
                if no_imp >= patience:
                    break
        out.append((c, x, names[k] if names else f"start {k}"))
        if verbose:
            print(f"  start: {out[-1][2]:22s} -> miękkie AUC {-c + KONTRAST_REG * np.sum(x ** 2):.4f}  "
                  f"kąty [°] {np.round(np.rad2deg(x), 1).tolist()}")
    out.sort(key=lambda r: r[0])
    return out[0][1], out[0][0], out


# ---------------- porównanie zmiennych w dwóch populacjach ----------------
def compare_populations(X_d, X_h, features) -> pd.DataFrame:
    """Dla każdej zmiennej: mediany, delta Cliffa (= 2·AUC − 1), d Cohena, p Manna–Whitneya i KS; klasa."""
    from scipy import stats as _st
    rows = []
    for k, f in enumerate(features):
        a, b = X_h[:, k], X_d[:, k]
        u = _st.mannwhitneyu(a, b, alternative="two-sided")
        delta = 2 * u.statistic / (len(a) * len(b)) - 1
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        rows.append(dict(zmienna=f, mediana_dysk=np.median(b), mediana_halo=np.median(a),
                         delta_Cliffa=delta, abs_delta=abs(delta), d_Cohena=(a.mean() - b.mean()) / sp if sp > 0 else np.nan,
                         p_MW=u.pvalue, p_KS=_st.ks_2samp(a, b).pvalue,
                         klasa="różna" if abs(delta) >= DIFF_DELTA else "podobna"))
    return pd.DataFrame(rows)


def pair_table(pairs, angles_deg, comp: pd.DataFrame, features, angles_std=None) -> pd.DataFrame:
    d = comp.set_index("zmienna")["abs_delta"]; c = comp.set_index("zmienna")["klasa"]
    rows = []
    for k, (i, j) in enumerate(pairs):
        fi, fj = features[i], features[j]
        n_r = int(c[fi] == "różna") + int(c[fj] == "różna")
        rows.append(dict(para=f"{fi}–{fj}", typ={0: "podobna–podobna", 1: "różna–podobna", 2: "różna–różna"}[n_r],
                         max_abs_delta=max(d[fi], d[fj]), suma_abs_delta=d[fi] + d[fj], kat_deg=angles_deg[k],
                         **({"kat_std_deg": angles_std[k]} if angles_std is not None else {})))
    return pd.DataFrame(rows)


def theory_check(T: pd.DataFrame) -> list[str]:
    """Teoria: pary zmiennych podobnych w obu populacjach → duże sprzężenie; pary z różną zmienną → małe."""
    from scipy import stats as _st
    lines = []
    for typ, g in T.groupby("typ"):
        lines.append(f"  {typ:18s}: n={len(g):2d}  średni kąt {g.kat_deg.mean():5.1f}°  mediana {g.kat_deg.median():5.1f}°")
    if len(T) >= 3 and T.max_abs_delta.nunique() > 1:
        r, p = _st.spearmanr(T.max_abs_delta, T.kat_deg)
        lines.append(f"  Spearman(kąt, max|δ| pary) = {r:+.2f} (p = {p:.3g}); teoria przewiduje wartość ujemną")
    else:
        r = np.nan
    pp = T[T.typ == "podobna–podobna"].kat_deg; rr = T[T.typ != "podobna–podobna"].kat_deg
    if len(pp) >= 2 and len(rr) >= 2:
        u = _st.mannwhitneyu(pp, rr, alternative="greater")
        lines.append(f"  Mann–Whitney: kąty par podobna–podobna > pozostałych: p = {u.pvalue:.3g}")
        ok = (pp.mean() > rr.mean()) and (not np.isfinite(r) or r < 0)
        lines.append("  WERDYKT: " + ("topologia ZGODNA z teorią (podobne silnie, różne słabo sprzężone)" if ok
                                     else "topologia NIEZGODNA z teorią"))
    elif len(T) == 1:
        lines.append(f"  tylko jedna para ({T.para.iloc[0]}, {T.typ.iloc[0]}): teoria przewiduje "
                     + ("duży kąt" if T.typ.iloc[0] == "podobna–podobna" else "mały kąt")
                     + f"; wyuczono {T.kat_deg.iloc[0]:.1f}°")
    else:
        lines.append("  za mało par jednego z typów, by porównać grupy")
    return lines


def plot_theory(T: pd.DataFrame, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cols = {"podobna–podobna": "#1baf7a", "różna–podobna": "#eda100", "różna–różna": "#e34948"}
    for typ, g in T.groupby("typ"):
        yerr = g["kat_std_deg"] if "kat_std_deg" in g else None
        ax.errorbar(g.max_abs_delta, g.kat_deg, yerr=yerr, fmt="o", ms=7, c=cols.get(typ, "gray"), label=typ, capsize=2)
        for _, r in g.iterrows():
            ax.annotate(r.para, (r.max_abs_delta, r.kat_deg), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axvline(DIFF_DELTA, c="gray", ls=":", lw=1)
    ax.set_xlabel("max |δ Cliffa| zmiennych pary (różnica halo – dysk)"); ax.set_ylabel("kąt sprzężenia RZZ [°]")
    ax.set_ylim(-3, 93); ax.grid(alpha=.25); ax.legend(fontsize=8); ax.set_title(title)
    save_fig(fig, path)

# ============================================================================
# ======================= 3. DANE, UCZENIE, RAPORT ============================
# ============================================================================

def add_kinematic_classes(df: pd.DataFrame) -> pd.DataFrame:
    v = df["v_tot_kms"]; good = df["dobra_astrometria"]
    df["klasa_kin"] = np.where(good & (v < DISK_VTOT_MAX), "dysk", np.where(good & (v > HALO_VTOT_MIN), "halo", "-"))
    print(f"  klasy kinematyczne: dysk={int((df.klasa_kin == 'dysk').sum())}, halo={int((df.klasa_kin == 'halo').sum())}, "
          f"pozostałe={int((df.klasa_kin == '-').sum())}")
    return df


def exact_auc(neg, pos) -> float:
    from scipy import stats as _st
    r = _st.rankdata(np.r_[neg, pos]); n1 = len(pos)
    return float((r[len(neg):].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))


def a0_scores(S, pairs, th, idx_d):
    P = encode_states(S, pairs, th); v = ground_state_vector(P[idx_d])
    return 1.0 - np.abs(P @ v.conj()) ** 2


def run_pipeline():
    ensure_dir(HILBERT_DIR)
    df = load_reference_data()
    df, S, fmin, fmax = scale_features(df)
    df = add_kinematic_classes(df)
    idx_d = np.where(df.klasa_kin == "dysk")[0]; idx_h = np.where(df.klasa_kin == "halo")[0]
    pairs = make_pairs(len(FEATURES))

    print("=" * 80); print("KROK 3.2: PORÓWNANIE ZMIENNYCH W DYSKU I HALO"); print("=" * 80)
    X = df[FEATURES].to_numpy(float)
    comp = compare_populations(X[idx_d], X[idx_h], FEATURES)
    print(comp[["zmienna", "delta_Cliffa", "d_Cohena", "klasa"]].round(3).to_string(index=False))

    print("=" * 80); print("KROK 3.3: UCZENIE TOPOLOGII – KRYTERIUM KONTRASTOWE (dysk = stan podstawowy)"); print("=" * 80)
    rng = np.random.default_rng(TOPOLOGY_SEED)
    starts = [np.zeros(len(pairs)), np.full(len(pairs), np.deg2rad(INITIAL_ANGLE_DEG))] + \
             [rng.uniform(*np.deg2rad(TOPOLOGY_BOUNDS_DEG), len(pairs)) for _ in range(N_RANDOM_STARTS)]
    names = ["θ0 = 0°", f"θ0 = {INITIAL_ANGLE_DEG:.0f}°"] + [f"losowy start {k + 1}" for k in range(N_RANDOM_STARTS)]
    t0 = time.time()
    x, cost, allres = learn_contrast_topology(S, pairs, idx_d, idx_h, starts, STEP_SCHEDULE_DEG, TRIALS_PER_STAGE,
                                              BATCH_PER_TRIAL, PATIENCE, TOPOLOGY_SEED, verbose=True, names=names)
    print(f"  ✓ najlepszy start: {allres[0][2]} | {time.time() - t0:.0f} s")

    # ----- jakość rozdzielenia i przerwa masowa dysku -----
    zero = np.zeros(len(pairs)); res_lines = []
    for nm, th in [("θ = 0", zero), ("kontrastowa (wyuczona)", x)]:
        a0 = a0_scores(S, pairs, th, idx_d)
        sp = spectral(density_matrix(encode_states(S[idx_d], pairs, th)))
        res_lines.append(f"  {nm:24s}: AUC(a0: halo vs dysk) = {exact_auc(a0[idx_d], a0[idx_h]):.4f}   "
                         f"gładkie AUC = {contrast_objective(S, pairs, th, idx_d, idx_h):.4f}   "
                         f"przerwa masowa dysku = {sp['mass_gap']:.3f}   λ0 = {sp['eigvals'][0]:.3f}")

    # ----- teoria: kąty vs różnice zmiennych -----
    T = pair_table(pairs, np.rad2deg(x), comp, FEATURES)
    th_lines = theory_check(T)
    # ablacja: pojedyncza para przy wyuczonym kącie (reszta 0) oraz wyuczona topologia bez tej pary
    base0 = contrast_objective(S, pairs, zero, idx_d, idx_h); full = contrast_objective(S, pairs, x, idx_d, idx_h)
    abl = []
    for k in range(len(pairs)):
        only = zero.copy(); only[k] = x[k]; wo = x.copy(); wo[k] = 0.0
        abl.append(dict(para=T.para[k], typ=T.typ[k], kat_deg=np.rad2deg(x[k]),
                        zysk_sama_para=contrast_objective(S, pairs, only, idx_d, idx_h) - base0,
                        strata_bez_pary=full - contrast_objective(S, pairs, wo, idx_d, idx_h)))
    A = pd.DataFrame(abl)
    T = T.merge(A[["para", "zysk_sama_para", "strata_bez_pary"]], on="para")
    comp.to_csv(HILBERT_DIR / "porownanie_zmiennych_dysk_halo.csv", index=False)
    T.to_csv(HILBERT_DIR / "topologia_vs_roznice.csv", index=False)

    write_report(HILBERT_DIR / "topology_report.txt", f"TOPOLOGIA KONTRASTOWA – {VARIANT}", [
        ("Dane", [f"  gwiazd: {len(S)}; dysk (v_tot < {DISK_VTOT_MAX:.0f}) = {len(idx_d)}, halo (v_tot > {HALO_VTOT_MIN:.0f}) = {len(idx_h)}",
                  "  kubity: " + ", ".join(f"q{i}={f}" for i, f in enumerate(FEATURES)),
                  f"  wymiar przestrzeni Hilberta: {2 ** len(FEATURES)}; obwód Ry(x·π/2) → Π RZZ(θ_ij) → Ry(x·π/2)"]),
        ("Zmienne w dysku i halo (δ Cliffa = 2·AUC − 1; „różna”, gdy |δ| ≥ %.2f)" % DIFF_DELTA,
         ["  " + l for l in comp.round(4).to_string(index=False).splitlines()]),
        ("Uczenie topologii (kryterium kontrastowe)",
         [f"  funkcja celu: −gładkie AUC(a0 halo > a0 dysk) + {KONTRAST_REG}·Σθ²; a0 względem stanu podstawowego ρ_dysk"]
         + [f"    {nm:22s} koszt {c:.4f}  kąty {np.round(np.rad2deg(th), 1).tolist()}" for c, th, nm in allres]),
        ("Rozdzielenie halo/dysk", res_lines),
        ("Kąty RZZ a różnice zmiennych (teoria)", ["  " + l for l in T.round(4).to_string(index=False).splitlines()] + [""] + th_lines),
        ("Interpretacja ablacji", ["  zysk_sama_para: zmiana gładkiego AUC, gdy włączona jest tylko ta para (kąt wyuczony),",
                                   "  strata_bez_pary: spadek gładkiego AUC po wyłączeniu tej pary w wyuczonej topologii."]),
        ("Weryfikacja kodowania Qiskitem", verify_with_qiskit(S, pairs, x) if VERIFY_WITH_QISKIT else ["  pominięto"]),
    ])
    plot_theory(T, HILBERT_DIR / "topologia_vs_roznice.png", f"Kąty RZZ a różnice halo–dysk – {VARIANT}")
    plot_topology_graph(pairs, x, FEATURES, HILBERT_DIR / "topology_graph.png", f"Topologia kontrastowa – {VARIANT}")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for nm, th, c in [("θ = 0", zero, "gray"), ("kontrastowa", x, "crimson")]:
        a0 = a0_scores(S, pairs, th, idx_d)
        bins = np.linspace(0, max(a0[np.r_[idx_d, idx_h]].max(), 1e-3), 40)
        ax.hist(a0[idx_d], bins=bins, histtype="step", density=True, color=c, ls="-", label=f"dysk ({nm})")
        ax.hist(a0[idx_h], bins=bins, histtype="step", density=True, color=c, ls="--", lw=2, label=f"halo ({nm})")
    ax.set_xlabel("a0 = 1 − F0 (względem stanu podstawowego dysku)"); ax.set_ylabel("gęstość"); ax.legend(fontsize=8)
    ax.grid(alpha=.25); save_fig(fig, HILBERT_DIR / "a0_dysk_halo.png")

    np.savez_compressed(STATES_NPZ, Psi=encode_states(S, pairs, x), S=S, features=np.array(FEATURES), feat_min=fmin,
                        feat_max=fmax, pairs=np.array(pairs), angles=x, angles_original=np.array([]),
                        star_id=df["star_id"].to_numpy(), mode=np.array(MODE),
                        disk_vtot_max=DISK_VTOT_MAX, halo_vtot_min=HALO_VTOT_MIN)
    pd.DataFrame({"pair_i": [i for i, _ in pairs], "pair_j": [j for _, j in pairs], "angle_rad": x,
                  "angle_deg": np.rad2deg(x)}).to_csv(TOPOLOGY_FILE, index=False)
    df.to_csv(HILBERT_DIR / "qiftdata_scaled.csv", index=False)
    print("\n".join(th_lines))
    print(f"\n✓ KONIEC KROKU 3 ({VARIANT}). Wyniki w:", HILBERT_DIR.resolve())
    return dict(data=df, pairs=pairs, angles=x, porownanie=comp, teoria=T)


if __name__ == "__main__":
    results_03 = run_pipeline()
