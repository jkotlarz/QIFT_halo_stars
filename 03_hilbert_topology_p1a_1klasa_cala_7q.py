# -*- coding: utf-8 -*-
"""
03_hilbert_topology_p1a_1klasa_cala_7q.py
======================================================================
KROK 3 potoku QIFT-spectra – PODEJŚCIE 1a: 7 latentów (atmosfera + 5 chemicznych), jedna klasa, referencja: cała populacja
  kubity: w_atm_1, w_atm_2, z_Fe, z_alpha, z_C, z_N, z_act
  referencja: jedna klasa – cała populacja (bez etykiet)

Co robi ten plik
----------------
1. Wczytuje dane referencyjne gwiazd z REF_CSV (gwiazdy_qift_kinematyka.csv:
   kinematyka Gaia DR3) i łączy je po obsid z wynikami kroku 02
   (astro_reports/qiftdata.csv: latenty VAE, sygnatury sig_*, MSE).
2. Wyznacza klasy kinematyczne (dobra astrometria Gaia):
     dysk: v_tot < DISK_VTOT_MAX,  halo: v_tot > HALO_VTOT_MIN,  pozostałe: „-”.
3. Skaluje cechy min–max do [0, 1] (na całej próbie – bez etykiet) i koduje
   gwiazdy obwodem Ry(x·π/2) → Π RZZ(θ_ij) → Ry(x·π/2) (Qiskit little-endian).
4. Uczy topologię RZZ WYŁĄCZNIE przez maksymalizację przerwy masowej E1 − E0
   operatora gęstości referencji (jak w d-QSA):
     MODE = "jedna_klasa_cala"  -> ρ całej populacji,
     MODE = "jedna_klasa_dysk"  -> ρ gwiazd dysku,
     MODE = "dwie_klasy"        -> średnia przerwa ρ_dysk i ρ_halo.
   Kilka startów (θ = INITIAL_ANGLE_DEG, topologia z oryginalnego potoku,
   starty losowe), hill-climbing, wybór najlepszego.
5. Zapisuje do HILBERT_DIR: topologię, raport, wykresy i qift_states.npz
   (stany przy kątach wyuczonych, kąty θ = 0 i oryginalne do porównań w kroku 04).
   W kroku 04 topologia jest dla uczciwości uczona ponownie w każdym foldzie CV
   (dla podejść z referencją dyskową lub dwiema klasami).

Wersja z nowymi latentami chemicznymi: z_C (węgiel), z_N (azot), z_act (aktywność
chromosferyczna) z kroku 02 w wersji 2. Kubity atmosfery (jeśli są) pozostają q0, q1.

Uruchomienie: python 03_hilbert_topology_p1a_1klasa_cala_7q.py
"""

from __future__ import annotations

import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

# ============================================================================
# ======================= PARAMETRY URUCHOMIENIOWE ===========================
# ============================================================================

VARIANT = "p1a_1klasa_cala_7q"
MODE = "jedna_klasa_cala"                 # "jedna_klasa_cala" | "jedna_klasa_dysk" | "dwie_klasy"

# --- Wejście ---
REF_CSV = Path("gwiazdy_qift_kinematyka.csv")
QIFTDATA_PATH = Path("astro_reports") / "qiftdata.csv"
LATENT_SOURCE = "qiftdata"        # "qiftdata" (latenty z kroku 02) | "ref_csv"
ORIGINAL_TOPOLOGY_FILE = Path("hilbertspace_topology") / "best_topology.csv"   # z oryginalnego potoku (4 kubity)

# --- Wyjście ---
HILBERT_DIR = Path("hilbertspace_topology") / VARIANT
TOPOLOGY_FILE = HILBERT_DIR / "best_topology.csv"
STATES_NPZ = HILBERT_DIR / "qift_states.npz"

# --- Cechy kodowane na kubity ---
FEATURES = ["w_atm_1", "w_atm_2", "z_Fe", "z_alpha", "z_C", "z_N", "z_act"]

# --- Klasy kinematyczne ---
DISK_VTOT_MAX = 70.0              # [km/s] dysk (referencja)
HALO_VTOT_MIN = 220.0             # [km/s] halo (etykieta ewaluacji; referencja w MODE="dwie_klasy")

# --- Uczenie topologii: maksymalizacja przerwy masowej ---
TOPOLOGY_BOUNDS_DEG = (0.0, 90.0)
INITIAL_ANGLE_DEG = 20.0
N_RANDOM_STARTS = 2               # dodatkowe losowe starty hill-climbingu
TOPOLOGY_REG_LAMBDA = 1e-4        # L2: λ·Σθ²
ANGLE_PENALTY_THRESHOLD_DEG = 80.0
ANGLE_PENALTY_LAMBDA = 1.5
ANGLE_PENALTY_POWER = 2.0
STEP_SCHEDULE_DEG = [12.0, 8.0, 3.0, 1.0, 0.3]
TRIALS_PER_STAGE = 100
BATCH_PER_TRIAL = 6
PATIENCE = 10
TOPOLOGY_SEED = 48

# --- Regularyzacja ρ i energie ---
RHO_EPS = 1e-6
ENERGY_EPS = 1e-12

# --- Diagnostyka ---
VERIFY_WITH_QISKIT = True
N_QISKIT_CHECK = 5
SENSITIVITY_GRID_DEG = np.linspace(0, 90, 31)
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



# ============================================================================
# ================= 3. ρ REFERENCJI, PRZERWA MASOWA, UCZENIE =================
# ============================================================================

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



def add_kinematic_classes(df: pd.DataFrame) -> pd.DataFrame:
    v = df["v_tot_kms"]; good = df["dobra_astrometria"]
    df["klasa_kin"] = np.where(good & (v < DISK_VTOT_MAX), "dysk", np.where(good & (v > HALO_VTOT_MIN), "halo", "-"))
    print(f"  klasy kinematyczne: dysk={int((df.klasa_kin == 'dysk').sum())}, halo={int((df.klasa_kin == 'halo').sum())}, "
          f"pozostałe={int((df.klasa_kin == '-').sum())}")
    return df


def reference_sets(df: pd.DataFrame) -> list[np.ndarray]:
    """Lista zbiorów indeksów, których ρ wchodzą do kryterium przerwy masowej."""
    if MODE == "jedna_klasa_cala":
        return [np.arange(len(df))]
    if MODE == "jedna_klasa_dysk":
        return [np.where(df.klasa_kin == "dysk")[0]]
    if MODE == "dwie_klasy":
        return [np.where(df.klasa_kin == "dysk")[0], np.where(df.klasa_kin == "halo")[0]]
    raise ValueError(MODE)


def mean_gap(S, pairs, th, ref_sets) -> float:
    P = encode_states(S, pairs, th)
    return float(np.mean([spectral(density_matrix(P[idx]))["mass_gap"] for idx in ref_sets]))


def angle_penalty(th) -> float:
    ex = np.maximum(np.abs(th) - np.deg2rad(ANGLE_PENALTY_THRESHOLD_DEG), 0.0)
    return float(ANGLE_PENALTY_LAMBDA * np.sum(ex ** ANGLE_PENALTY_POWER))


def clip_angles(th) -> np.ndarray:
    lo, hi = np.deg2rad(TOPOLOGY_BOUNDS_DEG)
    return np.clip(np.asarray(th, float), lo, hi)


def gap_cost(th, S, pairs, ref_sets) -> dict:
    th = clip_angles(th)
    g = mean_gap(S, pairs, th, ref_sets)
    return dict(cost=-g + TOPOLOGY_REG_LAMBDA * float(np.sum(th ** 2)) + angle_penalty(th), mass_gap=g)


def hill_climb(x0, S, pairs, ref_sets, seed):
    rng = np.random.default_rng(seed)
    x = clip_angles(x0); best = gap_cost(x, S, pairs, ref_sets); hist = [dict(eval=1, **best)]
    for step in STEP_SCHEDULE_DEG:
        no_imp = 0
        for _ in range(TRIALS_PER_STAGE):
            improved = False
            for _ in range(BATCH_PER_TRIAL):
                d = rng.normal(size=len(x)); d /= max(np.linalg.norm(d), 1e-12)
                xt = clip_angles(x + rng.uniform(0.3, 1.0) * np.deg2rad(step) * d)
                ft = gap_cost(xt, S, pairs, ref_sets)
                if ft["cost"] < best["cost"] - 1e-12:
                    x, best, improved = xt, ft, True
                    hist.append(dict(eval=len(hist) + 1, **best, **{f"theta_{i}_{j}_deg": np.rad2deg(a) for (i, j), a in zip(pairs, x)}))
            no_imp = 0 if improved else no_imp + 1
            if no_imp >= PATIENCE:
                break
    return x, best, pd.DataFrame(hist)


def load_original_topology(pairs):
    if not ORIGINAL_TOPOLOGY_FILE.exists():
        return None
    t = pd.read_csv(ORIGINAL_TOPOLOGY_FILE)
    p_f = list(zip(t.pair_i.astype(int), t.pair_j.astype(int)))
    if sorted(p_f) != sorted(pairs):
        return None
    amap = dict(zip(p_f, t.angle_rad.to_numpy(float)))
    return np.array([amap[p] for p in pairs])


def learn_topology(S, pairs, ref_sets):
    """Wielostartowa maksymalizacja przerwy masowej. Zwraca kąty, info i historię najlepszego startu."""
    rng = np.random.default_rng(TOPOLOGY_SEED)
    starts = [("θ0 = %.0f°" % INITIAL_ANGLE_DEG, np.full(len(pairs), np.deg2rad(INITIAL_ANGLE_DEG)))]
    orig = load_original_topology(pairs)
    if orig is not None:
        starts.append(("topologia oryginalna", orig))
    starts += [(f"losowy start {k + 1}", rng.uniform(*np.deg2rad(TOPOLOGY_BOUNDS_DEG), len(pairs))) for k in range(N_RANDOM_STARTS)]
    res, t0 = [], time.time()
    for k, (name, x0) in enumerate(starts):
        x, best, hist = hill_climb(x0, S, pairs, ref_sets, TOPOLOGY_SEED + k)
        res.append((best["cost"], name, x, best, hist))
        print(f"  start: {name:22s} -> przerwa {best['mass_gap']:.4f}  kąty [°] {np.round(np.rad2deg(x), 1).tolist()}")
    res.sort(key=lambda r: r[0])
    _, name, x, best, hist = res[0]
    print(f"  ✓ najlepszy start: {name} | przerwa {best['mass_gap']:.4f} | {time.time() - t0:.1f} s")
    return x, dict(best_start=name, mass_gap=best["mass_gap"],
                   all_starts=[(r[1], r[3]["mass_gap"], np.rad2deg(r[2]).round(1).tolist()) for r in res]), hist, orig


# ============================================================================
# ========================= 4. RAPORTY I WYKRESY ==============================
# ============================================================================

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



def report(S, df, pairs, x, info, hist, orig, ref_sets, out_dir: Path = HILBERT_DIR):
    print("=" * 80)
    print("KROK 3.3: RAPORTY I WYKRESY")
    print("=" * 80)
    n = len(FEATURES); zero = np.zeros(len(pairs))
    settings = [("θ = 0", zero), ("wyuczona (przerwa masowa)", x)] + ([("oryginalna", orig)] if orig is not None else [])
    lines = []
    for name, th in settings:
        P = encode_states(S, pairs, th)
        sp = [spectral(density_matrix(P[idx])) for idx in ref_sets]
        lines.append(f"  {name:28s}: przerwa = " + " | ".join(f"{s['mass_gap']:.4f}" for s in sp)
                     + "   λ0 = " + " | ".join(f"{s['eigvals'][0]:.4f}" for s in sp))
    sens = []
    for k, (i, j) in enumerate(pairs):
        for a in SENSITIVITY_GRID_DEG:
            th = x.copy(); th[k] = np.deg2rad(a)
            sens.append(dict(pair=f"{i}-{j}", angle_deg=a, mass_gap=mean_gap(S, pairs, th, ref_sets)))
    sens = pd.DataFrame(sens); sens.to_csv(out_dir / "topology_sensitivity_scan.csv", index=False)
    tab = pd.DataFrame({"para": [f"{i}-{j}" for i, j in pairs], "cechy": [f"{FEATURES[i]} ⊗ {FEATURES[j]}" for i, j in pairs],
                        "wyuczony_deg": np.rad2deg(x)})
    if orig is not None:
        tab["oryginalny_deg"] = np.rad2deg(orig)
    ref_names = {"jedna_klasa_cala": ["cała populacja"], "jedna_klasa_dysk": ["dysk"], "dwie_klasy": ["dysk", "halo"]}[MODE]
    write_report(out_dir / "topology_report.txt", f"RAPORT TOPOLOGII – {VARIANT}", [
        ("Dane", [f"  gwiazd: {len(S)} | REF_CSV: {REF_CSV} | latenty: {LATENT_SOURCE}",
                  f"  klasy kinematyczne: dysk (v_tot<{DISK_VTOT_MAX:.0f}) = {int((df.klasa_kin == 'dysk').sum())}, "
                  f"halo (v_tot>{HALO_VTOT_MIN:.0f}) = {int((df.klasa_kin == 'halo').sum())}",
                  f"  referencja: {MODE} -> " + ", ".join(f"{nm} ({len(idx)})" for nm, idx in zip(ref_names, ref_sets))]),
        ("Kodowanie", ["  kubity: " + ", ".join(f"q{i}={f}" for i, f in enumerate(FEATURES)),
                       f"  wymiar przestrzeni Hilberta: {2 ** n}; obwód Ry(x·π/2) → Π RZZ(θ_ij) → Ry(x·π/2)"]),
        ("Uczenie topologii (wyłącznie maksymalizacja przerwy masowej)",
         [f"  funkcja celu: −śr. przerwa(" + "/".join(ref_names) + f") + {TOPOLOGY_REG_LAMBDA}·Σθ² + kara za θ > {ANGLE_PENALTY_THRESHOLD_DEG:.0f}°",
          f"  najlepszy start: {info['best_start']}"] + [f"    {nm:22s} przerwa {g:.4f} kąty {a}" for nm, g, a in info["all_starts"]]),
        ("Kąty RZZ", ["  " + l for l in tab.round(2).to_string(index=False).splitlines()]),
        ("Przerwa masowa i λ0 referencji (" + " | ".join(ref_names) + ")", lines),
        ("Wrażliwość przerwy na kąt pojedynczej pary",
         [f"  para {p}: {g.mass_gap.min():.4f}–{g.mass_gap.max():.4f}, maks. przy {g.loc[g.mass_gap.idxmax(), 'angle_deg']:.0f}°"
          for p, g in sens.groupby("pair", sort=False)]),
        ("Weryfikacja kodowania Qiskitem", verify_with_qiskit(S, pairs, x) if VERIFY_WITH_QISKIT else ["  pominięto"]),
    ])
    plot_topology_graph(pairs, x, FEATURES, out_dir / "topology_graph.png",
                        f"Topologia RZZ – {VARIANT} (przerwa = {info['mass_gap']:.3f})")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for p, g in sens.groupby("pair", sort=False):
        axes[0].plot(g.angle_deg, g.mass_gap, label=p)
    for a in np.rad2deg(x):
        axes[0].axvline(a, color="gray", ls=":", lw=.6)
    axes[0].set_xlabel("kąt pary [°] (pozostałe ustalone)"); axes[0].set_ylabel("śr. przerwa masowa")
    axes[0].set_title("Wrażliwość przerwy masowej"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.25)
    for name, th in settings:
        P = encode_states(S, pairs, th); l = spectral(density_matrix(P[ref_sets[0]]))["eigvals"]
        axes[1].semilogy(np.arange(len(l)), l, "o-", ms=4, label=name)
    axes[1].set_xlabel("k"); axes[1].set_ylabel("λ_k"); axes[1].set_title(f"Widmo ρ referencji ({ref_names[0]})")
    axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); save_fig(fig, out_dir / "topology_spectrum_sensitivity.png")
    if hist is not None and len(hist) > 1:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(hist["eval"], hist["mass_gap"], "o-", ms=3); ax.set_xlabel("zaakceptowany krok"); ax.set_ylabel("przerwa masowa")
        ax.set_title("Historia uczenia topologii (najlepszy start)"); ax.grid(alpha=.25)
        save_fig(fig, out_dir / "topology_optimization_history.png")


def save_states(df, S, fmin, fmax, pairs, x, orig, path: Path = STATES_NPZ):
    np.savez_compressed(path, Psi=encode_states(S, pairs, x), S=S, features=np.array(FEATURES), feat_min=fmin, feat_max=fmax,
                        pairs=np.array(pairs), angles=x, angles_original=(orig if orig is not None else np.array([])),
                        star_id=df["star_id"].to_numpy(), mode=np.array(MODE),
                        disk_vtot_max=DISK_VTOT_MAX, halo_vtot_min=HALO_VTOT_MIN)
    pd.DataFrame({"pair_i": [i for i, _ in pairs], "pair_j": [j for _, j in pairs], "angle_rad": x,
                  "angle_deg": np.rad2deg(x)}).to_csv(TOPOLOGY_FILE, index=False)
    df.to_csv(HILBERT_DIR / "qiftdata_scaled.csv", index=False)
    print(f"  ✓ {path}\n  ✓ {TOPOLOGY_FILE}\n  ✓ {HILBERT_DIR / 'qiftdata_scaled.csv'}")


# ============================================================================
# ======================= URUCHOMIENIE PIPELINE'U =============================
# ============================================================================

def run_pipeline():
    ensure_dir(HILBERT_DIR)
    df = load_reference_data()
    df, S, fmin, fmax = scale_features(df)
    df = add_kinematic_classes(df)
    print("=" * 80)
    print(f"KROK 3.2: UCZENIE TOPOLOGII – MAKSYMALIZACJA PRZERWY MASOWEJ ({MODE})")
    print("=" * 80)
    pairs = make_pairs(len(FEATURES)); ref_sets = reference_sets(df)
    x, info, hist, orig = learn_topology(S, pairs, ref_sets)
    report(S, df, pairs, x, info, hist, orig, ref_sets)
    save_states(df, S, fmin, fmax, pairs, x, orig)
    print(f"\n✓ KONIEC KROKU 3 ({VARIANT}). Wyniki w:", HILBERT_DIR.resolve())
    return dict(data=df, pairs=pairs, angles=x)


if __name__ == "__main__":
    results_03 = run_pipeline()
