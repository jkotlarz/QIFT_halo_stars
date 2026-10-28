"""Wspólne funkcje skryptów 05*: kodowanie QIFT, entropia splątania, odczyt kątów z raportów kroku 04.

Kodowanie identyczne z krokami 03/04: |ψ⟩ = U_enc(s) U_ZZ(θ) U_enc(s) |0…0⟩, U_enc = ⊗ R_y(π s_j / 2),
U_ZZ = Π exp(−i θ_jk Z_j Z_k / 2); kubit j = bit j indeksu stanu (kubit 0 = najmłodszy bit).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

REPORTS = Path("qift_reports")
L7 = ["w_atm_1", "w_atm_2", "z_Fe", "z_alpha", "z_C", "z_N", "z_act"]
L5 = ["z_Fe", "z_alpha", "z_C", "z_N", "z_act"]


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def _apply(P, U, q, n):
    N = P.shape[0]; T = P.reshape(N, 2 ** (n - 1 - q), 2, 2 ** q)
    return np.einsum("nab,nxby->nxay", U, T).reshape(N, 2 ** n)


def encode(S: np.ndarray, angles_deg) -> np.ndarray:
    """S: (N, n) wartości przeskalowane do [0, 1]; angles_deg: kąty par (i<j) w kolejności leksykograficznej."""
    N, n = S.shape
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    a = S * np.pi / 2
    P = np.zeros((N, 2 ** n), complex); P[:, 0] = 1
    for q in range(n):
        P = _apply(P, _ry(a[:, q]).astype(complex), q, n)
    idx = np.arange(2 ** n); z = 1 - 2 * ((idx[:, None] >> np.arange(n)[None, :]) & 1)
    ph = np.zeros(2 ** n)
    for (i, j), th in zip(pairs, np.deg2rad(np.asarray(angles_deg, float))):
        ph += -0.5 * th * z[:, i] * z[:, j]
    P = P * np.exp(1j * ph)[None]
    for q in range(n):
        P = _apply(P, _ry(a[:, q]).astype(complex), q, n)
    return P


def entanglement_entropy(P: np.ndarray, n_a: int = 2) -> np.ndarray:
    """S(A|B) [bit] dla A = n_a najmłodszych kubitów (atmosfera) i B = reszta (chemia)."""
    n = int(np.log2(P.shape[1]))
    M = P.reshape(-1, 2 ** (n - n_a), 2 ** n_a)
    rA = np.einsum("nba,nbc->nac", M, M.conj())
    w = np.linalg.eigvalsh(rA).clip(1e-15)
    return -(w * np.log2(w)).sum(1)


def fold_mean_angles(run: str) -> list[float]:
    """Średnie kąty z foldów CV z raport_glowny.txt kroku 04 (linia 'kąty uczone w foldach [°]: średnia [...]');
    gdy jej brak (podejścia bez CV) – kąty z kroku 03 ('kąty z kroku 03 ...')."""
    txt = (REPORTS / run / "raport_glowny.txt").read_text(encoding="utf-8")
    for pat in [r"kąty uczone w foldach \[°\]: średnia \[([^\]]*)\]", r"kąty z kroku 03[^\[]*\[°\]: \[([^\]]*)\]"]:
        m = re.search(pat, txt)
        if m:
            return [float(x) for x in m.group(1).split(",")]
    raise ValueError(f"nie znaleziono kątów w {run}/raport_glowny.txt")
