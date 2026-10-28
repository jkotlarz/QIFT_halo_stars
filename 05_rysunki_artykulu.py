"""KROK 05 (artykuł): rysunki 1–9 i liczby do wersji 3 artykułu (7 latentów, topologia kontrastowa).

Użycie: python 05_rysunki_artykulu.py   (z katalogu głównego repozytorium, po krokach 03/04 wszystkich podejść)
Wejście: qift_reports/<podejście>/ (słownik RUN; nazwy U/D/B + liczba kubitów jak w artykule).
Wyjście: artykul/figures/*.pdf, *.png oraz artykul/numbers_v3.json
"""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_curve, matthews_corrcoef

R = "qift_reports/"
RUN = {  # komórka z artykułu -> katalog wyników kroku 04
    "U7": R + "p1a_1klasa_cala_7q", "U5": R + "p3a_1klasa_cala_5q", "U2": R + "p3a_1klasa_cala_2q",
    "D7": R + "k1_kontrast_wszystkie_7q", "D5": R + "k3_kontrast_chemia_5q", "D2": R + "k2_kontrast_Fe_alpha_2q",
    "B7": R + "p2_2klasy_7q", "B5": R + "p4_2klasy_5q", "B2": R + "p4_2klasy_2q",
    "G7": R + "p1b_1klasa_dysk_7q",  # dysk + przerwa masowa (decyzja i splątanie)
}
OUT = "artykul/figures/"; os.makedirs(OUT, exist_ok=True)
ENC = {"7": "#2a78d6", "5": "#eb6834", "2": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .5, "legend.frameon": False,
                     "font.family": "serif", "savefig.bbox": "tight", "mathtext.fontset": "dejavuserif"})
W1 = 5.9
NUM = {}


def save(fig, name):
    fig.savefig(OUT + name + ".pdf"); fig.savefig(OUT + name + ".png", dpi=200); plt.close(fig)


rank = {k: pd.read_csv(v + "/metryki_ranking.csv") for k, v in RUN.items()}
deci = {k: pd.read_csv(v + "/metryki_decyzje.csv") for k, v in RUN.items()}
pair = {k: pd.read_csv(v + "/porownania_parowane.csv") for k, v in RUN.items()}
res = {k: pd.read_csv(v + "/wyniki_gwiazd.csv") for k, v in RUN.items()}
base = res["G7"]
ev = base.eval_mask.astype(bool).to_numpy(); y = base.halo_kin.astype(int).to_numpy()[ev]
TAG = {"gap": {"U": "wyuczona (przerwa masowa)", "B": "wyuczona (przerwa masowa)", "D": "przerwa masowa dysku"},
       "th0": "θ = 0", "kontr": "kontrastowa (wyuczona)"}


def rget(cell, measure, setting):
    c = cell[0]
    tag = TAG["gap"][c] if setting == "gap" else TAG[setting]
    name = (f"QIFT s = F_halo − F_dysk [{tag}]" if measure == "s" else f"QIFT {measure} [{tag}]")
    r = rank[cell]; m = r[r.metoda == name]
    return m.iloc[0] if len(m) else None


def cget(cell, name):
    r = rank[cell]; return r[r.metoda == name].iloc[0]


# ------------------------------------------------------------------ Rys. 1: próba
d = base
fig, ax = plt.subplots(1, 2, figsize=(W1, 2.6), gridspec_kw=dict(wspace=.3))
UW = np.hypot(d.U_lsr_kms, d.W_lsr_kms)
sc = ax[0].scatter(d.V_lsr_kms, UW, c=d.feh, s=4, cmap="viridis_r", vmin=-2.3, vmax=0.5, lw=0)
t = np.linspace(0, np.pi, 200)
for r_, ls in [(70, ":"), (220, "--")]:
    ax[0].plot(r_ * np.cos(t), r_ * np.sin(t), color=INK, lw=.8, ls=ls)
ax[0].set(xlim=(-450, 150), ylim=(0, 450), xlabel=r"$V_{\rm LSR}$ [km s$^{-1}$]",
          ylabel=r"$(U_{\rm LSR}^2+W_{\rm LSR}^2)^{1/2}$ [km s$^{-1}$]")
ax[0].text(-440, 228, r"$v_{\rm tot}=220$", fontsize=7, color=INK2)
ax[1].scatter(d.teff, d.logg, c=d.feh, s=4, cmap="viridis_r", vmin=-2.3, vmax=0.5, lw=0)
hk = (d.klasa_kin == "halo") & d.eval_mask.astype(bool)
ax[1].scatter(d.teff[hk], d.logg[hk], s=22, facecolors="none", edgecolors=INK, lw=.6, label=f"kinematic halo ({hk.sum()})")
ax[1].set(xlim=(6700, 4000), ylim=(5.1, 0.4), xlabel=r"$T_{\rm eff}$ [K]", ylabel=r"$\log g$")
ax[1].legend(loc="upper left", fontsize=7)
cb = fig.colorbar(sc, ax=ax, shrink=.85, pad=.02); cb.set_label("[Fe/H] (LAMOST)")
save(fig, "fig_sample")

# ---------------------------------------------- Rys. 2: latenty × etykiety (r Pearsona)
L = ["w_atm_1", "w_atm_2", "z_Fe", "z_alpha", "z_C", "z_N", "z_act"]
LL = [r"$w_{\rm atm,1}$", r"$w_{\rm atm,2}$", r"$z_{\rm Fe}$", r"$z_\alpha$", r"$z_{\rm C}$", r"$z_{\rm N}$", r"$z_{\rm act}$"]
P = ["logg", "teff", "feh", "alpha_m", "kotwica_C", "kotwica_N", "kotwica_akt"]
PL = [r"$\log g$", r"$T_{\rm eff}$", "[Fe/H]", r"[$\alpha$/M]", "[C/Fe]", "[N/Fe]", r"$\Delta S$"]
Rm = np.array([[d[[a, b]].dropna().corr().iloc[0, 1] for b in P] for a in L])
Rc = np.array([[d[[a, b]].dropna().corr().iloc[0, 1] for b in P] for a in P])  # etykieta-celu × etykieta
fig, ax = plt.subplots(1, 2, figsize=(W1, 2.9), gridspec_kw=dict(wspace=.35))
for a, M, ttl, yl in [(ax[0], Rm, "latents vs. labels", LL), (ax[1], Rm - Rc, "latent $r$ minus target-label $r$", LL)]:
    im = a.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    a.set_xticks(range(7)); a.set_xticklabels(PL, rotation=45, ha="right"); a.set_yticks(range(7)); a.set_yticklabels(yl)
    a.grid(False); a.set_title(ttl, fontsize=8)
    for i in range(7):
        for j in range(7):
            a.text(j, i, f"{M[i, j]:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center", fontsize=5.5,
                   color="white" if abs(M[i, j]) > .6 else INK)
cb = fig.colorbar(im, ax=ax, shrink=.8, pad=.02); cb.set_label("Pearson $r$ / difference")
save(fig, "fig_latents")
NUM["latent_r_target"] = {l: float(Rm[i, i]) for i, l in enumerate(L)}
NUM["latent_max_excess"] = float(np.max(np.abs((Rm - Rc)[~np.eye(7, dtype=bool)])))
NUM["n_anchor"] = {p: int(d[p].notna().sum()) for p in P}

# ------------------------------------------------------------- Rys. 3: AUC/AP (forest)
rows = []
for e in ["7", "5", "2"]:
    rows += [(f"U{e}: $a_0$", e, [("th0", rget("U" + e, "a0", "th0")), ("gap", rget("U" + e, "a0", "gap"))]),
             (f"U{e}: $T^2$", e, [("th0", rget("U" + e, "T²", "th0")), ("gap", rget("U" + e, "T²", "gap"))]),
             (f"D{e}: $a_0$", e, [("th0", rget("D" + e, "a0", "th0")), ("gap", rget("D" + e, "a0", "gap")), ("kontr", rget("D" + e, "a0", "kontr"))]),
             (f"D{e}: $T^2$", e, [("th0", rget("D" + e, "T²", "th0")), ("gap", rget("D" + e, "T²", "gap")), ("kontr", rget("D" + e, "T²", "kontr"))]),
             (f"B{e}: $s$", e, [("th0", rget("B" + e, "s", "th0")), ("gap", rget("B" + e, "s", "gap"))])]
cls = [("Isolation Forest (D7)", cget("D7", "Isolation Forest")), ("MCD (U2)", cget("U2", "Mahalanobis (MCD)")),
       ("LOF (D7)", cget("D7", "LOF")), ("Log. regression (7)", cget("D7", "regresja logistyczna")),
       ("QDA (2)", cget("D2", "QDA")), (r"$-z_{\rm Fe}$ only", cget("D7", "−z_Fe"))]
MK = {"th0": dict(fmt="o", mfc="white"), "gap": dict(fmt="o"), "kontr": dict(fmt="D")}
OFF = {"th0": .22, "gap": 0, "kontr": -.22}
fig, ax = plt.subplots(1, 2, figsize=(W1, 5.6), sharey=True)
ny = len(rows) + len(cls)
for k, (met, lo, hi, xl) in enumerate([("AUC", "AUC_lo", "AUC_hi", "AUC"), ("AP", "AP_lo", "AP_hi", "average precision")]):
    a = ax[k]
    for i, (lab, e, sets) in enumerate(rows):
        yy = ny - i
        for s, r in sets:
            a.errorbar(r[met], yy + OFF[s], xerr=[[r[met] - r[lo]], [r[hi] - r[met]]], ms=3.5, color=ENC[e], lw=.7,
                       capsize=0, **MK[s])
    for j, (lab, r) in enumerate(cls):
        yy = ny - len(rows) - j
        a.errorbar(r[met], yy, xerr=[[r[met] - r[lo]], [r[hi] - r[met]]], fmt="s", ms=3.2, color=INK2, lw=.7)
    a.axvline(cls[-1][1][met], color=INK2, lw=.6, ls="--")
    for b_ in [5, 10]:
        a.axhline(ny - b_ + .5, color=GRID, lw=.6)
    a.axhline(ny - len(rows) + .5, color=INK2, lw=.5)
    a.set_xlabel(xl)
    if met == "AP":
        a.axvline(0.046, color=INK2, lw=.6, ls=":")
ax[0].set_yticks([ny - i for i in range(len(rows))] + [ny - len(rows) - j for j in range(len(cls))])
ax[0].set_yticklabels([r[0] for r in rows] + [c[0] for c in cls])
ax[0].set_ylim(0.3, ny + .7); ax[0].set_xlim(0.5, 1.0); ax[1].set_xlim(0, 0.8)
for s, lab in [("th0", r"$\theta=0$"), ("gap", "mass gap"), ("kontr", "contrastive")]:
    ax[1].errorbar([], [], color=INK2, label=lab, ms=3.5, **MK[s])
for e in ["7", "5", "2"]:
    ax[1].plot([], [], "s", color=ENC[e], label=f"{e} qubits")
fig.legend(*ax[1].get_legend_handles_labels(), loc="upper center", ncol=6, fontsize=6.5, bbox_to_anchor=(0.55, 0.95))
save(fig, "fig_metrics")

# ---------------------------------------------------------------- Rys. 4: ROC i PR
curves = [("U2: $T^2$, mass gap", "U2", "T2__wyucz", ENC["2"], "-"), ("D2: $T^2$, $\\theta=0$", "D2", "T2__th0", ENC["2"], "--"),
          ("D5: $a_0$, contrastive", "D5", "a0__kontr", ENC["5"], "-"), ("D7: $a_0$, contrastive", "D7", "a0__kontr", ENC["7"], "-"),
          ("B7: $s$, mass gap", "B7", "s__wyucz", ENC["7"], ":")]
fig, ax = plt.subplots(1, 2, figsize=(W1, 2.7))
for lab, c, col, colr, ls in curves:
    r = res[c]; e_ = r.eval_mask.astype(bool).to_numpy(); yy = r.halo_kin.astype(int).to_numpy()[e_]
    sc_ = r[col].to_numpy()[e_]
    f, t_, _ = roc_curve(yy, sc_); p_, r_, _ = precision_recall_curve(yy, sc_)
    ax[0].plot(f, t_, color=colr, lw=1.1, ls=ls, label=lab); ax[1].plot(r_, p_, color=colr, lw=1.1, ls=ls)
for lab, col, ls in [(r"$-z_{\rm Fe}$", "klas__−z_Fe", "--"), ("log. regression (7)", "klas__regresja logistyczna", ":")]:
    r = res["D7"]; e_ = r.eval_mask.astype(bool).to_numpy(); yy = r.halo_kin.astype(int).to_numpy()[e_]
    f, t_, _ = roc_curve(yy, r[col].to_numpy()[e_]); p_, r_, _ = precision_recall_curve(yy, r[col].to_numpy()[e_])
    ax[0].plot(f, t_, color=INK, lw=.9, ls=ls, label=lab); ax[1].plot(r_, p_, color=INK, lw=.9, ls=ls)
ax[0].plot([0, 1], [0, 1], color=GRID, lw=.6); ax[1].axhline(0.046, color=INK2, lw=.6, ls=":")
ax[0].set(xlabel="false positive rate", ylabel="true positive rate")
ax[1].set(xlabel="recall", ylabel="precision", ylim=(0, 1.02))
ax[0].legend(fontsize=6, loc="lower right")
save(fig, "fig_roc_pr")

# ------------------------------------------------------- Rys. 5: decyzje (MCC)
rng = np.random.default_rng(1)


def boot_mcc(yv, fl, B_=2000):
    n = len(yv); out = []
    for _ in range(B_):
        i = rng.integers(0, n, n)
        if yv[i].sum() == 0: continue
        out.append(matthews_corrcoef(yv[i], fl[i]))
    return np.percentile(out, [2.5, 97.5])


def drow(cell, key, lab, color):
    r = deci[cell]; r = r[r.decyzja == key].iloc[0]
    return (lab, color, r.MCC, r.MCC_lo, r.MCC_hi, int(r.n_oznaczonych), r.precyzja, r.czulosc, r.olbrzymy)


OR = "QIFT reguła oryginalna (F0≤q, E≥q)"; T2 = "QIFT T2 > próg (α=0.05)"; A0 = "QIFT a0 > próg (α=0.05)"
drows = [drow("U7", OR, "U7: original rule", ENC["7"]), drow("U5", OR, "U5: original rule", ENC["5"]),
         drow("U2", T2, r"U2: $T^2>t_{0.05}$", ENC["2"]), drow("U2", OR, "U2: original rule", ENC["2"]),
         drow("D7", OR, "D7 contr.: original rule", ENC["7"]), drow("D7", A0, r"D7 contr.: $a_0>t_{0.05}$", ENC["7"]),
         drow("D5", OR, "D5 contr.: original rule", ENC["5"]), drow("D5", T2, r"D5 contr.: $T^2>t_{0.05}$", ENC["5"]),
         drow("D2", T2, r"D2 ($\theta=0$): $T^2>t_{0.05}$", ENC["2"]), drow("D2", OR, r"D2 ($\theta=0$): original rule", ENC["2"]),
         drow("G7", OR, "D7 mass gap: original rule", ENC["7"]),
         drow("B7", "QIFT s > 0 (F_halo > F_dysk)", "B7: $s>0$", ENC["7"]), drow("B5", "QIFT s > 0 (F_halo > F_dysk)", "B5: $s>0$", ENC["5"]),
         drow("B2", "QIFT s > 0 (F_halo > F_dysk)", "B2: $s>0$", ENC["2"])]
crows = [drow("U2", "Mahalanobis (MCD) > próg", "MCD (U2)", INK2), drow("U2", "Isolation Forest > próg", "Isol. Forest (U2)", INK2),
         drow("D7", "LOF > próg", "LOF (D7)", INK2), drow("D7", "regresja logistyczna > 0", "Log. regr. (7) > 0", INK2),
         drow("D5", "QDA > 0", "QDA (5) > 0", INK2)]
zf = res["D7"]["klas__−z_Fe"].to_numpy()[res["D7"].eval_mask.astype(bool).to_numpy()]
yD = res["D7"].halo_kin.astype(int).to_numpy()[res["D7"].eval_mask.astype(bool).to_numpy()]
for nb in [45, 46, 48]:
    top = np.zeros(len(zf), bool); top[np.argsort(-zf)[:nb]] = True
    lo, hi = boot_mcc(yD, top)
    NUM[f"zfe_top{nb}"] = dict(TP=int((top & (yD == 1)).sum()), MCC=float(matthews_corrcoef(yD, top)), lo=float(lo), hi=float(hi))
z45 = NUM["zfe_top45"]
allr = drows + crows + [(r"$-z_{\rm Fe}$, top 45", INK, z45["MCC"], z45["lo"], z45["hi"], 45, z45["TP"] / 45, z45["TP"] / 50, np.nan)]
fig, ax = plt.subplots(figsize=(W1 * .85, 4.6))
for i, (lab, c, m, l_, h_, n_, p_, r_, g_) in enumerate(allr):
    yy = len(allr) - i
    ax.errorbar(m, yy, xerr=[[m - l_], [h_ - m]], fmt="o" if c not in (INK, INK2) else "s", ms=3.8, color=c, lw=.8)
    ax.text(0.83, yy, f"{n_:d} / {p_:.2f} / {r_:.2f}", fontsize=6, va="center", color=INK2)
ax.text(0.83, len(allr) + .9, r"$N_{\rm flag}$ / prec. / rec.", fontsize=6, color=INK2)
ax.set_yticks(range(len(allr), 0, -1)); ax.set_yticklabels([r[0] for r in allr])
ax.set_xlim(0, 1.05); ax.set_xlabel("Matthews correlation coefficient"); ax.set_ylim(.3, len(allr) + 1.4)
ax.axhline(len(allr) - len(drows) + .5, color=INK2, lw=.5)
save(fig, "fig_decisions")
NUM["decisions"] = {r[0]: dict(MCC=r[2], lo=r[3], hi=r[4], n=r[5], prec=r[6], rec=r[7], giants=r[8]) for r in allr}

# ------------------------------------------------ Rys. 6: efekt topologii (Δ vs θ = 0)
def prow(cell, measure, which):
    p = pair[cell]
    if cell[0] == "B":
        name = "QIFT s = F_halo − F_dysk [wyuczona (przerwa masowa)]  −  QIFT s = F_halo − F_dysk [θ = 0]"
    else:
        tag = {"gap": TAG["gap"][cell[0]], "kontr": TAG["kontr"]}[which]
        name = f"QIFT {measure} [{tag}]  −  QIFT {measure} [θ = 0]"
    m = p[p.porownanie == name]
    return m.iloc[0] if len(m) else None


trows = []
for e in ["7", "5", "2"]:
    trows += [(f"U{e}: $a_0$", e, "gap", prow("U" + e, "a0", "gap")), (f"U{e}: $T^2$", e, "gap", prow("U" + e, "T²", "gap"))]
    trows += [(f"D{e}: $a_0$, gap", e, "gap", prow("D" + e, "a0", "gap")), (f"D{e}: $a_0$, contr.", e, "kontr", prow("D" + e, "a0", "kontr")),
              (f"D{e}: $T^2$, contr.", e, "kontr", prow("D" + e, "T²", "kontr")), (f"B{e}: $s$", e, "gap", prow("B" + e, "s", "gap"))]
# D: T² mass gap − θ0 nie ma w porównaniach przebiegów kontrastowych → sparowany bootstrap z wyników gwiazd
from sklearn.metrics import roc_auc_score, average_precision_score
def paired(cell, a, b, B_=1000):
    r = res[cell]; e_ = r.eval_mask.astype(bool).to_numpy(); yy = r.halo_kin.astype(int).to_numpy()[e_]
    xa, xb = r[a].to_numpy()[e_], r[b].to_numpy()[e_]; g = np.random.default_rng(7); da, dp = [], []
    for _ in range(B_):
        i = g.integers(0, len(yy), len(yy))
        if yy[i].sum() == 0: continue
        da.append(roc_auc_score(yy[i], xa[i]) - roc_auc_score(yy[i], xb[i]))
        dp.append(average_precision_score(yy[i], xa[i]) - average_precision_score(yy[i], xb[i]))
    return pd.Series(dict(dAUC=np.median(da), dAUC_lo=np.percentile(da, 2.5), dAUC_hi=np.percentile(da, 97.5),
                          dAP=np.median(dp), dAP_lo=np.percentile(dp, 2.5), dAP_hi=np.percentile(dp, 97.5)))
tr2 = []
for lab, e, w, r in trows:
    tr2.append((lab, e, w, r))
    if lab.startswith("D") and "a_0" in lab and w == "gap":
        tr2.append((f"D{e}: $T^2$, gap", e, "gap", paired("D" + e, "T2__gap", "T2__th0")))
trows = [t for t in tr2 if t[3] is not None]
fig, ax = plt.subplots(1, 2, figsize=(W1, 4.3), sharey=True)
for k, (m, lo_, hi_) in enumerate([("dAUC", "dAUC_lo", "dAUC_hi"), ("dAP", "dAP_lo", "dAP_hi")]):
    for i, (lab, e, w, r) in enumerate(trows):
        yy = len(trows) - i
        sig = (r[lo_] > 0) or (r[hi_] < 0)
        ax[k].errorbar(r[m], yy, xerr=[[r[m] - r[lo_]], [r[hi_] - r[m]]], fmt="D" if w == "kontr" else "o", ms=3.8,
                       color=ENC[e], mfc=ENC[e] if sig else "white", lw=.8)
    ax[k].axvline(0, color=INK, lw=.7)
    ax[k].set_xlabel(r"$\Delta$AUC (topology $-$ $\theta{=}0$)" if k == 0 else r"$\Delta$AP (topology $-$ $\theta{=}0$)")
ax[0].set_yticks(range(len(trows), 0, -1)); ax[0].set_yticklabels([t[0] for t in trows])
ax[1].errorbar([], [], fmt="o", color=INK2, label="mass gap"); ax[1].errorbar([], [], fmt="D", color=INK2, label="contrastive")
ax[1].plot([], [], "o", mfc="white", color=INK2, label="95% CI includes 0")
fig.legend(*ax[1].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.55, 1.03))
save(fig, "fig_topology")
NUM["topology_delta"] = [dict(row=t[0], which=t[2], dAUC=float(t[3].dAUC), dAUC_lo=float(t[3].dAUC_lo), dAUC_hi=float(t[3].dAUC_hi),
                              dAP=float(t[3].dAP), dAP_lo=float(t[3].dAP_lo), dAP_hi=float(t[3].dAP_hi)) for t in trows]

# --------------------------------------------- Rys. 7: kąt vs różnica halo–dysk
TC = {"podobna–podobna": ("#2a78d6", "similar–similar"), "różna–podobna": ("#eb6834", "different–similar"),
      "różna–różna": (INK, "different–different")}
SHORT = {"w_atm_1": r"$\log g$", "w_atm_2": r"$T_{\rm eff}$", "z_Fe": "Fe", "z_alpha": r"$\alpha$", "z_C": "C", "z_N": "N", "z_act": "act"}
fig, ax = plt.subplots(1, 2, figsize=(W1, 2.8), sharey=True)
for a, cell, ttl in [(ax[0], "D5", "D5 (five chemical qubits)"), (ax[1], "D7", "D7 (seven qubits)")]:
    T = pd.read_csv(RUN[cell] + "/topologia_vs_roznice_CV.csv")
    for typ, g in T.groupby("typ"):
        c, lab = TC[typ]
        a.errorbar(g.max_abs_delta, g.kat_deg, yerr=g.kat_std_deg, fmt="o", ms=4, color=c, lw=.7, capsize=1.5, label=lab)
    for _, r in T.iterrows():
        i, j = r.para.split("–")
        if "z_Fe" in (i, j) or (i == "w_atm_1" and j in ("z_alpha", "z_C", "z_act")):
            continue
        if r.kat_deg > 12 or cell == "D5":
            a.annotate(f"{SHORT[i]}–{SHORT[j]}", (r.max_abs_delta, r.kat_deg), fontsize=5.5, xytext=(3, 2),
                       textcoords="offset points", color=INK2)
    a.annotate("Fe–all", (0.8935, 3), fontsize=5.5, xytext=(-28, 6), textcoords="offset points", color=INK2)
    if cell == "D7":
        a.annotate(r"$\log g$–($\alpha$, C, act)", (0.5774, 21.5), fontsize=5.5, xytext=(4, 4), textcoords="offset points", color=INK2)
    a.axvline(0.40, color=INK2, lw=.6, ls=":")
    rho, p = stats.spearmanr(T.max_abs_delta, T.kat_deg)
    a.set_title(f"{ttl}\nSpearman {rho:+.2f} (p = {p:.3f})", fontsize=7.5)
    a.set_xlabel(r"max $|\delta|$ of the pair (halo vs. disc)")
    NUM[f"theory_{cell}"] = dict(rho=float(rho), p=float(p), **{f"mean_{TC[k][1]}": float(g.kat_deg.mean()) for k, g in T.groupby("typ")})
ax[0].set_ylabel("coupling angle [deg]"); ax[0].set_ylim(-4, 92); ax[1].legend(fontsize=6, loc="upper right")
save(fig, "fig_theory")

# ------------------------------------------------------- Rys. 8: Kiel – oznaczone
panels = [("U7", "dec__QIFT reguła oryginalna (F0≤q, E≥q)", "U7: original rule"),
          ("U2", "dec__QIFT T2 > próg (α=0.05)", r"U2: $T^2>t_{0.05}$"),
          ("D2", "dec__QIFT T2 > próg (α=0.05)", r"D2: $T^2>t_{0.05}$"),
          ("D5", "dec__QIFT reguła oryginalna (F0≤q, E≥q)", "D5 contr.: orig. rule"),
          ("D7", "dec__QIFT reguła oryginalna (F0≤q, E≥q)", "D7 contr.: orig. rule"),
          ("B7", "dec__QIFT s > 0 (F_halo > F_dysk)", "B7: $s>0$")]
fig, axs = plt.subplots(2, 3, figsize=(W1, 3.9), sharex=True, sharey=True)
for a, (cell, key, ttl) in zip(axs.ravel(), panels):
    r = res[cell]; fl = r[key].astype(bool) & r.eval_mask.astype(bool)
    a.scatter(r.teff, r.logg, s=2, color="#c9c8c3", lw=0)
    a.scatter(r.teff[fl], r.logg[fl], s=6, color=ENC[cell[1]], lw=0)
    h = (r.klasa_kin == "halo") & r.eval_mask.astype(bool)
    a.scatter(r.teff[h], r.logg[h], s=16, facecolors="none", edgecolors=INK, lw=.5)
    a.set_title(f"{ttl} ({fl.sum()})", fontsize=7); a.set_xlim(6700, 4000); a.set_ylim(5.1, 0.4)
for a in axs[1]: a.set_xlabel(r"$T_{\rm eff}$ [K]")
for a in axs[:, 0]: a.set_ylabel(r"$\log g$")
save(fig, "fig_kiel_flagged")

# ------------------------------------------------------ Rys. 9: splątanie S(A|B)
from qift_analiza_wspolne import encode as enc, entanglement_entropy as SA, fold_mean_angles
Sx = d[["s_" + c for c in L]].to_numpy(float)
ANG = {"D7, mass gap": fold_mean_angles("p1b_1klasa_dysk_7q"),
       "B7, mass gap": fold_mean_angles("p2_2klasy_7q"),
       "D7, contrastive": fold_mean_angles("k1_kontrast_wszystkie_7q")}
evm = d.eval_mask.astype(bool); dkm = evm & (d.klasa_kin == "dysk"); hkm = evm & (d.halo_kin == 1)
dw = d.logg >= 4; gi = d.logg < 3.5
fig, axs = plt.subplots(1, 3, figsize=(W1, 2.5), sharey=True)
NUM["entanglement"] = {}
for a, (nm, ang) in zip(axs, ANG.items()):
    s = SA(enc(Sx, ang)); pos = 0; NUM["entanglement"][nm] = {}
    for grp, m in [("dwarfs", dw), ("giants", gi)]:
        hh, dd = s[hkm & m], s[dkm & m]
        bp = a.boxplot([dd, hh], positions=[pos, pos + .8], widths=.6, patch_artist=True, showfliers=False,
                       medianprops=dict(color=INK, lw=1))
        for patch, c in zip(bp["boxes"], ["#c9c8c3", "#eb6834"]):
            patch.set_facecolor(c); patch.set_edgecolor(INK2)
        u = stats.mannwhitneyu(hh, dd); dl = 2 * u.statistic / len(hh) / len(dd) - 1
        a.text(pos + .4, 2.08, f"$\\delta$={dl:+.2f}\np={u.pvalue:.1g}", ha="center", fontsize=6, color=INK2)
        NUM["entanglement"][nm][grp] = dict(S_halo=float(np.median(hh)), S_disc=float(np.median(dd)), delta=float(dl),
                                            p=float(u.pvalue), n_h=int(len(hh)), n_d=int(len(dd)))
        pos += 2
    a.set_xticks([0, .8, 2, 2.8]); a.set_xticklabels(["disc", "halo", "disc", "halo"], fontsize=6.5)
    a.text(.4, -.38, "dwarfs", ha="center", fontsize=7, transform=a.transData); a.text(2.4, -.38, "giants", ha="center", fontsize=7)
    a.set_title(nm, fontsize=7.5); a.set_ylim(-.05, 2.35)
axs[0].set_ylabel(r"$S(A|B)$ [bit]")
save(fig, "fig_entanglement")

json.dump(NUM, open("artykul/numbers_v3.json", "w"), indent=1, default=float)
print(json.dumps({k: NUM[k] for k in ["latent_r_target", "latent_max_excess", "zfe_top45", "zfe_top46", "zfe_top48",
                                      "theory_D5", "theory_D7", "entanglement"]}, indent=1, default=float))
