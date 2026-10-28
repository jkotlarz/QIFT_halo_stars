"""KROK 05c (artykuł, Sect. 4.5, H3): splątanie atmosfera–chemia dla 7 kubitów.
Entropia S(A|B) każdej gwiazdy dla czterech topologii (kąty odczytane z raportów kroku 04), porównanie halo/dysk
osobno dla karłów i olbrzymów, test czy S dodaje informację ponad latenty (regresja logistyczna, CV),
korelacje S z latentami wśród karłów.  Uruchomienie: python 05c_splatanie_kontrola.py
"""
from qift_analiza_wspolne import encode, entanglement_entropy, fold_mean_angles
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
d=pd.read_csv("qift_reports/p1b_1klasa_dysk_7q/wyniki_gwiazd.csv")
L=["w_atm_1","w_atm_2","z_Fe","z_alpha","z_C","z_N","z_act"]
Sx=d[["s_"+c for c in L]].to_numpy(float)
ANG={"1a przerwa masowa":fold_mean_angles("p1a_1klasa_cala_7q"),"1b przerwa masowa":fold_mean_angles("p1b_1klasa_dysk_7q"),
     "2 przerwa masowa":fold_mean_angles("p2_2klasy_7q"),"k1 kontrastowa":fold_mean_angles("k1_kontrast_wszystkie_7q")}
ev=d.eval_mask.astype(bool); y=d.halo_kin.astype(int); dw=d.logg>=4; gi=d.logg<3.5
dk=ev&(d.klasa_kin=="dysk"); hk=ev&(y==1)
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def cvauc(X,yy):
    m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)); p=cross_val_predict(m,X,yy,cv=cv,method="predict_proba")[:,1]
    return roc_auc_score(yy,p),average_precision_score(yy,p)
E=ev.to_numpy(); Xl=d[L].to_numpy(float)
print("LR 7 latentów:", np.round(cvauc(Xl[E],y[E]),3))
for k,a in ANG.items():
    s=entanglement_entropy(encode(Sx,a)); d["S_"+k]=s
    r=[]
    for nm,m in [("wszystkie",slice(None)),("karły",dw),("olbrzymy",gi)]:
        hh=s[(hk&m) if not isinstance(m,slice) else hk]; dd=s[(dk&m) if not isinstance(m,slice) else dk]
        u=stats.mannwhitneyu(hh,dd); r.append(f"{nm}: S_h={np.median(hh):.2f} S_d={np.median(dd):.2f} δ={2*u.statistic/len(hh)/len(dd)-1:+.2f} p={u.pvalue:.2g} (n={len(hh)}/{len(dd)})")
    a1=cvauc(Xl[E],y[E]); a2=cvauc(np.c_[Xl,s][E],y[E])
    D=(E&dw.to_numpy()); b1=cvauc(Xl[D][:,[2]],y[D]); b2=cvauc(np.c_[Xl[:,[2]],s][D],y[D])
    rho={c:stats.spearmanr(s[dw],d.loc[dw,c])[0] for c in L}
    print(f"\n== {k}\n  "+"\n  ".join(r))
    print(f"  LR latenty {a1[0]:.3f}/{a1[1]:.3f} -> +S {a2[0]:.3f}/{a2[1]:.3f};  karły: z_Fe {b1[0]:.3f}/{b1[1]:.3f} -> z_Fe+S {b2[0]:.3f}/{b2[1]:.3f}")
    print("  Spearman S~latent (karły):",{c:round(v,2) for c,v in rho.items()})
# halo vs disc dwarfs: latent differences
print("\nKarły halo vs dysk – mediany latentów:")
for c in L: 
    u=stats.mannwhitneyu(d.loc[hk&dw,c],d.loc[dk&dw,c]); print(f"  {c}: {d.loc[hk&dw,c].median():+.2f} vs {d.loc[dk&dw,c].median():+.2f}  δ={2*u.statistic/((hk&dw).sum()*(dk&dw).sum())-1:+.2f}")
