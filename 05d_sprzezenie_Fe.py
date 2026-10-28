"""KROK 05d (artykuł, Sect. 4.4, H2b): czy sprzęganie z_Fe szkodzi?
AUC(a0: halo vs dysk) w próbie dla topologii kontrastowej 5q (średnie kąty z foldów), θ = 0 oraz wariantów
z wymuszonymi sprzężeniami z_Fe (20°, 45°, 90°) i z samymi parami zmiennych podobnych.
Uruchomienie: python 05d_sprzezenie_Fe.py
"""
from qift_analiza_wspolne import encode, fold_mean_angles
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
d=pd.read_csv("qift_reports/p1b_1klasa_dysk_7q/wyniki_gwiazd.csv")
F=["z_Fe","z_alpha","z_C","z_N","z_act"]; X=d[["s_"+c for c in F]].to_numpy(float)
ev=d.eval_mask.astype(bool).to_numpy(); dk=ev&(d.klasa_kin=="dysk").to_numpy(); hk=ev&(d.halo_kin==1).to_numpy()
pairs=[(i,j) for i in range(5) for j in range(i+1,5)]
def auc(ang):
    P=encode(X,ang); rho=(P[dk].T@P[dk].conj())/dk.sum(); l,V=np.linalg.eigh(rho); v=V[:,-1]
    a0=1-np.abs(P@v.conj())**2; return roc_auc_score(np.r_[np.zeros(dk.sum()),np.ones(hk.sum())],np.r_[a0[dk],a0[hk]])
base=np.array(fold_mean_angles("k3_kontrast_chemia_5q"))  # Fe–α, Fe–C, Fe–N, Fe–act, α–C, ...
print("θ=0:",round(auc(np.zeros(10)),4),"kontrast:",round(auc(base),4))
for t in [20,45,90]:
    a=base.copy(); a[:4]=t; print(f"Fe sprzężone {t}°:",round(auc(a),4))
a=np.zeros(10); a[4:]=base[4:]; print("tylko pary podobne:",round(auc(a),4))
a=np.zeros(10); a[:4]=45; print("tylko Fe 45°:",round(auc(a),4))
