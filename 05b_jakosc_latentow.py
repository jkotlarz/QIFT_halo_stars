"""KROK 05b (artykuł, Sect. 4.1): jakość latentów.
Korelacje latentów z kotwicami, przeciek ponad korelację etykiet katalogu, korelacje cząstkowe przy stałym celu
i post-hoc test adwersarialny (czy Teff i log g da się odtworzyć z latentów chemicznych tak samo jak z etykiet).
Wejście: qift_reports/p1b_1klasa_dysk_7q/wyniki_gwiazd.csv (1352 gwiazdy, latenty + etykiety + kotwice).
Wyjście: wydruk + artykul/05b_przeciek_latentow.csv.  Uruchomienie: python 05b_jakosc_latentow.py
"""
from pathlib import Path
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
d=pd.read_csv("qift_reports/p1b_1klasa_dysk_7q/wyniki_gwiazd.csv")
d["c_fe_kotw"]=d.kotwica_C; d["n_fe_kotw"]=d.kotwica_N; d["akt_kotw"]=d.kotwica_akt
print("unikalne gwiazdy:",len(d))
L=["w_atm_1","w_atm_2","z_Fe","z_alpha","z_C","z_N","z_act"]
P=["teff","logg","feh","alpha_m","c_fe_kotw","n_fe_kotw","akt_kotw"]
T={"w_atm_1":"logg","w_atm_2":"teff","z_Fe":"feh","z_alpha":"alpha_m","z_C":"c_fe_kotw","z_N":"n_fe_kotw","z_act":"akt_kotw"}
r=lambda a,b: d[[a,b]].dropna().corr().iloc[0,1]
def pc(x,y,z):
    s=d[[x,y,z]].dropna(); 
    rx=s[x]-np.polyval(np.polyfit(s[z],s[x],1),s[z]); ry=s[y]-np.polyval(np.polyfit(s[z],s[y],1),s[z]); return np.corrcoef(rx,ry)[0,1]
rows=[]
for l in L:
  for p in P:
    if p==T[l]: continue
    rows.append(dict(latent=l,param=p,r_latent=r(l,p),r_kotwica=r(T[l],p),r_czastkowa=pc(l,p,T[l])))
R=pd.DataFrame(rows); R["nadwyzka_r2"]=R.r_latent**2-R.r_kotwica**2
print("\nn kotwic:",{p:int(d[p].notna().sum()) for p in P})
print("\nr z celem:",{l:round(r(l,T[l]),3) for l in L})
print("\nPrzeciek: największe |r_cząstkowa| (latent~param | cel) i nadwyżka r² ponad korelację katalogową")
print(R.reindex(R.r_czastkowa.abs().sort_values(ascending=False).index).head(14).round(3).to_string(index=False))
Path("artykul").mkdir(exist_ok=True); R.to_csv("artykul/05b_przeciek_latentow.csv",index=False)
# test adwersarialny post hoc
cv=KFold(5,shuffle=True,random_state=0)
ALL=["z_Fe","z_alpha","z_C","z_N","z_act","feh","alpha_m","c_fe_kotw","n_fe_kotw","akt_kotw"]
def r2(X,y):
  s=d[ALL+[y]].dropna(); out=[]
  for m in [LinearRegression(),HistGradientBoostingRegressor(max_iter=200)]:
    out.append(cross_val_score(m,s[X],s[y],cv=cv,scoring="r2").mean())
  return out,len(s)
print("\nTest adwersarialny post hoc: R² (CV, liniowy / GBM) odtworzenia atmosfery")
for y in ["teff","logg"]:
  for nm,X in [("latenty Fe,α",["z_Fe","z_alpha"]),("etykiety [Fe/H],[α/M]",["feh","alpha_m"]),
               ("5 latentów chem.",["z_Fe","z_alpha","z_C","z_N","z_act"]),("5 etykiet chem.",["feh","alpha_m","c_fe_kotw","n_fe_kotw","akt_kotw"])]:
    (a,b),n=r2(X,y); print(f"  {y:5s} z {nm:24s}: {a:.3f} / {b:.3f}  (n={n})")
