"""Batch-size experiment: paired Wilcoxon on the 219 blinded Synapse cases.
BN@8 vs BN@2, GN@8 vs GN@2 (primary); each @8 cell vs IN@2 baseline (secondary). Holm within metric family."""
import pandas as pd, numpy as np
from math import erf, sqrt
def _rankdata(a):
    a=np.asarray(a); order=np.argsort(a,kind="mergesort"); ranks=np.empty(len(a)); sa=a[order]
    i=0
    while i<len(a):
        j=i
        while j+1<len(a) and sa[j+1]==sa[i]: j+=1
        ranks[order[i:j+1]]=(i+j)/2+1; i=j+1
    return ranks
def wilcoxon_p(x,y):
    """Wilcoxon signed-rank, zeros dropped (wilcox), normal approximation with tie correction, no continuity correction (scipy approx default)."""
    d=np.asarray(x)-np.asarray(y); d=d[d!=0]; n=len(d)
    if n==0: return 1.0
    r=_rankdata(np.abs(d)); Wp=r[d>0].sum(); Wm=r[d<0].sum(); W=min(Wp,Wm)
    mn=n*(n+1)/4; var=n*(n+1)*(2*n+1)/24
    _,cnt=np.unique(r,return_counts=True); var-= (cnt**3-cnt).sum()/48
    if var<=0: return 1.0
    z=(W-mn)/sqrt(var); return 2*(0.5*(1+erf(-abs(z)/sqrt(2))))
def holm(p):
    p=np.asarray(p,float); m=len(p); o=np.argsort(p); adj=np.empty(m); run=0
    for k,i in enumerate(o):
        run=max(run,(m-k)*p[i]); adj[i]=min(run,1.0)
    return adj
S="Synapse/"
def load(f):
    d=pd.read_csv(S+f,index_col=0); d=d[d.index.str.startswith("BraTS")]; return d.sort_index()
C={"IN@2":load("LeakyReLU_IN.csv"),"BN@2":load("BN_LeakyReLU.csv"),"GN@2":load("GN_LeakyReLU.csv"),
   "BN@8":load("BN_LeakyReLU_bs8.csv"),"GN@8":load("GN_LeakyReLU_bs8.csv")}
idx=C["IN@2"].index
for k,v in C.items(): assert len(v)==219 and (v.index==idx).all(), k
M={"DSC":"LesionWise_Dice_{}","HD95":"LesionWise_Hausdorff95_{}","FP":"Num_FP_{}"}
R=["ET","TC","WT"]
# ---- summary table
rows=[]
for k,v in C.items():
    r={"cell":k}
    for reg in R:
        d=v[M["DSC"].format(reg)]; h=v[M["HD95"].format(reg)]; fp=v[M["FP"].format(reg)]
        r[f"DSC_{reg}_mean"]=d.mean(); r[f"DSC_{reg}_sd"]=d.std()
        r[f"HD95_{reg}_med"]=h.median(); r[f"HD95_{reg}_q1"]=h.quantile(.25); r[f"HD95_{reg}_q3"]=h.quantile(.75)
        r[f"HD95_{reg}_fail50"]=(h>50).mean()*100; r[f"HD95_{reg}_mean"]=h.mean()
        r[f"FP_{reg}_mean"]=fp.mean(); r[f"FP_{reg}_total"]=fp.sum()
    rows.append(r)
summ=pd.DataFrame(rows).set_index("cell"); summ.to_csv("batchsize_summary_219.csv")
# ---- paired tests
pairs=[("BN@8","BN@2"),("GN@8","GN@2"),("BN@8","IN@2"),("GN@8","IN@2"),("BN@8","GN@8")]
out=[]
for a,b in pairs:
    for m,pat in M.items():
        for reg in R:
            x=C[a][pat.format(reg)].values; y=C[b][pat.format(reg)].values; d=x-y
            nz=(d!=0).sum()
            p=wilcoxon_p(x,y)
            out.append(dict(comparison=f"{a} vs {b}",metric=m,region=reg,mean_a=x.mean(),mean_b=y.mean(),delta=x.mean()-y.mean(),
                            median_delta=np.median(d),n_nonzero=nz,p_raw=p))
T=pd.DataFrame(out)
T["p_holm"]=np.nan
for (c,m),g in T.groupby(["comparison","metric"]):
    T.loc[g.index,"p_holm"]=holm(g["p_raw"].values)
T["sig"]=np.where(T.p_holm<0.05,"*","")
T.to_csv("batchsize_wilcoxon_219.csv",index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
print(summ[[c for c in summ.columns if "mean" in c and ("DSC" in c or "FP" in c)]].round(4))
print(summ[[c for c in summ.columns if "HD95" in c and ("med" in c or "fail" in c)]].round(2))
print(T[["comparison","metric","region","mean_a","mean_b","delta","p_raw","p_holm","sig"]].round(4).to_string())
