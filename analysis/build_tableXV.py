"""Collect every Holm-significant comparison across the experiments into one table."""
import pandas as pd, numpy as np
def holm(p):
    p=np.asarray(p,float); m=len(p); o=np.argsort(p); adj=np.empty(m); run=0
    for k,i in enumerate(o): run=max(run,(m-k)*p[i]); adj[i]=min(run,1.0)
    return adj
def pf(p): return "<0.001" if p<0.001 else (f"{p:.3f}" if p>=0.01 else f"{p:.2e}".replace("e-0","e-"))
rows=[]
# --- screen (Table VII source)
s=pd.read_csv("IN_paired_wilcoxon_vs_LeakyReLU.csv"); s=s[s.sig]
name={"HardElish":"HardELiSH","Elish":"ELiSH"}
for _,r in s.iterrows():
    d=f"{r.delta:+.4f}" if r.metric=="Dice" else f"{r.delta:+.2f} mm"
    rows.append(["In-domain screen (IN)", f"IN + {name.get(r.activation,r.activation)} vs IN + LeakyReLU", r.region, "Dice" if r.metric=="Dice" else "HD95", d, pf(r.p), pf(r.p_holm), "worse"])
# --- factorial
c=pd.read_csv("../CLEAN219_table_cells_2026-06-14.csv"); base=c[c.model=="IN+LeakyReLU"].set_index(["metric","region"])["mean"]
c=c[c.model!="IN+LeakyReLU"].copy(); c["delta"]=[r["mean"]-base[(r.metric,r.region)] for _,r in c.iterrows()]
c["p_holm"]=np.nan
for (m,mod),g in c.groupby(["metric","model"]): c.loc[g.index,"p_holm"]=holm(g.pW.values)
assert ((c.p_holm<0.05)==(c.holmW_sig=="*")).all(), "Holm rule mismatch"
order=["IN+PReLU","IN+Swish","IN+TanhExp","BN+LeakyReLU","BN+PReLU","BN+Swish","BN+TanhExp","GN+LeakyReLU","GN+PReLU","GN+Swish","GN+TanhExp"]
c["o"]=c.model.map({m:i for i,m in enumerate(order)}); c["mo"]=c.metric.map({"DSC":0,"HD95":1}); c["ro"]=c.region.map({"ET":0,"TC":1,"WT":2})
for _,r in c[c.p_holm<0.05].sort_values(["mo","o","ro"]).iterrows():
    d=f"{r.delta:+.4f}" if r.metric=="DSC" else f"{r.delta:+.2f} mm"
    worse=(r.delta<0) if r.metric=="DSC" else (r.delta>0)
    rows.append(["Factorial", f"{r.model.replace('+',' + ')} vs IN + LeakyReLU", r.region, "Dice" if r.metric=="DSC" else "HD95", d, pf(r.pW), pf(r.p_holm), "worse" if worse else "better"])
# --- batch size
w=pd.read_csv("batchsize_wilcoxon_219.csv"); w=w[(w.metric!="FP")&(w.p_holm<0.05)&(w.comparison.isin(["BN@8 vs BN@2","GN@8 vs GN@2","BN@8 vs IN@2"]))]
lab={"BN@8 vs BN@2":"BN + LeakyReLU batch 8 vs batch 2","GN@8 vs GN@2":"GN + LeakyReLU batch 8 vs batch 2 (control)","BN@8 vs IN@2":"BN + LeakyReLU batch 8 vs IN + LeakyReLU"}
w["co"]=w.comparison.map({"BN@8 vs BN@2":0,"GN@8 vs GN@2":1,"BN@8 vs IN@2":2}); w["mo"]=w.metric.map({"DSC":0,"HD95":1}); w["ro"]=w.region.map({"ET":0,"TC":1,"WT":2})
for _,r in w.sort_values(["co","mo","ro"]).iterrows():
    d=f"{r.delta:+.4f}" if r.metric=="DSC" else f"{r.delta:+.2f} mm"
    worse=(r.delta<0) if r.metric=="DSC" else (r.delta>0)
    rows.append(["Batch size", lab[r.comparison], r.region, "Dice" if r.metric=="DSC" else "HD95", d, pf(r.p_raw), pf(r.p_holm), "worse" if worse else "better"])
T=pd.DataFrame(rows,columns=["Experiment","Comparison","Region","Metric","Δ (first − second)","p (raw)","p (Holm)","Direction"])
T.to_csv("tableXV_consolidated.csv",index=False); print(T.to_string()); print(len(T),"rows; better:",(T.Direction=="better").sum())
print("WT rows:", T[T.Region=="WT"][["Experiment","Comparison","Metric","Δ (first − second)","Direction"]].to_string())
