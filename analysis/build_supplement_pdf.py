#!/usr/bin/env python3
"""Render supplementary_full_matrix.csv as a standalone Supplementary Material PDF."""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "latex_revised", "supplementary.tex")

df = pd.read_csv(os.path.join(HERE, "supplementary_full_matrix.csv"))


def fp(p):
    if p != p:
        return "---"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def fd(v, metric):
    return f"{v:+.4f}" if metric == "Dice" else f"{v:+.2f}"


BLOCKS = [
    ("S1 in-domain screen", "S1",
     "In-Domain Activation Screen: All 66 Comparisons Against IN\\,+\\,LeakyReLU",
     "Every activation function versus the nnU-Net default, on the 219 blinded BraTS 2023 "
     "validation cases. Holm correction is applied within each metric family (33 tests). "
     "No comparison is favourable in either family."),
    ("S2 factorial", "S2",
     "Normalisation $\\times$ Activation Factorial: All 66 Comparisons Against IN\\,+\\,LeakyReLU",
     "Every factorial cell versus the baseline. Holm correction is applied across the three "
     "subregions of each model and metric, as in the main text."),
    ("S3 batch size", "S3",
     "Batch-Size Experiment: All 24 Paired Comparisons",
     "Holm correction is applied across the three subregions of each contrast and metric. "
     "The batch-8 runs use 62 iterations per epoch, so they take four times fewer optimiser "
     "steps for the same number of samples."),
    ("S4 Arm A normalisation (residual U-Net, BraTS 2020)", "S4",
     "Arm A Cross-Architecture Replication: All 100 Comparisons Against Instance Normalisation",
     "Residual 3D U-Net on BraTS 2020, seed 2025, 73 paired validation cases, five regions. "
     "Holm correction is applied across the five activations within each region and metric. "
     "Paired $t$-test values are not reported for this block."),
]

L = [r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{amsmath}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\setlength{\LTcapwidth}{\textwidth}
\renewcommand{\arraystretch}{1.08}
\begin{document}

\begin{center}
{\Large\bfseries Supplementary Material}\\[4pt]
{\large Activation Functions and Normalisation Strategies in nnU-Net:\\
A Systematic Study of Their Interaction for 3D Brain Tumour Segmentation}\\[6pt]
Mushtaq Mahyoob Saleh, Eltahir Mohamed Hussein, Mosab Elkheir Salih, Mohamed A. A. Ahmed
\end{center}

\section*{Scope}

This document contains the complete matrix of statistical comparisons underlying the paper,
including every test that does \emph{not} reach significance. The main text reports only the
comparisons that survive Holm correction; those appear here alongside all the others, so that
non-significant results can be inspected directly rather than inferred from their absence.

\section*{Methods}

All comparisons are patient-paired. Arm B comparisons use the 219 blinded BraTS 2023
validation cases scored by the official lesion-wise pipeline; the Arm A block uses the fixed
73-case BraTS 2020 validation split at seed 2025.

The primary test is the Wilcoxon signed-rank test, with zeros dropped. Because the
lesion-wise HD95 distribution is heavily right-skewed---missed and spurious lesions are
assigned 374\,mm---a rank-based test is the appropriate primary measure. The paired
Student's $t$-test is reported as a secondary marker only; where the two disagree, the
disagreement is itself informative and is driven by a small number of geometric outliers.

Holm correction is applied within the families stated in each table caption, matching the
main text exactly. $\Delta$ is the mean paired difference, first condition minus second.
``Better'' denotes higher Dice or lower HD95. $n$ is the number of non-zero paired
differences entering the Wilcoxon test.

\paragraph{Implementation note.} These values were computed with a direct implementation of
the Wilcoxon normal approximation (with tie correction, without continuity correction) and of
the Student's $t$ distribution function, rather than with \texttt{scipy}. Values may
therefore differ from a \texttt{scipy} computation in the third decimal place. The analysis
scripts are included in the public repository.

\section*{Summary}

\begin{center}
\begin{tabular}{@{}llrrr@{}}
\toprule
& \textbf{Block} & \textbf{Tests} & \textbf{Significant} & \textbf{Favourable} \\
\midrule
"""]

for key, tag, title, _ in BLOCKS:
    g = df[df.block == key]
    sig = g[g.significant == "yes"]
    L.append(f"{tag} & {title.split(':')[0]} & {len(g)} & {len(sig)} & "
             f"{(sig.direction == 'better').sum()} \\\\\n")
L.append(r"""\bottomrule
\end{tabular}
\end{center}

\vspace{4pt}
\noindent Of the 66 comparisons in the in-domain screen, none is favourable: no activation
function is significantly better than the nnU-Net default on any region, for either metric.

\clearpage
""")

for key, tag, title, note in BLOCKS:
    g = df[df.block == key].copy()
    g = g.sort_values(["metric", "region", "comparison"])
    has_t = g.p_ttest.notna().any()
    spec = "@{}p{6.6cm}ccrrrr" + ("r" if has_t else "") + "l@{}"
    hdr = (r"\textbf{Comparison} & \textbf{Reg.} & \textbf{Metric} & $\Delta$ & "
           r"\textbf{$p$ raw} & \textbf{$p$ Holm} & " +
           (r"\textbf{$p$ $t$-test} & " if has_t else "") + r"\textbf{$n$} & \textbf{Dir.}")
    ncol = 8 + (1 if has_t else 0)
    L.append(f"""
\\begin{{longtable}}{{{spec}}}
\\caption{{{title}. {note}}}\\label{{tab:{tag}}}\\\\
\\toprule
{hdr} \\\\
\\midrule
\\endfirsthead
\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{Table \\ref{{tab:{tag}}} continued}}}}\\\\
\\toprule
{hdr} \\\\
\\midrule
\\endhead
\\midrule
\\multicolumn{{{ncol}}}{{r@{{}}}}{{\\emph{{continued on the next page}}}}\\\\
\\endfoot
\\bottomrule
\\endlastfoot
""")
    for _, r in g.iterrows():
        star = "$^{*}$" if r.significant == "yes" else ""
        cells = [r.comparison.replace("&", "\\&"), r.region, r.metric,
                 fd(r.delta, r.metric) + star, fp(r.p_wilcoxon_raw), fp(r.p_wilcoxon_holm)]
        if has_t:
            cells.append(fp(r.p_ttest))
        cells += [str(int(r.n_nonzero)), r.direction]
        L.append(" & ".join(cells) + " \\\\\n")
    L.append("\\end{longtable}\n\n\\clearpage\n")

L.append(r"""
\noindent $^{*}$ significant at $p<0.05$ after Holm correction within the stated family.

\end{document}
""")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("".join(L))
print("wrote", OUT, f"({len(df)} comparisons)")
