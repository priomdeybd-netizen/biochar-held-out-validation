#!/usr/bin/env python
"""Build the manuscript (LaTeX) from adsorption_results/ produced by the Colab notebook.
Usage: python -I make_paper_ads.py --res adsorption_results --out paper_out
Every number is read from the result tables; qualitative claims are asserted against them."""
import argparse, json, os, shutil
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--res", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); R, OUT = a.res, a.out
os.makedirs(OUT, exist_ok=True)

S = json.load(open(f"{R}/summary.json")); D = S["diagnostics"]
M = pd.read_csv(f"{R}/main_results_all_models.csv"); GAP = pd.read_csv(f"{R}/random_vs_heldout_gap.csv")
BEST = pd.read_csv(f"{R}/heldout_mae_by_model.csv"); CONF = pd.read_csv(f"{R}/conformal_coverage.csv")
CB = pd.read_csv(f"{R}/conformal_coverage_by_biochar.csv"); BYG = pd.read_csv(f"{R}/skill_by_heldout_group.csv")
IMP = pd.read_csv(f"{R}/permutation_importance.csv"); ABL = pd.read_csv(f"{R}/ablation_input_groups.csv")
LC = pd.read_csv(f"{R}/learning_curve_n_biochars.csv"); DER = pd.read_csv(f"{R}/capacity_direct_vs_derived.csv")

RM = pd.read_csv(f"{R}/replication_main.csv").set_index(["target", "protocol"]); RA = pd.read_csv(f"{R}/replication_ablation.csv").set_index(["target", "protocol", "inputs"])
D2 = S["replication_diagnostics"]
def f(x, d=2): return f"{x:.{d}f}"
def ci(lo, hi, d=2): return f"[{lo:.{d}f}, {hi:.{d}f}]"
TG = ["Capacity", "removal_pct"]; PR = ["random", "replicate", "pair", "adsorbent", "pollutant"]
TL = {"Capacity": "capacity", "removal_pct": "removal"}
PL = {"random": "random 5-fold", "replicate": "no exact input twins", "pair": "held-out pair", "adsorbent": "held-out biochar", "pollutant": "held-out pollutant"}
rf = M[M.model == "RandomForest"].set_index(["target", "protocol"])
sk = lambda t, p, c="skill_vs_mean": rf.loc[(t, p), c]
ab = ABL.set_index(["target", "protocol", "inputs"]); AL, CO, BI = "all inputs", "experimental conditions only", "biochar properties only"
lc = LC.set_index(["target", "k_train_biochars"]); der = DER.set_index("protocol")
gp = GAP.set_index(["target", "protocol"])

# ----------------------------------------------------------------------------------- assertions
assert S["n_rows"] == 3757 and S["adsorbents"] == 15 and S["pollutants"] == 14 and S["pairs"] == 39
assert S["negative_capacity_rows"] == 27 and S["removal_excluded_rows"] == 27
assert (GAP.diff_lo > 0).all(), "all held-out protocols (incl. replicate-free) must have a clearly higher error than random folds"
assert all(0.85 <= sk(t, "random") <= 0.95 for t in TG)
assert all(abs(sk(t, "random") - sk(t, "replicate")) <= 0.06 for t in TG), "removing exact twins must change skill by <= 0.06"
assert all(sk(t, "adsorbent") < sk(t, "pair") < sk(t, "random") for t in TG)
assert all(sk(t, "adsorbent") < sk(t, "pollutant") for t in TG)
assert sk("Capacity", "adsorbent", "skill_lo") <= 0 and sk("removal_pct", "adsorbent", "skill_lo") > 0
assert D["rows_with_identical_input_elsewhere"] > 0.9 and D["random_cv_test_rows_with_exact_input_twin_in_training"] > 0.85
assert D["capacity_equals_dC_over_dose_share"] > 0.97
bm = BEST.set_index(["target", "protocol"])
assert bm.loc[("Capacity", "adsorbent"), "best_learned"] == "HistGB" and bm.loc[("removal_pct", "adsorbent"), "best_learned"] == "Ridge"
assert bm.loc[("removal_pct", "adsorbent"), "Ridge"] < bm.loc[("removal_pct", "adsorbent"), "RandomForest"]
assert all(bm.loc[("Capacity", p), "RandomForest"] < bm.loc[("Capacity", p), "Ridge"] for p in ("replicate", "pair", "pollutant"))
c80 = CONF[np.isclose(CONF.nominal, .8)].set_index("target"); c90 = CONF[np.isclose(CONF.nominal, .9)].set_index("target")
assert (abs(CONF.coverage_pooled - CONF.nominal) <= 0.07).all() and (CONF.coverage_biochar_min < 0.2).all()
wmin, wmax = CONF.width_over_target_sd.min(), CONF.width_over_target_sd.max()
bg = BYG.set_index(["target", "held_out", "group"])
neg_cap = BYG[(BYG.target == "Capacity") & (BYG.held_out == "adsorbent") & (BYG.skill < 0)].group.tolist()
pos_cap_n = int(((BYG.target == "Capacity") & (BYG.held_out == "adsorbent") & (BYG.skill > 0)).sum()); assert pos_cap_n == 10 and len(neg_cap) == 5
pos_rm_n = int(((BYG.target == "removal_pct") & (BYG.held_out == "adsorbent") & (BYG.skill > 0)).sum()); assert pos_rm_n == 9
cap80 = CB[(CB.target == "Capacity") & np.isclose(CB.nominal, .8)].set_index("biochar").coverage
assert cap80.min() < 0.05 and int((cap80 < 0.5).sum()) == 2
top = {t: IMP[IMP.target == t].sort_values("importance", ascending=False).reset_index(drop=True) for t in TG}
assert top["Capacity"].feature[0] == "Initial concentration" and top["Capacity"].importance[0] > 0.1
# ablation
for t in TG:
    assert ab.loc[(t, "adsorbent", CO), "skill"] < 0, "conditions-only must be worse than the mean for held-out biochars"
    assert ab.loc[(t, "adsorbent", BI), "skill"] > 0.2
    assert ab.loc[(t, "adsorbent", BI), "skill"] >= ab.loc[(t, "adsorbent", AL), "skill"] - 0.05
    assert ab.loc[(t, "random", BI), "skill"] > ab.loc[(t, "random", CO), "skill"]
    assert ab.loc[(t, "random", AL), "skill"] > ab.loc[(t, "random", BI), "skill"] + 0.2
# learning curve
for t in TG:
    assert lc.loc[(t, 14), "skill"] > lc.loc[(t, 2), "skill"] + 0.2 and lc.loc[(t, 2), "skill_lo"] <= 0 < lc.loc[(t, 14), "skill_hi"]
    assert lc.loc[(t, 14), "skill"] >= lc.loc[(t, 8), "skill"] - 0.02
# derived capacity
assert der.loc["adsorbent", "skill_derived"] > der.loc["adsorbent", "skill_direct"] + 0.1
DER_CLEAR = bool(der.loc["adsorbent", "diff_lo"] > 0)   # is the improvement for held-out biochars statistically clear?
assert abs(der.loc["random", "skill_derived"] - der.loc["random", "skill_direct"]) < 0.03

# replication on the second data set (claims in the text are asserted here)
T2G = ["qe", "removal_pct"]
assert D2["rows"] == 1518 and D2["adsorbents"] == 16 and D2["metals"] == 7 and D2["pairs"] == 25 and D2["negative_qe_rows"] == 0 and D2["removal_outside_0_100_rows"] == 1
assert all(0.8 <= RM.loc[(t, "random"), "skill"] <= 0.95 for t in T2G)
assert all((RM.loc[(t, p), "diff_lo"] > 0) for t in T2G for p in ("replicate", "pair", "adsorbent", "pollutant")), "gap must replicate in direction"
assert RM.loc[("qe", "adsorbent"), "skill_lo"] > 0.5, "capacity transfer to new biochars must be clearly better in data set 2"
assert RM.loc[("removal_pct", "adsorbent"), "skill_lo"] <= 0 and RM.loc[("removal_pct", "adsorbent"), "skill"] < 0.3
assert RA.loc[("qe", "adsorbent", CO), "skill"] > RA.loc[("qe", "adsorbent", AL), "skill"] and RA.loc[("qe", "adsorbent", BI), "skill_lo"] <= 0 and RA.loc[("qe", "adsorbent", BI), "skill"] < 0.3, "ablation must NOT replicate"
assert all(RA.loc[("removal_pct", "adsorbent", g), "skill_lo"] <= 0 for g in (AL, CO, BI))
assert RA.loc[("qe", "random", BI), "skill"] < RA.loc[("qe", "random", CO), "skill"]
# ----------------------------------------------------------------------------------- tables
t1 = []
for t in TG:
    for p in PR:
        r = rf.loc[(t, p)]
        t1.append(f"{TL[t]} & {PL[p]} & {int(r.n_clusters)} & {f(r.MAE,1)} {ci(r.MAE_lo,r.MAE_hi,1)} & {f(r.skill_vs_mean)} {ci(r.skill_lo,r.skill_hi)} & {f(r.R2)} \\\\")
TAB1 = "\n".join(t1)
t2 = []
for _, r in GAP.iterrows():
    t2.append(f"{TL[r.target]} & {PL[r.protocol]} & {f(r.MAE_random,1)} & {f(r.MAE_protocol,1)} & {f(r['diff'],1)} {ci(r.diff_lo,r.diff_hi,1)} \\\\")
TAB2 = "\n".join(t2)
t3 = []
for t in TG:
    for p in ["replicate", "pair", "adsorbent", "pollutant"]:
        r = bm.loc[(t, p)]; learned = ["Ridge", "RandomForest", "ExtraTrees", "HistGB"]; cells = []
        for m_ in ["Mean"] + learned:
            v = f(r[m_], 1); cells.append(r"\textbf{" + v + "}" if (m_ in learned and m_ == r.best_learned) else v)
        t3.append(f"{TL[t]} & {PL[p]} & " + " & ".join(cells) + r" \\")
TAB3 = "\n".join(t3)
t4 = []
for t in TG:
    t4.append(f"{TL[t]} & {f(c80.loc[t,'coverage_pooled'])} & {f(c90.loc[t,'coverage_pooled'])} & {f(c80.loc[t,'width_over_target_sd'],1)} & {f(c90.loc[t,'width_over_target_sd'],1)} & {f(c80.loc[t,'coverage_biochar_min'])} & {int(round(100*c80.loc[t,'share_biochars_below_nominal']))}\\% \\\\")
TAB4 = "\n".join(t4)
order = BYG[(BYG.target == "Capacity") & (BYG.held_out == "adsorbent")].sort_values("skill", ascending=False).group.tolist()
t5 = []
for g in order:
    cs = bg.loc[("Capacity", "adsorbent", g), "skill"]; rs = bg.loc[("removal_pct", "adsorbent", g), "skill"]
    t5.append(r"\texttt{" + g.replace("_", r"\_") + f"}} & {int(bg.loc[('Capacity','adsorbent',g),'n'])} & {f(cs)} & {f(rs)} & {f(cap80[g])} \\\\")
TAB5 = "\n".join(t5)
FN = {"Initial concentration": "initial concentration", "Adsorption time": "contact time", "Ion concentration": "ion concentration", "Pollutant": "pollutant",
      "Solution pH": "solution pH", "Adsorption temperature": "temperature", "Adsorbent dosage": "adsorbent dosage"}
TAB6 = "\n".join(f"{TL[t]} & " + "; ".join(f"{FN.get(x, x)} ({f(v,3)})" for x, v in zip(top[t].feature[:4], top[t].importance[:4])) + r" \\" for t in TG)
tA = []
for t in TG:
    for p in ["random", "adsorbent"]:
        for nm, lab in [(AL, "all inputs"), (CO, "experimental conditions only"), (BI, "biochar properties only")]:
            r = ab.loc[(t, p, nm)]; tA.append(f"{TL[t]} & {PL[p]} & {lab} & {f(r.MAE,1)} & {f(r.skill)} {ci(r.skill_lo,r.skill_hi)} \\\\")
TABA = "\n".join(tA)
tL = []
for k in sorted(LC.k_train_biochars.unique()):
    c_, r_ = lc.loc[("Capacity", k)], lc.loc[("removal_pct", k)]
    tL.append(f"{int(k)} & {f(c_.skill)} {ci(c_.skill_lo,c_.skill_hi)} & {f(r_.skill)} {ci(r_.skill_lo,r_.skill_hi)} \\\\")
TABL = "\n".join(tL)
tD = []
for p in PR:
    r = der.loc[p]
    tD.append(f"{PL[p]} & {f(r.MAE_direct,1)} & {f(r.MAE_derived_from_removal,1)} & {f(r.diff_direct_minus_derived,1)} {ci(r.diff_lo,r.diff_hi,1)} & {f(r.skill_direct)} & {f(r.skill_derived)} \\\\")
TABD = "\n".join(tD)
tR = []
for t, tn in (("qe", "capacity ($q_e$)"), ("removal_pct", "removal")):
    for p in PR:
        r = RM.loc[(t, p)]
        tR.append(f"{tn} & {PL[p]} & {int(r.n_clusters)} & {f(r.MAE,1)} {ci(r.MAE_lo,r.MAE_hi,1)} & {f(r.skill)} {ci(r.skill_lo,r.skill_hi)} & {f(r['diff_vs_random'],1)} {ci(r.diff_lo,r.diff_hi,1)} \\\\")
TABR = "\n".join(tR)
tRA = []
for t, tn in (("qe", "capacity ($q_e$)"), ("removal_pct", "removal")):
    for p in ["random", "adsorbent"]:
        for nm in (AL, CO, BI):
            r = RA.loc[(t, p, nm)]; tRA.append(f"{tn} & {PL[p]} & {nm} & {f(r.MAE,1)} & {f(r.skill)} {ci(r.skill_lo,r.skill_hi)} \\\\")
TABRA = "\n".join(tRA)
neg_txt = ", ".join(r"\texttt{" + g.replace("_", r"\_") + "}" for g in neg_cap)

# ----------------------------------------------------------------------------------- text
sA = lambda t, p, c="skill": ab.loc[(t, p, c if c in (AL, CO, BI) else AL), "skill"]
DER_ABS = "an improvement that was statistically clear" if DER_CLEAR else "a difference whose 95\\% interval still included zero"
ABSTRACT = rf"""Machine-learning models for pollutant adsorption on biochar are usually evaluated with random train/test splits. Using an open compilation of {S['n_rows']:,} batch-adsorption measurements of {S['pollutants']} emerging contaminants on {S['adsorbents']} biochars, we asked how much of the accuracy survives when whole biochars, pollutants or biochar--pollutant pairs are withheld, what the models actually use, and how many biochars are needed. For a random forest, random five-fold cross-validation gave a skill (reduction in mean absolute error relative to predicting the training mean) of {f(sk('Capacity','random'))} for capacity and {f(sk('removal_pct','random'))} for removal. Although {int(round(100*D['rows_with_identical_input_elsewhere']))}\% of the measurements have an exact input twin, removing the twins changed the skill only to {f(sk('Capacity','replicate'))} and {f(sk('removal_pct','replicate'))}; holding out whole biochars reduced it to {f(sk('Capacity','adsorbent'))} {ci(sk('Capacity','adsorbent','skill_lo'),sk('Capacity','adsorbent','skill_hi'))} and {f(sk('removal_pct','adsorbent'))} {ci(sk('removal_pct','adsorbent','skill_lo'),sk('removal_pct','adsorbent','skill_hi'))} (95\% bootstrap intervals over biochars), with a clear increase in error in every comparison. The loss therefore comes from the shared origin of measurements (same biochar, same experimental series), not from exact repeats. In this data set, models that used only biochar properties reached a held-out-biochar skill of {f(sA('Capacity','adsorbent',BI))} and {f(sA('removal_pct','adsorbent',BI))}, not lower than models using all inputs ({f(sA('Capacity','adsorbent',AL))} and {f(sA('removal_pct','adsorbent',AL))}), whereas models using only experimental conditions were worse than the training mean ({f(sA('Capacity','adsorbent',CO))} and {f(sA('removal_pct','adsorbent',CO))}). The skill rose from about zero with two training biochars to {f(lc.loc[('Capacity',14),'skill'])} and {f(lc.loc[('removal_pct',14),'skill'])} with fourteen, without a plateau. Predicting removal and converting it to capacity raised the held-out-biochar skill for capacity from {f(der.loc['adsorbent','skill_direct'])} to {f(der.loc['adsorbent','skill_derived'])}, {DER_ABS}. A replication of the protocol comparison and the ablation on a second public data set ({D2['rows']:,} measurements of {D2['metals']} metal ions on {D2['adsorbents']} biochars) reproduced the clear increase in error for held-out biochars, but with a different size (held-out-biochar skill {f(RM.loc[('qe','adsorbent'),'skill'])} for capacity and {f(RM.loc[('removal_pct','adsorbent'),'skill'])} for removal) and not the ablation pattern: there, experimental conditions were more informative than biochar properties. Conformal intervals calibrated on other biochars had near-nominal pooled coverage but were {f(wmin,1)} to {f(wmax,1)} standard deviations wide and missed some biochars almost completely. Each data set has only about {S['adsorbents']} biochars, so the results are case studies, not general estimates."""

TEMPLATE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs,graphicx,amsmath,hyperref,float,caption}
\hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue}
\linespread{1.12}
\graphicspath{{figures/}{./}}
\title{How Well Can Machine Learning Predict Adsorption on a Biochar It Has Never Seen? Evidence from Two Public Data Sets}
\author{Priom Dey\textsuperscript{*}, Avro Talukder, Yuvraj Roy\\[2pt] \small Department of Chemistry, Shahjalal University of Science and Technology, Sylhet, Bangladesh \\[2pt] \small *Corresponding author: \texttt{2023131044@student.sust.edu} \\ \small ORCID: Priom Dey \href{https://orcid.org/0009-0007-5722-8240}{0009-0007-5722-8240}; Avro Talukder \href{https://orcid.org/0009-0007-6996-7673}{0009-0007-6996-7673} \\ \small Other e-mail addresses: \texttt{2023131055@student.sust.edu} (A.T.);\\ \small \texttt{2023131056@student.sust.edu} (Y.R.)}
\date{}
\begin{document}\sloppy\maketitle
\begin{abstract}
@@ABSTRACT@@
\end{abstract}
\noindent\textbf{Keywords:} biochar; adsorption; machine learning; validation; data leakage; learning curve; conformal prediction

\section{Introduction}
Biochar is widely studied as a low-cost adsorbent for pharmaceuticals and other contaminants in water, and many machine-learning models have been proposed to predict adsorption capacity or removal from biochar properties and experimental conditions \cite{Jaffari,LiGuptaYou}. These studies typically collect measurements from many experiments, split the rows at random into training and test sets, and report a high test accuracy. The rows of such a data set are not independent: one biochar is tested at many concentrations, doses and contact times, often with several pollutants. A random split puts closely related rows into both the training and the test set, which is known to give optimistic estimates when the data have a group structure \cite{Roberts2017,Bates2024}, and leakage of this kind is a recognised cause of unreliable results in machine-learning-based science \cite{Kapoor2023}. What a user needs to know is different: how well does a model predict a biochar, a pollutant or a combination that was not in its training data \cite{Meredig2018}?

Concern about data leakage and data handling in machine-learning studies of biochar has already been raised in the literature \cite{JangLeakage,ShortcomingsHM}. We do not claim to be the first to point at this problem. We use one public data set as a quantified, fully reproducible case study and ask four questions that go beyond a comparison of random and grouped validation: (i) \emph{where does the optimism of random validation come from}, from exact repeats of the same inputs or from the shared origin of rows; (ii) \emph{what do the models use}, biochar properties or experimental conditions; (iii) \emph{how many biochars are needed} before a model transfers to a new one; and (iv) \emph{can the target be chosen so as to transfer better}, by predicting the bounded removal percentage and converting it to capacity. We repeat the protocol comparison and the ablation on a second public data set to see which findings are specific to one data set, and test whether conformal prediction intervals calibrated on other biochars are reliable for a new biochar. The data set is that of Jaffari et al.\ \cite{Jaffari,JaffariData}; the code released with it evaluates its models with a random 70/30 train/test split (\texttt{train\_test\_split}), and we use it as an example of common practice, not as a criticism of that work.

\section{Data and Methods}
\subsection{Data}
We used the public file \texttt{Raw\_data.csv} of Jaffari et al.\ \cite{JaffariData} (repository commit \texttt{25d8188}), which contains @@NROWS@@ batch-adsorption measurements of @@NPOL@@ emerging contaminants on @@NADS@@ biochars or activated-carbon samples, with pyrolysis conditions, elemental composition, ash, surface area and pore properties of the adsorbent, and the experimental conditions (contact time, initial concentration, pH, shaking speed, volume, dose, temperature, ion concentration, humic acid, wastewater type and adsorption type). The notebook downloads the original file; it is not redistributed because the repository carries no licence file. The file does not state which publication or experimental series each row comes from, so we used the adsorbent name and the pollutant as grouping variables. The data contain @@NPAIRS@@ different biochar--pollutant pairs. The file gives no units. The capacity equals $(C_i-C_f)/m$, where $C_i$ and $C_f$ are the initial and final concentrations and $m$ the dose column, to within 0.1\% for @@IDENT@@ of the rows with $C_i\neq C_f$; with concentrations in mg\,L$^{-1}$ (as stated in the repository README) and dose in g\,L$^{-1}$ this is mg\,g$^{-1}$, which we assume for capacity.

\subsection{Second data set (replication)}
To check which findings are particular to one data set, the protocol comparison and the ablation were repeated, with the same code and settings, on the public file \texttt{HMI\_data.xlsx} of the same group \cite{HMIData} (repository commit \texttt{086671f}), which contains @@R_ROWS@@ measurements of @@R_METALS@@ heavy-metal ions on @@R_ADS@@ biochars (@@R_PAIRS@@ biochar--ion pairs). The inputs were 15 biochar properties (pyrolysis conditions, composition, ash, surface area and pores), the feedstock, and 13 experimental or ion descriptors (ionic and hydrated radius, first ionisation energy, contact time, initial concentration, pH, stirring speed, volume, mass, dose, temperature, ion concentration, dissolved organic matter) plus the ion and anion type. The targets were the reported adsorption capacity $q_e$ (units of the file, not verified here, because $q_e\,m/(C_i-C_f)$ was not equal to one for all rows) and the removal $100(C_i-C_f)/C_i$ (one row outside 0--100\% was excluded). The final concentration was not an input. @@R_TWIN@@ of the rows share their complete input vector with another row. For the held-out-ion protocol only @@R_METALS@@ ions exist, so its intervals are very wide. The conformal intervals, the learning curve and the derived-capacity analysis were not repeated.

\subsection{Targets and inputs}
The targets were the adsorption capacity and the removal percentage $100(C_i-C_f)/C_i$. Because of the relation above, capacity $=$ removal $\times\, C_i/(100\,m)$. The final concentration was \emph{not} used as an input because both targets are calculated from it. Twenty-seven measurements with negative capacity (physically impossible) were removed from the capacity target, and 27 rows with removal outside 0--100\% from the removal target. The inputs were 23 numerical variables and three categorical variables (pollutant, wastewater type, adsorption type). We call 14 of the numerical variables \emph{biochar properties} (pyrolysis temperature and time, C, H, O, N, (O+N)/C, ash, H/C, O/C, N/C, surface area, pore volume, average pore size) and the other nine numerical variables together with the three categorical ones \emph{experimental conditions} (contact time, initial concentration, pH, shaking speed, volume, dose, temperature, ion concentration, humic acid, wastewater type, adsorption type, pollutant). The adsorbent name was used only for grouping. In the held-out-pollutant protocol the pollutant was removed from the inputs because the model cannot know an unseen compound. All preprocessing was fitted inside each training fold.

\subsection{Models}
We compared ridge regression, random forests (200 trees, half of the features per split, minimum leaf size 2) \cite{Breiman2001}, extra trees (same settings) \cite{Geurts2006} and histogram gradient boosting (150 iterations, learning rate 0.05, depth 3), all from \texttt{scikit-learn} \cite{Pedregosa2011}. The baseline predicts the training mean. No hyperparameters were tuned. The random forest was used for all additional analyses.

\subsection{Validation protocols and statistics}
Five protocols were used. \emph{Random}: five-fold cross-validation of rows, repeated three times. \emph{No exact input twins}: five-fold cross-validation in which all rows with an identical input vector are in the same fold, so no test row has an exact twin in the training data. \emph{Held-out pair}: all measurements of one biochar--pollutant pair are withheld. \emph{Held-out biochar}: all measurements of one biochar are withheld (15 folds). \emph{Held-out pollutant}: all measurements of one pollutant are withheld (14 folds). We report the mean absolute error (MAE), $R^2$ and the \emph{skill} $=1-\mathrm{MAE}/\mathrm{MAE}_{\text{mean}}$, where $\mathrm{MAE}_{\text{mean}}$ is the error of the training mean under the same protocol; skill 0 means no better than the training mean. Intervals are 95\% bootstrap intervals (2000 resamples) in which the resampled unit is the held-out unit (pairs, biochars or pollutants; biochars for the random and no-twin protocols), and the same resamples are used for paired differences between protocols.

\subsection{Input repeats, input-group ablation, learning curve and derived capacity}
We counted the rows whose complete input vector occurs in at least one other row, and the share of test rows with an exact twin in the training rows of random folds. For the \emph{ablation} we trained the random forest with (a) all inputs, (b) experimental conditions only, or (c) biochar properties only, under random folds and under held-out biochars. For the \emph{learning curve} we held out each biochar in turn and trained on a random subset of $k=2,4,6,8,10$ or 14 of the other biochars (five random subsets for each $k$ and held-out biochar), and report the skill against the training mean of the same subset. For the \emph{derived capacity} we multiplied the out-of-fold prediction of removal by $C_i/(100\,m)$ and compared the error against the directly predicted capacity on the same rows, using the bootstrap over biochars for the difference.

\subsection{Prediction intervals and importance}
For the random forest we built split-conformal intervals \cite{Lei2018,Angelopoulos2023} for the held-out-biochar protocol. For each held-out biochar, absolute residuals came from an inner grouped five-fold cross-validation over the \emph{training} biochars only, and the half-width was the finite-sample-corrected quantile for the nominal level (80\% and 90\%). We report pooled coverage, the lowest coverage of any biochar, the share of biochars below nominal, and the width relative to the standard deviation of the target. Permutation importance was computed on held-out biochars (three shuffles per input) as the increase in squared error; it describes the fitted models, not causal effects.

\section{Results}
\subsection{Exact input repeats and the validation ladder}
@@NREP@@ of the @@NUSED@@ measurements share their complete input vector with at least one other measurement (median group size @@MEDREP@@, maximum @@MAXREP@@; @@NDISTINCT@@ distinct input vectors in @@NROWS@@ rows). In random five-fold cross-validation, @@NTWIN@@ of the test rows had an exact twin in the training rows, and the random forest reached a skill of @@SKR1@@ for capacity and @@SKR2@@ for removal (Table~\ref{tab:main}, Fig.~\ref{fig:skill}). The mean absolute difference in capacity between rows with identical inputs was @@MAD@@~mg\,g$^{-1}$, of the same order as the random-split MAE (@@MAEREL@@~mg\,g$^{-1}$), so random folds come close to the replicate spread of the data.

Removing the exact twins did not remove the optimism. With no exact twin in training the skill was still @@SKT1@@ for capacity and @@SKT2@@ for removal, an increase in MAE of only @@TWD1@@ and @@TWD2@@ relative to random folds (Table~\ref{tab:gap}), although this increase was statistically distinguishable from zero. Holding out a pollutant (skill @@SKP1@@ and @@SKP2@@) or a pair (@@SKQ1@@ and @@SKQ2@@) was clearly more damaging, and holding out a biochar gave @@SKA1@@ for capacity (95\% interval @@CIA1@@, which includes zero) and @@SKA2@@ for removal (@@CIA2@@). Thus the loss of accuracy is not caused by exact repeats; it is caused by the shared origin of rows, since neighbouring conditions of the same biochar and series stay in the training data unless the whole biochar, pollutant or pair is withheld. Because the file contains no descriptors of the pollutants, the positive skill for unseen pollutants mainly reflects the experimental conditions, and should not be read as evidence that the model understands pollutant chemistry.

Per-biochar results are in Table~\ref{tab:bychar} and Fig.~\ref{fig:parity}. For capacity, the model was better than the training mean for @@POSC@@ of the @@NADS@@ held-out biochars and worse for @@NEGC@@ (@@NEGLIST@@); for removal it was better for @@POSR@@ of @@NADS@@. For held-out pollutants, capacity skill was positive for all 14 and removal skill for 13.

\begin{table}[H]\centering\footnotesize\caption{Random forest under five validation protocols. Clusters are the units resampled for the 95\% intervals. MAE in the units of the target (mg\,g$^{-1}$ for capacity, \% for removal); skill $=1-\mathrm{MAE}/\mathrm{MAE}_{\text{mean}}$.}\label{tab:main}
\begin{tabular}{llcccc}\toprule
Target & Protocol & Clusters & MAE [95\% CI] & Skill [95\% CI] & $R^2$\\\midrule
@@TAB1@@
\bottomrule\end{tabular}\end{table}

\begin{table}[H]\centering\footnotesize\caption{Increase in MAE of the random forest relative to random five-fold validation, with 95\% bootstrap intervals.}\label{tab:gap}
\begin{tabular}{llccc}\toprule
Target & Protocol & MAE random & MAE protocol & Increase [95\% CI]\\\midrule
@@TAB2@@
\bottomrule\end{tabular}\end{table}

\begin{figure}[H]\centering\includegraphics[width=.95\linewidth]{fig1_skill_by_protocol.png}\caption{Skill of the random forest under the five protocols (error bars: 95\% bootstrap interval).}\label{fig:skill}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=.95\linewidth]{fig2_parity_biochar_held_out.png}\caption{Predictions for held-out biochars, coloured by pollutant.}\label{fig:parity}\end{figure}

\begin{table}[H]\centering\footnotesize\caption{Random forest skill for each held-out biochar (names as in the data file), sorted by capacity skill, and conformal coverage at the 80\% level.}\label{tab:bychar}
\begin{tabular}{lcccc}\toprule
Biochar & $n$ & Capacity skill & Removal skill & Coverage (80\%)\\\midrule
@@TAB5@@
\bottomrule\end{tabular}\end{table}

\subsection{Comparison of models}
Table~\ref{tab:models} gives the held-out MAE of all models. No model was best everywhere. For unseen biochars, gradient boosting had the lowest capacity error (@@HGBCAP@@ against @@RFCAP@@ for the random forest and @@MEANCAP@@ for the training mean), whereas ridge regression had the lowest removal error (@@RIDREM@@ against @@RFREM@@ for the random forest), better than all tree ensembles. For no-twin, pair and pollutant holdouts the tree ensembles were clearly better than ridge regression for capacity. The ranking of models therefore depends on what is held out, and conclusions about the best model drawn from random splits may not carry over. As no model was tuned, these rankings should not be over-interpreted.

\begin{table}[H]\centering\footnotesize\caption{Held-out MAE of all models; bold = lowest among the four learned models.}\label{tab:models}
\begin{tabular}{llccccc}\toprule
Target & Protocol & Training mean & Ridge & Random forest & Extra trees & Grad.\ boosting\\\midrule
@@TAB3@@
\bottomrule\end{tabular}\end{table}

\subsection{What carries the signal: input-group ablation}
Table~\ref{tab:abl} and Fig.~\ref{fig:abl} show the ablation. When whole biochars were held out, a model with only biochar properties reached a skill of @@ABBIC@@ for capacity and @@ABBIR@@ for removal, not lower than the models using all inputs (@@ABALC@@ and @@ABALR@@; the intervals overlap, Table~\ref{tab:abl}), whereas a model with only experimental conditions was worse than the training mean (@@ABCOC@@ and @@ABCOR@@). Under random folds the picture was different: biochar properties alone gave @@ABRBIC@@ and @@ABRBIR@@ and conditions alone only @@ABRCOC@@ and @@ABRCOR@@, but the combination reached @@SKR1@@ and @@SKR2@@. Within a biochar, the conditions explain the variation (concentration, time), but they carry no information about which biochar is being predicted; across biochars, the properties give the overall level of adsorption, and adding the conditions did not improve transfer to a new biochar. This agrees with the permutation importance of the full model on held-out biochars (Table~\ref{tab:imp}, Fig.~\ref{fig:imp}), in which the initial concentration was by far the most important input for capacity (@@IMPCAP@@) and contact time for removal. These importances show what the fitted model relies on, not what helps it transfer. In addition, capacity scales with $C_i$ by definition, which explains the importance of $C_i$ for that target.

\begin{table}[H]\centering\footnotesize\caption{Skill of the random forest with different input groups.}\label{tab:abl}
\begin{tabular}{lllcc}\toprule
Target & Protocol & Inputs & MAE & Skill [95\% CI]\\\midrule
@@TABA@@
\bottomrule\end{tabular}\end{table}
\begin{figure}[H]\centering\includegraphics[width=.95\linewidth]{fig6_input_group_ablation.png}\caption{Skill with all inputs, with experimental conditions only and with biochar properties only (random folds and held-out biochars).}\label{fig:abl}\end{figure}

\begin{table}[H]\centering\footnotesize\caption{Four largest permutation importances on held-out biochars (increase in squared error relative to unshuffled; random forest).}\label{tab:imp}
\begin{tabular}{ll}\toprule
Target & Largest importances\\\midrule
@@TAB6@@
\bottomrule\end{tabular}\end{table}
\begin{figure}[H]\centering\includegraphics[width=.95\linewidth]{fig4_permutation_importance.png}\caption{Permutation importance on held-out biochars (random forest).}\label{fig:imp}\end{figure}

\subsection{How many biochars are needed}
Fig.~\ref{fig:lc} and Table~\ref{tab:lc} show the learning curve. With two training biochars the skill on the held-out biochar was @@LC2C@@ for capacity and @@LC2R@@ for removal, with intervals that include zero. It rose to @@LC14C@@ and @@LC14R@@ with 14 training biochars, but not monotonically and with wide intervals, and there was no plateau at 14: more biochars were still improving the prediction. The data set with 15 biochars is therefore too small to say how good a model could become, and the held-out skill reported here is likely to be a lower bound for a larger collection of biochars.

\begin{table}[H]\centering\footnotesize\caption{Learning curve: skill on the held-out biochar against the number of training biochars (random forest; five random subsets per biochar and $k$).}\label{tab:lc}
\begin{tabular}{lcc}\toprule
Training biochars & Capacity skill [95\% CI] & Removal skill [95\% CI]\\\midrule
@@TABL@@
\bottomrule\end{tabular}\end{table}
\begin{figure}[H]\centering\includegraphics[width=.9\linewidth]{fig5_learning_curve.png}\caption{Skill on the held-out biochar against the number of training biochars.}\label{fig:lc}\end{figure}

\subsection{Choosing the target: removal converted to capacity}
Table~\ref{tab:der} compares the directly predicted capacity with the capacity obtained from the predicted removal. Under random folds the two were practically equal (skill @@DRD@@ and @@DRV@@). For held-out biochars the derived capacity had the lower error: skill @@DAV@@ against @@DAD@@ for the direct prediction. The difference in MAE (@@DAMAE@@~mg\,g$^{-1}$) had a 95\% interval of @@DACI@@, @@DAJUDGE@@ A plausible reason is that removal is bounded between 0 and 100\% and the dependence on concentration and dose is then handled by an exact formula instead of being learned from 15 biochars; we did not test this explanation.

\begin{table}[H]\centering\footnotesize\caption{Capacity predicted directly and derived from the predicted removal (random forest; common rows). Difference = direct minus derived MAE, with 95\% bootstrap interval.}\label{tab:der}
\begin{tabular}{lccccc}\toprule
Protocol & MAE direct & MAE derived & Difference [95\% CI] & Skill direct & Skill derived\\\midrule
@@TABD@@
\bottomrule\end{tabular}\end{table}

\subsection{Prediction intervals for new biochars}
Table~\ref{tab:conf} and Fig.~\ref{fig:conf} show the conformal intervals for held-out biochars. The pooled coverage was close to nominal (@@COVLO@@ to @@COVHI@@ for nominal levels 0.8 and 0.9), but the intervals were wide: @@WMIN@@ to @@WMAX@@ standard deviations of the target. Coverage varied strongly between biochars: for capacity at the 80\% level, two biochars had coverage below 50\% (the lowest was @@COVMIN@@), and for @@SHLO@@\% to @@SHHI@@\% of the biochars the coverage was below nominal. A pooled coverage figure thus hides the fact that a new biochar may lie almost entirely outside the interval.

\begin{table}[H]\centering\footnotesize\setlength{\tabcolsep}{4pt}\caption{Conformal intervals for held-out biochars (random forest). Width is the full interval width divided by the standard deviation of the target.}\label{tab:conf}
\begin{tabular}{lcccccc}\toprule
Target & Coverage (80\%) & Coverage (90\%) & Width/SD (80\%) & Width/SD (90\%) & Lowest biochar coverage (80\%) & Biochars below nominal (80\%)\\\midrule
@@TAB4@@
\bottomrule\end{tabular}\end{table}
\begin{figure}[H]\centering\includegraphics[width=.6\linewidth]{fig3_conformal_coverage_by_biochar.png}\caption{Coverage of the conformal intervals for each held-out biochar (capacity).}\label{fig:conf}\end{figure}

\subsection{Replication on a second data set}
Table~\ref{tab:rep} gives the protocol comparison for the second data set. As in the first, every held-out protocol had a higher error than random folds, with 95\% intervals of the increase that exclude zero (random skill @@R_SK1@@ for capacity and @@R_SK2@@ for removal). The size of the effect differed. For capacity, the skill for unseen biochars was @@R_AD1@@ @@R_AD1CI@@, much higher than in the first data set, and removal was poorly transferable: @@R_AD2@@ @@R_AD2CI@@, not distinguishable from zero; held-out pairs gave @@R_PQ1@@ and @@R_PQ2@@, and held-out ions @@R_PO1@@ and @@R_PO2@@ (only @@R_METALS@@ ions). Removing exact input twins reduced the skill only to @@R_TW1@@ and @@R_TW2@@.

The ablation did \emph{not} replicate (Table~\ref{tab:repabl}). For capacity and held-out biochars, the model with only experimental and ion descriptors had a skill of @@R_COC@@, higher than the model with all inputs (@@R_ALC@@), whereas biochar properties alone gave @@R_BIC@@ @@R_BICCI@@, an interval that includes zero. For removal, none of the three input groups transferred clearly (all intervals include zero). A plausible but untested reason is that in this data set the ion descriptors (radius, ionisation energy) separate the large differences in capacity between ions, which the first data set could not express (it had no pollutant descriptors); the held-out-biochar protocol then mainly tests how well these descriptors and the conditions are used. The finding that biochar properties carry the transfer to new biochars is therefore specific to the first data set and should not be generalised.

\begin{table}[H]\centering\footnotesize\caption{Replication on the second data set: random forest under five validation protocols (MAE in the units of the file for $q_e$ and in \% for removal; skill relative to the training mean; the last column is the MAE increase relative to random folds).}\label{tab:rep}
\begin{tabular}{llcccc}\toprule
Target & Protocol & Clusters & MAE [95\% CI] & Skill [95\% CI] & Increase vs random [95\% CI]\\\midrule
@@TABR@@
\bottomrule\end{tabular}\end{table}

\begin{table}[H]\centering\footnotesize\caption{Replication on the second data set: skill of the random forest with different input groups.}\label{tab:repabl}
\begin{tabular}{lllcc}\toprule
Target & Protocol & Inputs & MAE & Skill [95\% CI]\\\midrule
@@TABRA@@
\bottomrule\end{tabular}\end{table}

\section{Discussion}
Random cross-validation gave a skill of about 0.9 for both targets, while holding out whole biochars gave @@SKA1@@ for capacity and @@SKA2@@ for removal. The source of this gap is informative. About nine in ten rows have an exact input twin, yet removing the twins left the skill almost unchanged, so the optimism does not come from literal duplicates; it comes from the fact that neighbouring measurements of the same biochar and series remain in the training data. In practice this means that deduplicating a data set is not enough to make a random split trustworthy: the split must keep the biochar, the series or the publication together.

In the first data set the ablation shows that, for a new biochar, what carries over is information about the biochar itself; this did not replicate in the second data set, where the experimental and ion descriptors were more informative than the biochar properties (Table~\ref{tab:repabl}), so the result depends on the data set and on which descriptors exist. Models using only biochar properties transferred at least as well as models using all inputs, and models using only experimental conditions failed, but within a biochar the conditions are essential. We caution that with 15 biochars, biochar properties may act mainly as a fingerprint that places a biochar on a scale of overall adsorption, not as evidence that the model has learned general structure--property relations; the learning curve, which was still rising at 14 biochars, supports the view that the data are too few for the latter. The practical conclusions are (i) to validate by holding out whole adsorbents, with intervals that account for the small number of held-out units; (ii) to consider predicting a bounded quantity such as removal and converting it by the exact relation, which lowered the error for new biochars in our data (@@DERNOTE@@); (iii) not to assume that a flexible model is better, since ridge regression was best for removal on unseen biochars; and (iv) to share data with adsorbent and source identifiers and to add more biochars, which the learning curve indicates is the most effective way to improve transfer.

\paragraph{Limitations.}
(i) The data are one compilation by one group; the file does not identify the source experiments, so held-out biochars are a proxy for held-out studies, and the true number of independent experimental series is unknown. (ii) There are only @@NADS@@ biochars, so all intervals are wide and a different set of biochars could give different numbers, in particular in the learning curve and the ablation. (iii) The file has no pollutant descriptors, so results for unseen pollutants are less informative about chemistry than about experimental conditions. (iv) The units are inferred from the data relation and the README (Section~2.1). (v) No model was tuned, and we used one random forest for the additional analyses; absolute errors could be lower with tuning, but the conclusions concern the differences between protocols and input groups. (vi) The 27 negative capacities and 27 removal values outside 0--100\% suggest data errors that we only partly handled. (vii) Related work on leakage in biochar machine learning exists \cite{JangLeakage,ShortcomingsHM}; this paper adds a quantified single-data-set case, with an analysis of the origin of the optimism, and does not replace those analyses. (viii) The explanations of the ablation and of the benefit of the derived capacity are plausible but were not tested. (ix) The replication uses a second compilation from the same group, so it is not independent of the first in the way a data set from other laboratories would be; it did not repeat the conformal, learning-curve and derived-capacity analyses, and the units of the second target were not verified. (x) The conformal procedure assumes exchangeability of biochars, which is doubtful with 15 samples.

\section{Conclusions}
For the public biochar adsorption data set of Jaffari et al., random cross-validation of a random forest gave a skill of about 0.9, whereas holding out whole biochars reduced it to @@SKA1@@ (capacity) and @@SKA2@@ (removal). The optimism does not come from exact input repeats, which were common but whose removal changed the skill little, but from the shared origin of the rows. In the first data set biochar properties alone transferred to new biochars at least as well as all inputs and experimental conditions alone did not, but in a second data set this ablation pattern did not reproduce, although the larger error for held-out biochars did. The skill was still rising with the number of training biochars at 14. Predicting removal and converting it to capacity gave a lower error for new biochars (@@DERNOTE@@). Conformal intervals reached nominal coverage on average but were wide and failed for individual biochars. We recommend validation on held-out adsorbents, publication of source identifiers with the data, and the collection of more biochars.

\section*{Data and Code Availability}
The experimental data are the public data set of Jaffari et al.\ \cite{JaffariData}; please cite that work when using it. The analysis notebook, result tables, figures and the script that generates this manuscript are provided as Supplementary Material and will be deposited in a public repository (GitHub, archived on Zenodo) before publication. The results come from \texttt{scikit-learn} @@SKV@@ run in Google Colab.

\section*{Author Contributions}
\textbf{Priom Dey:} conceptualization, methodology, software, formal analysis, investigation, data curation, validation, visualization, writing -- original draft, writing -- review and editing. \textbf{Avro Talukder} and \textbf{Yuvraj Roy:} validation (checking of cited sources and of reported numbers against the supplied tables), writing -- review and editing.

\section*{Declaration of AI Assistance}
Generative AI (Anthropic's Claude) was used to help write the analysis code and to draft and edit the manuscript. The corresponding author ran the notebook; all authors reviewed the manuscript, and the authors take full responsibility for the content. AI tools are not listed as authors.

\section*{Funding}
This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.

\section*{Competing Interests}
The authors declare no competing interests.

\begin{thebibliography}{99}\small
\bibitem{Jaffari} Z.~H.~Jaffari, H.~Jeong, J.~Shin, J.~Kwak, C.~Son, Y.-G.~Lee, S.~Kim, K.~Chon, K.~H.~Cho, Machine-learning-based prediction and optimization of emerging contaminants' adsorption capacity on biochar materials, \emph{Chem.\ Eng.\ J.} \textbf{466}, 143073 (2023). doi:10.1016/j.cej.2023.143073
\bibitem{JaffariData} Z.~H.~Jaffari et al., Adsorption-capacity-prediction-for-ECs (data and code), GitHub repository, \url{https://github.com/ZeeshanHJ/Adsorption-capacity-prediction-for-ECs}, commit 25d8188 (accessed 2026).
\bibitem{HMIData} Z.~H.~Jaffari, A.~Abbas, C.-M.~Kim, J.~Kwak, C.~Son, Y.-G.~Lee, S.~Kim, K.~Chon, K.~H.~Cho, Transformer-based deep learning models for adsorption capacity prediction of heavy metal ions toward biochar-based adsorbents (data and code), GitHub repository \url{https://github.com/ZeeshanHJ/Transformer_models_prediction}, commit 086671f (accessed 2026). The associated article appears in \emph{J.\ Hazard.\ Mater.} (2023; PubMed ID 37866140). [\textbf{COMPLETE BEFORE SUBMISSION: volume, article number and DOI; cite the article itself}]
\bibitem{LiGuptaYou} Y.~Li, R.~Gupta, S.~You, Machine learning assisted prediction of biochar yield and composition via pyrolysis of biomass, \emph{Bioresour.\ Technol.} \textbf{359}, 127511 (2022). doi:10.1016/j.biortech.2022.127511
\bibitem{JangLeakage} Jang et al., Illusions of accuracy: widespread data leakage distorts machine learning results in biochar research, \emph{GCB Bioenergy} (2026). doi:10.1111/gcbb.70173. [\textbf{COMPLETE BEFORE SUBMISSION: first-author initial, remaining authors, volume and article number; read the paper and check the statements in the Introduction and Limitations}]
\bibitem{ShortcomingsHM} [\textbf{COMPLETE BEFORE SUBMISSION: authors}] Addressing data handling shortcomings in machine learning studies on biochar for heavy metal remediation, \emph{J.\ Hazard.\ Mater.} (2025), PubMed ID 40081054, ScienceDirect PII S0304-3894(25)08015 (the page of the article was not accessible to us). [\textbf{Read the paper; add authors, volume, article number and DOI, and check the statements in the Introduction and Limitations}]
\bibitem{Kapoor2023} S.~Kapoor, A.~Narayanan, Leakage and the reproducibility crisis in machine-learning-based science, \emph{Patterns}, 100804 (2023). doi:10.1016/j.patter.2023.100804
\bibitem{Roberts2017} D.~R.~Roberts, V.~Bahn, S.~Ciuti, M.~S.~Boyce, J.~Elith, G.~Guillera-Arroita, S.~Hauenstein, J.~J.~Lahoz-Monfort, B.~Schroeder, W.~Thuiller, D.~I.~Warton, B.~A.~Wintle, F.~Hartig, C.~F.~Dormann, Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure, \emph{Ecography} \textbf{40}, 913--929 (2017).
\bibitem{Bates2024} S.~Bates, T.~Hastie, R.~Tibshirani, Cross-validation: what does it estimate and how well does it do it?, \emph{J.\ Am.\ Stat.\ Assoc.} \textbf{119}, 1434--1445 (2024). doi:10.1080/01621459.2023.2197686
\bibitem{Meredig2018} B.~Meredig, E.~Antono, C.~Church, M.~Hutchinson, J.~Ling, S.~Paradiso, B.~Blaiszik, I.~Foster, B.~Gibbons, J.~Hattrick-Simpers, A.~Mehta, L.~Ward, Can machine learning identify the next high-temperature superconductor? Examining extrapolation performance for materials discovery, \emph{Mol.\ Syst.\ Des.\ Eng.} \textbf{3}, 819--825 (2018). doi:10.1039/C8ME00012C
\bibitem{Breiman2001} L.~Breiman, Random forests, \emph{Mach.\ Learn.} \textbf{45}, 5--32 (2001).
\bibitem{Geurts2006} P.~Geurts, D.~Ernst, L.~Wehenkel, Extremely randomized trees, \emph{Mach.\ Learn.} \textbf{63}, 3--42 (2006).
\bibitem{Pedregosa2011} F.~Pedregosa et al., Scikit-learn: machine learning in Python, \emph{J.\ Mach.\ Learn.\ Res.} \textbf{12}, 2825--2830 (2011).
\bibitem{Lei2018} J.~Lei, M.~G'Sell, A.~Rinaldo, R.~J.~Tibshirani, L.~Wasserman, Distribution-free predictive inference for regression, \emph{J.\ Am.\ Stat.\ Assoc.} \textbf{113}, 1094--1111 (2018). doi:10.1080/01621459.2017.1307116
\bibitem{Angelopoulos2023} A.~N.~Angelopoulos, S.~Bates, Conformal prediction: a gentle introduction, \emph{Found.\ Trends Mach.\ Learn.} \textbf{16}, 494--591 (2023). doi:10.1561/2200000101
\end{thebibliography}
\end{document}
"""
used = int(D["rows"])
ca = rf.loc[("Capacity", "adsorbent")]; ra = rf.loc[("removal_pct", "adsorbent")]
rep = {"ABSTRACT": ABSTRACT, "NROWS": f"{S['n_rows']:,}", "NPOL": str(S["pollutants"]), "NADS": str(S["adsorbents"]), "NPAIRS": str(S["pairs"]),
       "IDENT": f"{int(round(100*D['capacity_equals_dC_over_dose_share']))}\\%", "NREP": f"{int(round(100*D['rows_with_identical_input_elsewhere']))}\\%",
       "NUSED": f"{used:,}", "MEDREP": f"{D['median_rows_per_input_vector']:.0f}", "MAXREP": str(D["max_rows_per_input_vector"]), "NDISTINCT": f"{D['distinct_input_vectors']:,}",
       "NTWIN": f"{int(round(100*D['random_cv_test_rows_with_exact_input_twin_in_training']))}\\%", "MAD": f(D["replicate_label_MAD_mg_per_g"], 1),
       "MAEREL": f(sk("Capacity", "random", "MAE"), 1), "SKR1": f(sk("Capacity", "random")), "SKR2": f(sk("removal_pct", "random")),
       "SKT1": f(sk("Capacity", "replicate")), "SKT2": f(sk("removal_pct", "replicate")),
       "TWD1": f(gp.loc[("Capacity", "replicate"), "diff"], 1) + "~mg\\,g$^{-1}$", "TWD2": f(gp.loc[("removal_pct", "replicate"), "diff"], 1) + " percentage points",
       "SKA1": f(sk("Capacity", "adsorbent")), "SKA2": f(sk("removal_pct", "adsorbent")),
       "CIA1": ci(ca.skill_lo, ca.skill_hi), "CIA2": ci(ra.skill_lo, ra.skill_hi),
       "SKP1": f(sk("Capacity", "pollutant")), "SKP2": f(sk("removal_pct", "pollutant")), "SKQ1": f(sk("Capacity", "pair")), "SKQ2": f(sk("removal_pct", "pair")),
       "POSC": str(pos_cap_n), "NEGC": str(len(neg_cap)), "NEGLIST": neg_txt, "POSR": str(pos_rm_n),
       "HGBCAP": f(bm.loc[("Capacity", "adsorbent"), "HistGB"], 1), "RFCAP": f(bm.loc[("Capacity", "adsorbent"), "RandomForest"], 1), "MEANCAP": f(bm.loc[("Capacity", "adsorbent"), "Mean"], 1),
       "RIDREM": f(bm.loc[("removal_pct", "adsorbent"), "Ridge"], 1), "RFREM": f(bm.loc[("removal_pct", "adsorbent"), "RandomForest"], 1),
       "ABBIC": f(sA("Capacity", "adsorbent", BI)), "ABBIR": f(sA("removal_pct", "adsorbent", BI)), "ABALC": f(sA("Capacity", "adsorbent", AL)), "ABALR": f(sA("removal_pct", "adsorbent", AL)),
       "ABCOC": f(sA("Capacity", "adsorbent", CO)), "ABCOR": f(sA("removal_pct", "adsorbent", CO)),
       "ABRBIC": f(sA("Capacity", "random", BI)), "ABRBIR": f(sA("removal_pct", "random", BI)), "ABRCOC": f(sA("Capacity", "random", CO)), "ABRCOR": f(sA("removal_pct", "random", CO)),
       "IMPCAP": f(top["Capacity"].importance[0], 3),
       "LC2C": f(lc.loc[("Capacity", 2), "skill"]), "LC2R": f(lc.loc[("removal_pct", 2), "skill"]), "LC14C": f(lc.loc[("Capacity", 14), "skill"]), "LC14R": f(lc.loc[("removal_pct", 14), "skill"]),
       "DRD": f(der.loc["random", "skill_direct"]), "DRV": f(der.loc["random", "skill_derived"]), "DAD": f(der.loc["adsorbent", "skill_direct"]), "DAV": f(der.loc["adsorbent", "skill_derived"]),
       "DAJUDGE": ("so the improvement is statistically clear." if DER_CLEAR else "which includes zero, so with 15 biochars this improvement is suggestive and not established."),
       "DAMAE": f(der.loc["adsorbent", "diff_direct_minus_derived"], 1), "DERNOTE": ("statistically clear" if DER_CLEAR else "suggestive only, as the interval of the difference includes zero"), "DACI": ci(der.loc["adsorbent", "diff_lo"], der.loc["adsorbent", "diff_hi"], 1),
       "COVLO": f(CONF.coverage_pooled.min()), "COVHI": f(CONF.coverage_pooled.max()), "WMIN": f(wmin, 1), "WMAX": f(wmax, 1),
       "COVMIN": f(cap80.min()), "SHLO": str(int(round(100 * c80.share_biochars_below_nominal.min()))), "SHHI": str(int(round(100 * c80.share_biochars_below_nominal.max()))),
       "R_ROWS": f"{D2['rows']:,}", "R_METALS": str(D2["metals"]), "R_ADS": str(D2["adsorbents"]), "R_PAIRS": str(D2["pairs"]), "R_TWIN": f"{int(round(100*D2['rows_with_identical_input_elsewhere']))}\\%",
       "R_SK1": f(RM.loc[("qe","random"),"skill"]), "R_SK2": f(RM.loc[("removal_pct","random"),"skill"]),
       "R_AD1": f(RM.loc[("qe","adsorbent"),"skill"]), "R_AD1CI": ci(RM.loc[("qe","adsorbent"),"skill_lo"], RM.loc[("qe","adsorbent"),"skill_hi"]),
       "R_AD2": f(RM.loc[("removal_pct","adsorbent"),"skill"]), "R_AD2CI": ci(RM.loc[("removal_pct","adsorbent"),"skill_lo"], RM.loc[("removal_pct","adsorbent"),"skill_hi"]),
       "R_PQ1": f(RM.loc[("qe","pair"),"skill"]), "R_PQ2": f(RM.loc[("removal_pct","pair"),"skill"]), "R_PO1": f(RM.loc[("qe","pollutant"),"skill"]), "R_PO2": f(RM.loc[("removal_pct","pollutant"),"skill"]),
       "R_TW1": f(RM.loc[("qe","replicate"),"skill"]), "R_TW2": f(RM.loc[("removal_pct","replicate"),"skill"]),
       "R_COC": f(RA.loc[("qe","adsorbent",CO),"skill"]), "R_ALC": f(RA.loc[("qe","adsorbent",AL),"skill"]), "R_BIC": f(RA.loc[("qe","adsorbent",BI),"skill"]), "R_BICCI": ci(RA.loc[("qe","adsorbent",BI),"skill_lo"], RA.loc[("qe","adsorbent",BI),"skill_hi"]),
       "TABR": TABR, "TABRA": TABRA,
       "SKV": S["sklearn"], "TAB1": TAB1, "TAB2": TAB2, "TAB3": TAB3, "TAB4": TAB4, "TAB5": TAB5, "TAB6": TAB6, "TABA": TABA, "TABL": TABL, "TABD": TABD}
tex = TEMPLATE
for k, v in rep.items(): tex = tex.replace(f"@@{k}@@", v)
assert "@@" not in tex
tex = tex.replace(r"\begin{tabular}", r"\resizebox{\linewidth}{!}{\begin{tabular}").replace(r"\end{tabular}", r"\end{tabular}}")
open(f"{OUT}/paper.tex", "w", encoding="utf-8").write(tex)
os.makedirs(f"{OUT}/figures", exist_ok=True)
for fn in ["fig1_skill_by_protocol.png", "fig2_parity_biochar_held_out.png", "fig3_conformal_coverage_by_biochar.png", "fig4_permutation_importance.png", "fig5_learning_curve.png", "fig6_input_group_ablation.png"]:
    shutil.copy(f"{R}/{fn}", f"{OUT}/figures/{fn}")
HL = [f"Random CV overstates biochar adsorption skill: {f(sk('Capacity','random'),2)} vs {f(sk('Capacity','adsorbent'),2)} on unseen biochars",
      "Removing exact input twins barely matters; withholding whole biochars does",
      "Ablation result is data-set specific: not reproduced on a second public data set",
      f"Capacity via predicted removal: held-out skill {f(der.loc['adsorbent','skill_direct'])} to {f(der.loc['adsorbent','skill_derived'])} (suggestive)" if not DER_CLEAR else f"Capacity via predicted removal: held-out skill {f(der.loc['adsorbent','skill_direct'])} to {f(der.loc['adsorbent','skill_derived'])}",
      "Skill still rising at 14 training biochars: the data set is too small"]
assert all(len(h) <= 85 for h in HL), [(len(h), h) for h in HL]
open(f"{OUT}/highlights.txt", "w").write("\n".join(HL) + "\n")
print("paper.tex written")
