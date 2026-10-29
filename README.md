# Held-out validation of machine-learning models for biochar adsorption (two public data sets)

Code and results for the manuscript "How Well Can Machine Learning Predict Adsorption on a Biochar It Has Never Seen? Evidence from Two Public Data Sets" (Priom Dey, Avro Talukder, Yuvraj Roy; Department of Chemistry, Shahjalal University of Science and Technology, Sylhet, Bangladesh).

## Data (not included)
The experimental data are public files from the same research group and are downloaded at run time. Please cite the original works.
- `Raw_data.csv`: https://github.com/ZeeshanHJ/Adsorption-capacity-prediction-for-ECs (commit 25d8188af64b2086e2285d4e8cfef17e621c684a)
- `HMI_data.xlsx`: https://github.com/ZeeshanHJ/Transformer_models_prediction (commit 086671f34cafb0d7d88a473a4eda05be32376890)

Neither repository has a licence file, so the data are not redistributed here; check their terms before reuse.

## Run
Open `biochar_adsorption_held_out_validation.ipynb` in Google Colab and run all cells (about 12-15 minutes). Outputs appear in `adsorption_results/`.

## Files
- `biochar_adsorption_held_out_validation.ipynb` - analysis notebook
- `*.csv`, `summary.json` - result tables of the reported run (seed 42, scikit-learn 1.6.1); `replication_*.csv` belong to the second data set
- `fig1`-`fig6` `.png` - figures as produced by the notebook. In the manuscript the numbering differs for two files: the ablation figure (`fig6_input_group_ablation.png` here) is Figure 3 and the conformal-coverage figure (`fig3_conformal_coverage_by_biochar.png` here) is Figure 6.
- `check_numbers_and_build_draft.py` - reads the result files, asserts every quantitative statement made in the manuscript, and writes a draft of the manuscript text (`python check_numbers_and_build_draft.py --res . --out draft_out`). The submitted manuscript was edited by hand afterwards.
- `requirements.txt` - Python packages
- `LICENSE` (MIT, code), `CITATION.cff`

## Licence note
Code: MIT. Result tables and figures: CC BY 4.0. No data from the original repositories are included.
