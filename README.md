# COAD
Validation of Biomarkers Predictive of Tumor Location in Coloadenocarcinoma, an Analysis of the TCGA COAD Dataset

Tumor localization correlates with prognosis in coloadenocarcinoma, with aboral
tumors having a better overall survival. This can be attributed to their better
response to biologicals such as the anti-EGFR (epidermal growth factor receptor)
cetuximab. It was hypothesized that –regardless of the causal relationships– this “sidedness”
of coloadenocarcinomas could be reconstructed on a genomic and transcriptomic
level.

A two machine learning models, `logistic regression` and `random forest`, were implemented in
`python3` using `pandas` and `sklearn`, among other libraries.

## Structure
There are two versions of the analysis, the original jupyter notebook, and a reworked script.
The script uses the harmonized gdc api, can be configured using config flags, and handles caching
more elegantly, and so should be preferred. However, `paper.pdf` references the jupyter notebook.
