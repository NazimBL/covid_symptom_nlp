

scripts/        # independent experiments (zero-shot mapping, cohort filtering, plotting, etc.)

data/           # input CSV/XLSX
output/         # generated artifacts (figures, tables)
requirements.txt


### Script highlights
- **SymptomCount_NLP.py** — zero-shot classify clinical notes (Cohort 1) + fuzzy match list (Cohort 2); writes `cohort1_symptom_counts.csv` / `cohort2_symptom_counts.csv` and plots counts.   
- **main.py** — zero-shot classification across sheets in `Cohort1 Symptoms6 months or later.3-6Months.xlsx`; aggregates and plots counts by sheet.  
- **AvgGeneExpression.py** — builds one-hot symptom matrix from tokenized lists, computes symptom-by-pathway average expression, saves CSV and a heatmap. 
- **GeneExpressionMapping.py** — merges symptom one-hot flags with cohort pathways, computes average-expression matrix, plots heatmap + top-pathways per symptom.  
- **CohortFiltering.py** — filters pathway matrix IDs to PASC samples for 3–12 months and saves filtered CSV.  
- **ZeroShotMapping.py** — generates one-hot symptom flags + raw scores via BART; note it currently writes to an absolute Windows path—change to a relative path before running. 


