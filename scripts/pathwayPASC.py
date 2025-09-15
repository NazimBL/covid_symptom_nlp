import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the one-hot symptom mapping and the cohort gene expression data
mapping_df = pd.read_csv("Cohort1_Original18_SymptomMapping.csv")
cohort_df = pd.read_excel("Cohort1 Symptoms6 months or later.3-6Months.xlsx")

# 2. Merge on PatientID, Sampleid, and Symptoms
df = cohort_df.merge(
    mapping_df,
    on=["PatientID", "Sampleid", "Symptoms"],
    how="left"
)

# 3.  original 18 symptom columns
symptom_cols = [
    "Fatigue", "Pain", "Headache", "Fever", "Cough",
    "Shortness of Breath", "Chest Pain", "Brain Fog",
    "Dizziness", "Abdominal Pain", "Palpitations",
    "Sleep Disorders", "Neurological Symptoms", "Mental Health",
    "Skin Symptoms", "Oral Symptoms", "Loss of Smell or Taste", "Edema"
]

# 4. excluding metadata & symptom flags and select all gene‐pathway columns
non_gene_cols = ["PatientID", "Sampleid", "Symptoms", "GENDER",
                 "DAYS POST INFECTION", "GROUP"] + symptom_cols
gene_cols = [c for c in df.columns if c not in non_gene_cols]

# 5. heatmap of gene expression
#    average expression per symptom = sum(expr for patients with symptom) / count(patients with symptom)
symptom_counts = df[symptom_cols].sum().replace(0, pd.NA)  # avoid division by zero
correlation_matrix = df[symptom_cols].T.dot(df[gene_cols]) / symptom_counts.values[:, None]

# 6.  heatmap of symptom vs. gene pathway average expression
plt.figure(figsize=(20, 10))
sns.heatmap(
    correlation_matrix,
    cmap="vlag",
    center=0,
    cbar_kws={"label": "Avg Expression"},
    xticklabels=True,
    yticklabels=True
)
plt.title("Average Gene Expression per Symptom Category")
plt.xlabel("Gene Expression Pathway")
plt.ylabel("Symptom Category")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 7.  Top 3 Gene Pathways per Symptom using horizontal bar plots
top_n = 3
n = len(symptom_cols)
ncols = 3
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 5 * nrows))
axes = axes.flatten()

for i, symptom in enumerate(symptom_cols):
    # Select top N genes for this symptom
    top_genes = correlation_matrix.loc[symptom].nlargest(top_n)
    ax = axes[i]
    sns.barplot(
        x=top_genes.values,
        y=top_genes.index,
        ax=ax,
        orient="h"
    )
    ax.set_title(f"Top {top_n} Genes – {symptom}")
    ax.set_xlabel("Avg Expression")
    ax.set_ylabel("")
    ax.set_xlim(0, correlation_matrix.max().max() * 1.1)

# unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()