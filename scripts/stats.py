import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1.  one‐hot mapping & the cohort gene expression data
mapping_df = pd.read_csv("Cohort1_ZeroShot18_SymptomMappingScores.csv")
cohort_df   = pd.read_excel("Cohort1 Symptoms6 months or later.3-6Months.xlsx")

# 2. Merge so each row has gene expression +  18 binary symptom flags
df = pd.merge(
    cohort_df,
    mapping_df,
    on=["PatientID", "Sampleid", "Symptoms"],
    how="left"
)

# 3. original  symptom columns
symptom_cols = [
    "Fatigue", "Pain", "Headache", "Fever", "Cough",
    "Shortness of Breath", "Chest Pain", "Brain Fog",
    "Dizziness", "Abdominal Pain", "Palpitations",
    "Sleep Disorders", "Neurological", "Mental Health",
    "Skin", "Oral", "Loss of Smell/Taste", "Edema"
]

# 4. excluding metadata & symptom flags and select all gene‐pathway columns
exclude = ["PatientID", "Sampleid", "Symptoms", "GENDER", "DAYS POST INFECTION", "GROUP"] + symptom_cols
gene_cols = [c for c in df.columns if c not in exclude]

# 5.  “average expression per symptom” matrix:
#    sum_of_expr_for_patients_with_symptom / count_of_patients_with_symptom
counts = df[symptom_cols].sum().values  # total patients per symptom
corr_mat = df[symptom_cols].T.dot(df[gene_cols]) / counts[:, None]

# 6. heatmap plot
plt.figure(figsize=(20, 10))
sns.heatmap(
    corr_mat,
    cmap="coolwarm",
    vmin=corr_mat.values.min(),
    vmax=corr_mat.values.max(),
    cbar_kws={"label": "Avg Expression"}
)
plt.title("Average Gene Expression per Symptom Category")
plt.xlabel("Gene Expression Pathway")
plt.ylabel("Symptom Category")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 7. Top 3 Gene Pathways per Symptom
top_n = 3
colors = ["#e74c3c", "#f39c12", "#f1c40f"]  # red=smallest, orange=mid, yellow=largest
fig, axes = plt.subplots(6, 3, figsize=(18, 20))
axes = axes.flatten()

for i, symptom in enumerate(symptom_cols):
    top3 = corr_mat.loc[symptom].nlargest(top_n)
    # sort small→large so bar colors map correctly
    top3_sorted = top3.sort_values(ascending=True)

    ax = axes[i]
    sns.barplot(
        x=top3_sorted.values,
        y=top3_sorted.index,
        palette=colors,
        ax=ax
    )
    ax.set_title(f"Top {top_n} Genes – {symptom}")
    ax.set_xlabel("Avg Expression")
    ax.set_xlim(0, corr_mat.values.max() * 1.05)
    ax.set_ylabel("")

# unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
