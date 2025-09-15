import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Load the CSV ===
df = pd.read_csv("merged_filtered_symptom_pathway_data3-6.csv")

# === 2. Define symptom mapping ===
token_to_categories = {
    "anaphylaxis": ["Skin Symptoms"], "bladder": ["Abdominal Pain"], "cold": ["Fever", "Cough"],
    "color": ["Vision"], "cough": ["Cough"], "dryeyes": ["Vision"], "fatigue": ["Fatigue"],
    "goofy": ["Neurological Symptoms"], "headache": ["Headache"], "hearing": ["Neurological Symptoms"],
    "heart": ["Palpitations", "Chest Pain"], "itching": ["Skin Symptoms"], "malaise": ["Fatigue"],
    "mood": ["Mental Health"], "nerve": ["Neurological Symptoms"], "pain": ["Pain"],
    "rash": ["Skin Symptoms"], "sense": ["Loss of Smell or Taste"], "sinus": ["Cough", "Headache"],
    "sleep": ["Sleep Disorders"], "smellsick": ["Loss of Smell or Taste"],
    "sob": ["Shortness of Breath"], "soreness": ["Pain"], "temp": ["Fever"],
    "think": ["Brain Fog"], "vision": ["Vision"], "weak": ["Fatigue"], "wheeze": ["Shortness of Breath"]
}

SYMPTOM_LABELS = [
    "Fatigue", "Pain", "Headache", "Fever", "Cough", "Shortness of Breath", "Chest Pain",
    "Brain Fog", "Dizziness", "Abdominal Pain", "Palpitations", "Sleep Disorders",
    "Neurological Symptoms", "Mental Health", "Skin Symptoms", "Oral Symptoms",
    "Loss of Smell or Taste", "Edema", "Vision"
]

# === 3.  one-hot matrix for symptom categories ===
symptom_matrix = pd.DataFrame(0, index=df.index, columns=SYMPTOM_LABELS)
for i, row in df.iterrows():
    symptoms = str(row['symptom_list']).split(",") if pd.notna(row['symptom_list']) else []
    seen = set()
    for raw_symptom in symptoms:
        token = raw_symptom.strip().lower()
        categories = token_to_categories.get(token, [])
        for cat in categories:
            if cat in symptom_matrix.columns and cat not in seen:
                symptom_matrix.at[i, cat] = 1
                seen.add(cat)

# === 4. Identify gene expression columns ===
gene_cols = [col for col in df.columns if col not in ["Patient ID", "symptom_list"]]
gene_cols = [col for col in gene_cols if pd.api.types.is_numeric_dtype(df[col])]

print(gene_cols)
# === 5. Compute average expression per symptom ===
counts = symptom_matrix.sum().values  # number of patients per symptom category
print(counts)
corr_mat = symptom_matrix.T.dot(df[gene_cols]) / counts[:, None]


#  Save the matrix to a CSV file
corr_mat.to_csv("average_expression_per_symptom_3-6M.csv")

# === 6. Plot heatmap ===
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