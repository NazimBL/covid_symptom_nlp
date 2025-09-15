from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

# 1) Load excel sheets
xls = pd.ExcelFile("Cohort1 Symptoms6 months or later.3-6Months.xlsx")
dfs = {name: xls.parse(name) for name in xls.sheet_names}

# 2) Zero-shot classifier with a smaller MNLI model
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0  # -1 CPU only;  0 GPU
)

labels = [
    "Fatigue", "Pain", "Headache", "Fever", "Cough",
    "Shortness of Breath", "Chest Pain", "Brain Fog",
    "Dizziness", "Abdominal Pain", "Palpitations",
    "Sleep Disorders", "Neurological", "Mental Health",
    "Skin", "Oral", "Loss of Smell or Taste", "Edema"
]

# 3) Tally function
def count_symptoms(df, threshold=0.5):
    notes = df["Symptoms"].fillna("").tolist()
    results = classifier(notes, candidate_labels=labels, multi_label=True)
    counts = {lab: 0 for lab in labels}
    for r in results:
        for lab, score in zip(r["labels"], r["scores"]):
            if score >= threshold:
                counts[lab] += 1
    return counts

# 4) Compute counts per sheet
sheet_counts = {sheet: count_symptoms(df) for sheet, df in dfs.items()}

# 5) visuals
counts_df = pd.DataFrame(sheet_counts)
ax = counts_df.plot.bar(
    figsize=(14,6),
    color=["blue","red"]
)
ax.set_xlabel("Symptom Category")
ax.set_ylabel("Count of Notes")
ax.set_title("Symptom Counts per Sheet")
ax.legend(title="Sheet")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
