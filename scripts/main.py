import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from transformers import pipeline


xls = pd.ExcelFile("Cohort1 Symptoms6 months or later.3-6Months.xlsx")
dfs = {name: xls.parse(name) for name in xls.sheet_names}


classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    framework="pt",
    device=-1
)

labels = [
    "Fatigue", "Pain", "Headache", "Fever", "Cough",
    "Shortness of Breath", "Chest Pain", "Brain Fog",
    "Dizziness", "Abdominal Pain", "Palpitations",
    "Sleep Disorders", "Neurological Symptoms", "Mental Health",
    "Skin Symptoms", "Oral Symptoms", "Loss of Smell or Taste", "Edema"
]

def count_symptoms(df, threshold=0.5):
    notes = df["Symptoms"].fillna("").tolist()
    results = classifier(notes, candidate_labels=labels, multi_label=True)
    counts = {lab: 0 for lab in labels}
    for r in results:
        for lab, score in zip(r["labels"], r["scores"]):
            if score >= threshold:
                counts[lab] += 1
    return counts


sheet_counts = {sheet: count_symptoms(df, threshold=0.5) for sheet, df in dfs.items()}

counts_df = pd.DataFrame(sheet_counts)


ax = counts_df.plot(
    kind="bar",
    figsize=(14, 6),
    color=["blue", "red"]
)
ax.set_xlabel("Symptom Category")
ax.set_ylabel("Count of Notes")
ax.set_title("Symptom Counts per Sheet")
ax.legend(title="Sheet")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()