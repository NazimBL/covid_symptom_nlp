import pandas as pd
from transformers import pipeline

# --- 1. load  csv
df = pd.read_csv("Cohort1 Symptoms6 months or later.3-6Months.csv",
    encoding="utf-8"
)

# --- 2. Instantiate the zero-shot classifier ---
#     facebook/bart-large-mnli under the hood.
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0  # -1 CPU only;  0 GPU
)

# --- 3. symptom labels ---
labels = [
    "Fatigue", "Pain", "Headache", "Fever", "Cough",
    "Shortness of Breath", "Chest Pain", "Brain Fog",
    "Dizziness", "Abdominal Pain", "Palpitations",
    "Sleep Disorders", "Neurological", "Mental Health",
    "Skin", "Oral", "Loss of Smell/Taste", "Edema"
]

# --- 4. classification  ---
#    returns a list of dicts: each has 'labels' and 'scores' for that description.
results = classifier(
    df["Symptoms"].tolist(),
    candidate_labels=labels,
    multi_label=True
)

# --- 5. Convert results into a DataFrame of one-hot flags ---
threshold = 0.7  # adjust between 0.3–0.7 to trade recall vs precision


out = pd.DataFrame({
    "PatientID": df["PatientID"],
    "Sampleid":  df["Sampleid"],
    "Symptoms":  df["Symptoms"],
})

# for each symptom label, set 1 if score > threshold else 0
for lab in labels:
    out[lab] = [
        int(lab in result["labels"] and result["scores"][result["labels"].index(lab)] >= threshold)
        for result in results
    ]

# --- 6.  keep the raw scores  ---
for lab in labels:
    out[f"{lab}_score"] = [
        (result["scores"][result["labels"].index(lab)]
         if lab in result["labels"] else 0.0)
        for result in results
    ]

# --- 7.  mapping to CSV ---
out.to_csv(
    r"C:\Users\NazimAhmed_Belabbaci\PycharmProjects\CovidNLP\Cohort1_ZeroShot18_SymptomMapping.csv",
    index=False
)

print("Zero‐Shot mapping complete! File saved.")
