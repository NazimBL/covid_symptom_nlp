import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import difflib
import os
from transformers import pipeline
from collections import defaultdict

# Disable TensorFlow usage for Transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"

#zero-shot classification (bart large mnli)
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    framework="pt",
    device=-1
)

SYMPTOM_LABELS = [
    "Fatigue", "Pain", "Headache", "Fever", "Cough",
    "Shortness of Breath", "Chest Pain", "Brain Fog",
    "Dizziness", "Abdominal Pain", "Palpitations",
    "Sleep Disorders", "Neurological Symptoms", "Mental Health",
    "Skin Symptoms", "Oral Symptoms", "Loss of Smell or Taste", "Edema"
]

# Fuzzy match helper
def fuzzy_match(symptom, label_set, cutoff=0.6):
    matches = difflib.get_close_matches(symptom.lower(), [label.lower() for label in label_set], n=1, cutoff=cutoff)
    if matches:
        match_index = [label.lower() for label in label_set].index(matches[0])
        return label_set[match_index]
    return None

# Cohort 1 (BART classification)
def classify_symptoms_bart(df, threshold=0.6):
    symptom_map = defaultdict(lambda: {'PASC': 0, 'NO PASC': 0})
    for _, row in df.iterrows():
        status = row['PASCSTATUS']
        text = str(row['Symptom Notes']) if pd.notna(row['Symptom Notes']) else ""
        result = classifier(text, candidate_labels=SYMPTOM_LABELS, multi_label=True)
        for label, score in zip(result['labels'], result['scores']):
            if score >= threshold:
                symptom_map[label][status] += 1
    return symptom_map

# Function for Cohort 2 (fuzzy matching)
def classify_symptoms_fuzzy(df):
    symptom_map = defaultdict(lambda: {'PASC': 0, 'NO PASC': 0})
    for _, row in df.iterrows():
        status = row['PASCSTATUS']
        symptoms = str(row['symptom_list']).split(",") if pd.notna(row['symptom_list']) else []
        seen = set()
        for symptom in symptoms:
            symptom_clean = symptom.strip().lower()
            matched = fuzzy_match(symptom_clean, SYMPTOM_LABELS)
            if matched and matched not in seen:
                symptom_map[matched][status] += 1
                seen.add(matched)
    return symptom_map

# Convert to summary table
def convert_to_summary(symptom_map, total_pasc, total_no_pasc):
    data = []
    for symptom in SYMPTOM_LABELS:
        pasc_count = symptom_map[symptom]['PASC']
        no_pasc_count = symptom_map[symptom]['NO PASC']
        pasc_pct = pasc_count / total_pasc * 100 if total_pasc else 0
        no_pasc_pct = no_pasc_count / total_no_pasc * 100 if total_no_pasc else 0
        data.append({
            "Symptom": symptom,
            "Symptom count for PASC": pasc_count,
            "Symptom count for NO PASC": no_pasc_count,
            "Symptom percent for PASC": pasc_pct,
            "Symptom percent for NO PASC": no_pasc_pct
        })
    return pd.DataFrame(data)

# Plotting
def plot_symptom_counts(df, title):
    df_plot = df.set_index("Symptom")[["Symptom count for PASC", "Symptom count for NO PASC"]]
    ax = df_plot.plot(kind='bar', figsize=(14, 6), color=["blue", "red"])
    ax.set_xlabel("Symptom")
    ax.set_ylabel("Number of Patients")
    ax.set_title(title)
    ax.legend(title="Group")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# === Main Processing ===

# Load files
cohort1_df = pd.read_excel("cohort1_clinical_symptoms.xlsx")
cohort2_df = pd.read_excel("cohort2_symptom_list.xlsx")

# Cohort 1 classification
print("Classifying symptoms for Cohort 1...")
symptom_counts_cohort1 = classify_symptoms_bart(cohort1_df)
total_pasc1 = (cohort1_df['PASCSTATUS'] == 'PASC').sum()
total_no_pasc1 = (cohort1_df['PASCSTATUS'] == 'NO PASC').sum()
summary_cohort1 = convert_to_summary(symptom_counts_cohort1, total_pasc1, total_no_pasc1)
summary_cohort1.to_csv("cohort1_symptom_counts.csv", index=False)
plot_symptom_counts(summary_cohort1, "Cohort 1: Symptom Count Comparison (PASC vs NO PASC)")

# Cohort 2 classification
print("Classifying symptoms for Cohort 2...")
symptom_counts_cohort2 = classify_symptoms_fuzzy(cohort2_df)
total_pasc2 = (cohort2_df['PASCSTATUS'] == 'PASC').sum()
total_no_pasc2 = (cohort2_df['PASCSTATUS'] == 'NO PASC').sum()
summary_cohort2 = convert_to_summary(symptom_counts_cohort2, total_pasc2, total_no_pasc2)
summary_cohort2.to_csv("cohort2_symptom_counts.csv", index=False)
plot_symptom_counts(summary_cohort2, "Cohort 2: Symptom Count Comparison (PASC vs NO PASC)")
