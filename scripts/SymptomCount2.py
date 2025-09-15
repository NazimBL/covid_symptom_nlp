import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict

df = pd.read_excel("cohort2_symptom_list.xlsx")

# mapping from raw tokens to 18 symptom categories
token_to_categories = {
    "anaphylaxis": ["Skin Symptoms"],
    "bladder": ["Abdominal Pain"],
    "cold": ["Fever", "Cough"],
    "color": ["Vision"],
    "cough": ["Cough"],
    "dryeyes": ["Vision"],
    "fatigue": ["Fatigue"],
    "goofy": ["Neurological Symptoms"],
    "headache": ["Headache"],
    "hearing": ["Neurological Symptoms"],
    "heart": ["Palpitations", "Chest Pain"],
    "itching": ["Skin Symptoms"],
    "malaise": ["Fatigue"],
    "mood": ["Mental Health"],
    "nerve": ["Neurological Symptoms"],
    "pain": ["Pain"],
    "rash": ["Skin Symptoms"],
    "sense": ["Loss of Smell or Taste"],
    "sinus": ["Cough", "Headache"],
    "sleep": ["Sleep Disorders"],
    "smellsick": ["Loss of Smell or Taste"],
    "sob": ["Shortness of Breath"],
    "soreness": ["Pain"],
    "temp": ["Fever"],
    "think": ["Brain Fog"],
    "vision": ["Vision"],
    "weak": ["Fatigue"],
    "wheeze": ["Shortness of Breath"]
}

SYMPTOM_LABELS = [
    "Fatigue", "Pain", "Headache", "Fever", "Cough",
    "Shortness of Breath", "Chest Pain", "Brain Fog",
    "Dizziness", "Abdominal Pain", "Palpitations",
    "Sleep Disorders", "Neurological Symptoms", "Mental Health",
    "Skin Symptoms", "Oral Symptoms", "Loss of Smell or Taste", "Edema",
    "Vision"
]

# Count symptoms per PASC group
def classify_symptoms_mapped(df, mapping):
    symptom_map = defaultdict(lambda: {'PASC': 0, 'NO PASC': 0})
    for _, row in df.iterrows():
        status = row['PASCSTATUS']
        symptoms = str(row['symptom_list']).split(",") if pd.notna(row['symptom_list']) else []
        seen = set()
        for raw_symptom in symptoms:
            token = raw_symptom.strip().lower()
            categories = mapping.get(token, [])
            for cat in categories:
                if cat not in seen:
                    symptom_map[cat][status] += 1
                    seen.add(cat)
    return symptom_map

# Create summary table
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

# Plotting function
def plot_symptom_counts(df, title):
    df_plot = df.set_index("Symptom")[["Symptom count for PASC", "Symptom count for NO PASC"]]
    ax = df_plot.plot(kind='bar', figsize=(14, 6), color=["blue", "red"])
    ax.set_xlabel("Symptom")
    ax.set_ylabel("Number of Patients")
    ax.set_title(title)
    ax.legend(title="Group")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show(block=True)

# === Run the analysis ===
symptom_counts = classify_symptoms_mapped(df, token_to_categories)
total_pasc = (df['PASCSTATUS'] == 'PASC').sum()
total_no_pasc = (df['PASCSTATUS'] == 'NO PASC').sum()

summary_df = convert_to_summary(symptom_counts, total_pasc, total_no_pasc)

summary_df.to_csv("cohort2_symptom_counts.csv", index=False)

# Plot
plot_symptom_counts(summary_df, "Cohort 2: Symptom Count Comparison (PASC vs NO PASC)")
