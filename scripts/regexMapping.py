import pandas as pd
import re

# 1. Load your cohort CSV
df = pd.read_csv(
    r"C:\Users\NazimAhmed_Belabbaci\PycharmProjects\CovidNLP\Cohort1 Symptoms6 months or later.3-6Months.csv",
    encoding="utf-8"
)

# 2.  original 18 symptom categories with regex patterns
original_18_patterns = {
    "Fatigue":             r"\bfatigue\b|\btired\b|\bweakness\b|\bweak\b|\bexhaust\b|\bmalaise\b|\bdifficulty to exercise\b",
    "Pain":                r"\bpain\b|\bache\b|\bhurts\b|\bsore\b",
    "Headache":            r"\bheadache\b|\bmigraine\b",
    "Fever":               r"\bfever\b|\bchill\b|\bflush\b|\bsweat\b",
    "Cough":               r"\bcough\b|\bcoughing\b",
    "Shortness of Breath": r"\bshortness of breath\b|\bdifficulty breathing\b|\bbreathless\b|\bdyspnea\b",
    "Chest Pain":          r"\bchest pain\b|\btight chest\b",
    "Brain Fog":           r"\bbrain fog\b|\bconcentration\b|\bconfusion\b|\bmemory\b|\bfocus\b|\bcognitive\b",
    "Dizziness":           r"\bdizzy\b|\bdizziness\b|\bvertigo\b|\blightheaded\b|\bfaint\b",
    "Abdominal Pain":      r"\bnausea\b|\bvomit\b|\bvomiting\b|\bdiarrhea\b|\bbladder\b|\babdominal\b|\bstomach\b",
    "Palpitations":        r"\bpalpitations\b|\bracing heart\b|\barrhythmia\b|\bheartbeat\b|\btachycardia\b",
    "Sleep Disorders":     r"\bsleep\b|\binsomnia\b|\bapnea\b|\bstopping breathing during sleep\b",
    "Neurological":        r"\btremor\b|\bshaking\b|\babnormal movement\b|\bseizure\b|\bnerve\b",
    "Mental Health":       r"\banxiety\b|\bdepression\b|\bstress\b|\bmood\b",
    "Skin":                r"\brash\b|\bskin\b|\bred\b|\bwhite discoloration\b|\bblue discoloration\b",
    "Oral":                r"\bdry mouth\b|\bmouth\b|\boral\b|\btaste\b|\bsaliva\b|\btongue\b",
    "Loss of Smell/Taste": r"\bloss of smell\b|\bloss of taste\b|\bchange in smell\b|\bchange in taste\b",
    "Edema":               r"\bswelling\b|\bedema\b"
}

# 3.  mapping DataFrame
mapping_df = df[["PatientID", "Sampleid", "Symptoms"]].copy()

# 4.  one-hot encoding
for category, pattern in original_18_patterns.items():
    mapping_df[category] = (
        df["Symptoms"]
          .str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
          .astype(int)
    )

# 5. Save to CSV
output_path = r"C:\Users\NazimAhmed_Belabbaci\PycharmProjects\CovidNLP\Cohort1_Original18_SymptomMapping.csv"
mapping_df.to_csv(output_path, index=False)

print(f"One-hot mapping saved to: {output_path}")
