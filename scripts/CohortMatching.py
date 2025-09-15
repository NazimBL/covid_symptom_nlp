import pandas as pd

# Load the filtered pathway data
filtered_pathway_df = pd.read_csv("filtered_PASC_3-12M_samples.csv")

# Load the symptom list data
symptom_df = pd.read_excel("SymptomList_38IDs.cohort2.xlsx")

# Extract patient ID prefix (e.g., 'BI4001') from both datasets
filtered_pathway_df['patient_id_prefix'] = filtered_pathway_df.iloc[:, 0].astype(str).str.extract(r'^(BI\d+)')
symptom_df['patient_id_prefix'] = symptom_df.iloc[:, 0].astype(str).str.extract(r'^(BI\d+)')

# merge based on the patient ID prefix
merged_df = pd.merge(symptom_df, filtered_pathway_df, on='patient_id_prefix', how='inner')

# Drop the helper column after the merge
merged_df = merged_df.drop(columns=['patient_id_prefix'])

# Save the merged dataframe to a new CSV
merged_df.to_csv("merged_filtered_symptom_pathway_data3-12M.csv", index=False)

print(merged_df.head())
