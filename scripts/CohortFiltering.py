import pandas as pd

df = pd.read_csv("LC_3-6M_pathway_matrix_cohort2.csv")

# Ensure the first column contains sample labels
id_col = df.columns[0]

# Convert ID column to string just in case
df[id_col] = df[id_col].astype(str)

# Filter rows that contain both 'PASC' and '3_6M' in the ID column
df_filtered = df[df[id_col].str.contains("3_6M_PASC") | df[id_col].str.contains("6_12M_PASC")]

# Save or view result
df_filtered.to_csv("filtered_PASC_3-12M_samples.csv", index=False)

# Optional: Print a few filtered sample names
print(df_filtered[id_col].head())
