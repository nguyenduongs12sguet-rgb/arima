import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Function to clean and convert columns to float
def clean_numeric(col):
    return pd.to_numeric(col.str.replace('"', '').str.replace(',', '.'), errors='coerce')

# Temperature post-hoc
df_temp = pd.read_csv(r"E:\FINAL OF FINAL\data\temperature.csv", sep=';')
df_temp['Temp'] = clean_numeric(df_temp['Average Mean Surface Air Temperature (Annual Mean )'])
tukey_temp = pairwise_tukeyhsd(df_temp['Temp'], df_temp['Area'])
print("Temperature Post-Hoc:")
print(tukey_temp)

# Pesticides post-hoc (using total)
df_pest = pd.read_csv(r"E:\FINAL OF FINAL\data\thuoctrusau.csv", sep=';')
df_pest['Pesticides_total'] = pd.to_numeric(df_pest['Pesticides (total)'], errors='coerce')
tukey_pest = pairwise_tukeyhsd(df_pest['Pesticides_total'], df_pest['Area'])
print("Pesticides Post-Hoc:")
print(tukey_pest)

# Fertilizer post-hoc
df_fert = pd.read_csv(r"E:\FINAL OF FINAL\data\phanbontieuthu.csv", sep=';')
df_fert['Fertilizer_kg_ha'] = clean_numeric(df_fert['Fertilizer (kg/ha)'])
tukey_fert = pairwise_tukeyhsd(df_fert['Fertilizer_kg_ha'], df_fert['Area'])
print("Fertilizer Post-Hoc:")
print(tukey_fert)

# Precipitation post-hoc
df_rain = pd.read_csv(r"E:\FINAL OF FINAL\data\newrain.csv", sep=';')
df_rain['Precip_mm'] = clean_numeric(df_rain['Precipitation (mm) '])
tukey_rain = pairwise_tukeyhsd(df_rain['Precip_mm'], df_rain['Area'])
print("Precipitation Post-Hoc:")
print(tukey_rain)

# Yield post-hoc (potato yield kg/ha)
df_yield = pd.read_csv(r"E:\FINAL OF FINAL\data\nangsuat_potatoes_only.csv", sep=';')
df_yield['Yield_kg_ha'] = pd.to_numeric(df_yield['Yield (kg/ha)'], errors='coerce')
tukey_yield = pairwise_tukeyhsd(df_yield['Yield_kg_ha'], df_yield['Area'])
print("Yield Post-Hoc:")
print(tukey_yield)