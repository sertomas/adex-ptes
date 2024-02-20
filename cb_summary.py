import numpy as np
import pandas as pd

# Load the data
df_hp_streams_base = pd.read_csv('outputs/adex_hp/hp_streams_real.csv', index_col=0)
df_orc_streams_base = pd.read_csv('outputs/adex_orc/orc_streams_real.csv', index_col=0)

# Drop the last two columns
df_hp_streams_base = df_hp_streams_base.iloc[:-4, :-2]
df_orc_streams_base = df_orc_streams_base.iloc[:-4, :-2]

print(df_hp_streams_base)
print(df_orc_streams_base)

df_hp_streams_base.round(2).to_csv('outputs/hp_base_results.csv')
df_orc_streams_base.round(2).to_csv('outputs/orc_base_results.csv')

df_hp_mexo = pd.read_csv('outputs/adex_hp/hp_adex_analysis.csv', index_col=0)
df_hp_mexo.fillna(0, inplace=True)
print(df_hp_mexo)
df_hp_mexo.to_csv('outputs/adex_hp/hp_adex_analysis.csv')

epsilon = pd.read_csv('outputs/adex_orc/orc_comps_real.csv', index_col=0)['epsilon']
print(epsilon)