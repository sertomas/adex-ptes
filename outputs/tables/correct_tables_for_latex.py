import pandas as pd

# Run this code after hp_simult and orc_simult

# Load the data
df_hp_streams_base = pd.read_csv('../adex_hp/hp_streams_real.csv', index_col=0)
df_orc_streams_base = pd.read_csv('../adex_orc/orc_streams_real.csv', index_col=0)

# Drop the last two columns
df_hp_streams_base = df_hp_streams_base.iloc[:-4, :-2]
df_orc_streams_base = df_orc_streams_base.iloc[:-4, :-2]

df_hp_streams_base.round(2).to_csv('hp_base_results.csv')
df_orc_streams_base.round(2).to_csv('orc_base_results.csv')

df_hp_mexo = pd.read_csv('../adex_hp/hp_adex_analysis.csv', index_col=0)
df_orc_mexo = pd.read_csv('../adex_orc/orc_adex_analysis.csv', index_col=0)

new_column_names = ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A']  # to avoid compiling error in Latex
df_hp_mexo.columns = new_column_names
df_orc_mexo.columns = new_column_names
df_hp_mexo.to_csv('hp_adex_analysis.csv')
df_orc_mexo.to_csv('orc_adex_analysis.csv')

df_hp_mexo = pd.read_csv('../adex_hp/hp_adex_analysis.csv', index_col=[0, 1])
sum_values_by_l = df_hp_mexo.groupby(level=1)['ED EX l [kW]'].sum()
print(sum_values_by_l)

df_orc_mexo = pd.read_csv('../adex_orc/orc_adex_analysis.csv', index_col=[0, 1])
sum_values_by_l = df_orc_mexo.groupby(level=1)['ED EX l [kW]'].sum()
print(sum_values_by_l)