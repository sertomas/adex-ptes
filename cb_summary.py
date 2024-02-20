import numpy as np
import pandas as pd

df_hp_streams_base = pd.read_csv('outputs/adex_hp/hp_streams_real.csv', index_col=0)
df_orc_streams_base = pd.read_csv('outputs/adex_orc/orc_streams_real.csv', index_col=0)

print(df_hp_streams_base)

