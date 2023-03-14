import pandas as pd

from cb_set_pars import hp_settings_compressor, hp_settings_throttle
from cb_network import hp_expander, hp_network
from cb_exergy import exerg_an_hp

# calculation of ED_EN of compressor
hp_compr = hp_expander(['R245fa', 'water'])

delta_t_min = 5  # minimum temperature difference [K]
T_amb = 10  # temperature of ambient [°C]
p_amb = 1.013  # pressure of ambient [bar]
hp_settings_compressor(hp_compr, T_amb, p_amb, delta_t_min)

# ERROR with CoolProp in the following two lines:
hp_compr.solve(mode='design')
hp_compr.results['Connection'].round(5).to_csv("hp_comp_results.csv")

base_case = pd.read_csv("hp_results.csv", index_col=0)
m7 = base_case.iloc[7, 0]

hp_thr = hp_network(['R245fa', 'water'])
hp_settings_throttle(hp_thr, T_amb, p_amb, delta_t_min, m7)
hp_thr.solve(mode='design')
hp_thr.results['Connection'].round(5).to_csv("hp_throttle_results.csv")

ex_an_hp_thr = exerg_an_hp(hp_thr, T_amb, p_amb)
(ex_an_hp_thr.connection_data / 1e3).round(5).to_csv("hp_ex_an_throttle_conn.csv")  # in kJ/kg
ex_an_hp_thr.component_data[ex_an_hp_thr.component_data["E_F"] > 0].round(5).to_csv("hp_ex_an_throttle_comp.csv")

# only the exergy analysis of the heat pump cycle is taken into account

s8 = base_case.iloc[8, 6]
s7 = base_case.iloc[7, 6]
s3 = hp_thr.results['Connection'].iloc[2, 6]
s4 = hp_thr.results['Connection'].iloc[3, 6]
m2 = m7 * (s8 - s7) / (s3 - s4)

e1 = (ex_an_hp_thr.connection_data / 1e3).iloc[0, 3]
e4 = (ex_an_hp_thr.connection_data / 1e3).iloc[3, 3]
ED_EN_THR = m2 * (e4 - e1)
print(ED_EN_THR)

