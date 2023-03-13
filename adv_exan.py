from cb_set_pars import hp_settings_expander
from cb_network import hp_expander, hp_network

hp_exp = hp_expander(['R245fa', 'water'])

delta_t_min = 5  # minimum temperature difference [K]
T_amb = 10  # temperature of ambient [°C]
p_amb = 1.013  # pressure of ambient [bar]
hp_settings_expander(hp_exp, T_amb, p_amb, delta_t_min)

hp_exp.solve(mode='design')
hp_exp.results['Connection'].round(3).to_csv("hp_expander_results.csv")


