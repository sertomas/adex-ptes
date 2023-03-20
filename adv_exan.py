import pandas as pd

from cb_set_pars import hp_settings_throttle, hp_settings_234
from cb_network import hp_network, hp_234
from cb_exergy import exerg_an_hp

# the base case is read from the results of the simulation
bc_s = pd.read_csv("hp_results.csv", index_col=0)  # streams of base case
bc_c = pd.read_csv("hp_ex_an_comp.csv", index_col=0)  # components of base case
# example: bc_s.loc[2, 's'] calls the entropy of stream 2

m2 = bc_s.loc[2, 'm']  # of the base case
m7 = bc_s.loc[7, 'm']  # of the base case
s8 = bc_s.loc[8, 's']  # of the base case
s7 = bc_s.loc[7, 's']  # of the base case
Q_cond = bc_s.loc[7, 'm'] * (bc_s.loc[7, 'h'] - bc_s.loc[8, 'h'])

delta_t_min = 5  # minimum temperature difference [K]
T_amb = 10  # temperature of ambient [°C]
p_amb = 1.013  # pressure of ambient [bar]


# --- ED_EN OF COMPRESSOR -----------------------------------------------------------------

hp_comp = hp_234(['R245fa', 'water'])
hp_settings_234(hp_comp, T_amb, p_amb, delta_t_min, Q_cond*1e3)
hp_comp.solve(mode='design')
hp_comp.results['Connection'].round(5).to_csv("hp_compressor_results.csv")
hp_comp_c = hp_comp.results['Connection']

m2_comp = bc_s.loc[7, 'm'] * (bc_s.loc[8, 's'] - bc_s.loc[7, 's']) / (hp_comp_c.loc['3', 's'] - hp_comp_c.loc['4', 's'])
print("COMPRESSOR:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2_comp, 3), "kg/s and not", round(bc_s.loc[2, 'm'], 3), "kg/s.")

e2_minus_e3 = hp_comp_c.loc['2', 'h'] - hp_comp_c.loc['3', 'h'] - (T_amb+273.15) * (hp_comp_c.loc['2', 's'] - hp_comp_c.loc['3', 's'])
ED_EN_COMP = m2_comp * e2_minus_e3 + m2_comp * (hp_comp_c.loc['3', 'h'] - hp_comp_c.loc['2', 'h'])
ED_COMP = bc_c.loc['compressor', 'E_D']
print("The endogenous exergy destruction in the compressor is", round(ED_EN_COMP, 3), "W, compared to the total value of", round(ED_COMP, 3), "W.")
print("The endogenous exergy destruction in the compressor is", round(ED_EN_COMP/ED_COMP*100, 1), "% of the total exergy destruction in the compressor.")


# --- ED_EN OF THROTTLE  -----------------------------------------------------------------

hp_thr = hp_network(['R245fa', 'water'])
hp_settings_throttle(hp_thr, T_amb, p_amb, delta_t_min, m7)
hp_thr.solve(mode='design')
hp_thr.results['Connection'].round(5).to_csv("hp_throttle_results.csv")
hp_thr_c = hp_thr.results['Connection']

m2_thr = bc_s.loc[7, 'm'] * (bc_s.loc[8, 's'] - bc_s.loc[7, 's']) / (hp_thr_c.loc['3', 's'] - hp_thr_c.loc['4', 's'])
print("THROTTLE:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2, 3), "kg/s and not", round(bc_s.loc[2, 'm'], 3), "kg/s.")

e4_minus_e1 = hp_thr_c.loc['4', 'h'] - hp_thr_c.loc['1', 'h'] - (T_amb+273.15) * (hp_thr_c.loc['4', 's'] - hp_thr_c.loc['1', 's'])
ED_EN_THR = m2_thr * e4_minus_e1
ED_THR = bc_c.loc['expansion valve', 'E_D']
print("The endogenous exergy destruction in the throttle is", round(ED_EN_THR, 3), "compared to the total value of", round(ED_THR, 3), "W.")
print("The endogenous exergy destruction in the throttle is", round(ED_EN_THR/ED_THR*100, 1), "% of the total exergy destruction in the compressor.")


#W_COMP = m2 * (hp_thr.results['Connection'].iloc[2, 2] - hp_thr.results['Connection'].iloc[1, 2])
#W_COMP_base = bc_s.iloc[0,0] * (bc_s.iloc[2,2] - bc_s.iloc[1,2])
#print("For example, the power needed from the compressor is", round(W_COMP, 3), "instead of", round(W_COMP_base, 3), "kW of the base case.")
