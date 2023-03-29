import pandas as pd
import CoolProp.CoolProp as cp
from cb_set_pars import hp_settings_throttle, hp_settings_ideal, hp_settings_open
from cb_network import hp_network, hp_open
from endo_exo_mexo import endo_comp, endo_thr, endo_eva, endo_cond
from config import delta_t_min, T_amb, p_amb


# the base case is read from the results of the simulation
bc_s = pd.read_csv("hp_base_case.csv", index_col=0)  # streams of base case
bc_c = pd.read_csv("hp_exan_components.csv", index_col=0)  # components of base case
# example: bc_s.loc[2, 's'] calls the entropy of stream 2

# the product of the system is the heat flow to the TES, which is kept constant
Q_cond = bc_s.loc[7, 'm'] * (bc_s.loc[7, 'h'] - bc_s.loc[8, 'h'])  # negative ==> heat extraction from the heat pump
m7 = bc_s.loc[7, 'm']  # [kg/s]
delta_t_min = 5  # minimum temperature difference [K]
T_amb = 20  # [°C] temperature of ambient
p_amb = 1.013  # [bar] pressure of ambient

# because the cycle of the HP is sometimes open, the enthalpy and entropy of the exit stream must be defined according to cases
s4_id = cp.PropsSI('S', 'P', bc_s.loc[4, 'p'] * 1e5, 'T', bc_s.loc[7, 'T'] + 273.15, 'R245fa') * 1e-3  # [kJ/kgK] s4=s1 (isentropic expansion) in fully ideal system
s4_id_delta_T = cp.PropsSI('S', 'P', bc_s.loc[4, 'p'] * 1e5, 'T', bc_s.loc[7, 'T'] + delta_t_min + 273.15, 'R245fa') * 1e-3  # [kJ/kgK] s4=s1 (isentropic expansion) with minimal temperature difference in condenser
h1_id = cp.PropsSI('H', 'P', bc_s.loc[2, 'p'] * 1e5, 'S', s4_id * 1e3, 'R245fa') * 1e-3  # [kJ/kg] enthalpy after expansion (h1=f(s4,p1)) in fully ideal system
h1_id_delta_T = cp.PropsSI('H', 'P', bc_s.loc[2, 'p'] * 1e5, 'S', s4_id_delta_T * 1e3, 'R245fa') * 1e-3  # [kJ/kg] enthalpy after expansion (h1=f(s4,p1)) with minimal temperature difference in condenser
h1_id_p_loss = cp.PropsSI('H', 'P', bc_s.loc[1, 'p'] * 1e5, 'S', s4_id * 1e3, 'R245fa') * 1e-3  # [kJ/kg] enthalpy after expansion (h1=f(s4,p1)) with pressure drops in expander
h1_id_thr = cp.PropsSI('H', 'P', bc_s.loc[4, 'p'] * 1e5, 'S', s4_id * 1e3, 'R245fa') * 1e-3  # [kJ/kg] enthalpy before and after the throttle is the same
h1_id_delta_T_p_loss = cp.PropsSI('H', 'P', bc_s.loc[1, 'p'] * 1e5, 'S', s4_id_delta_T * 1e3, 'R245fa') * 1e-3  # [kJ/kg] enthalpy after expansion (h1=f(s4,p1)) with pressure drops in expander and minimal temperature difference in condenser


# ideal process
hp_ideal = hp_open(['R245fa', 'water'])  # open cycle is used because TESPy Turbine does not work with liquids
hp_settings_ideal(hp_ideal, Q_cond*1e3, h1_id)
hp_ideal.solve(mode='design')
hp_ideal.results['Connection'].round(5).to_csv("adex_ideal.csv")
hp_ideal_s = hp_ideal.results['Connection']

m2_id = bc_s.loc[7, 'm'] * (bc_s.loc[8, 's'] - bc_s.loc[7, 's']) / (hp_ideal_s.loc['3', 's'] - hp_ideal_s.loc['4', 's'])  # [kg/s]
print('In the ideal case, the mass flow is equal to', round(m2_id, 3), 'kg/s.')

# --- ED_EN OF COMPRESSOR ----------------------------------------------------------------------------------------------

hp_comp = hp_open(['R245fa', 'water'])
hp_settings_open(hp_comp, Q_cond*1e3, h1_id, COMP=True)
hp_comp.solve(mode='design')
hp_comp.results['Connection'].round(5).to_csv("adex_comp.csv")
hp_comp_s = hp_comp.results['Connection']

m2_comp, ED_EN_COMP = endo_comp(hp_comp_s, bc_s, bc_c, details=True)


# --- ED_EN OF THROTTLE  -----------------------------------------------------------------------------------------------

hp_thr = hp_network(['R245fa', 'water'])
hp_settings_throttle(hp_thr, m7)
hp_thr.solve(mode='design')
hp_thr.results['Connection'].round(5).to_csv("adex_thr.csv")
hp_thr_s = hp_thr.results['Connection']

m2_thr, ED_EN_THR = endo_thr(hp_thr_s, bc_s, bc_c)


# --- ED_EN OF EVAPORATOR  ---------------------------------------------------------------------------------------------

hp_eva = hp_open(['R245fa', 'water'])
hp_settings_open(hp_eva, Q_cond*1e3, h1_id_p_loss, EVA=True)
hp_eva.solve(mode='design')
hp_eva.results['Connection'].round(5).to_csv("adex_eva.csv")
hp_eva_s = hp_eva.results['Connection']

m2_eva, ED_EN_EVA = endo_eva(hp_eva_s, bc_s, bc_c)


# --- ED_EN OF CONDENSER  ----------------------------------------------------------------------------------------------

hp_cond = hp_open(['R245fa', 'water'])
hp_settings_open(hp_cond, Q_cond*1e3, h1_id_delta_T, COND=True)
hp_cond.solve(mode='design')
hp_cond.results['Connection'].round(5).to_csv("adex_cond.csv")
hp_cond_s = hp_cond.results['Connection']

m2_cond, ED_EN_COND = endo_cond(hp_cond_s, bc_s, bc_c)


# --- RESULTS OF ED_EN -------------------------------------------------------------------------------------------------

ED_EN = [ED_EN_COND, ED_EN_THR, ED_EN_COMP, ED_EN_EVA]
m2_vector = [m2_cond, m2_thr, m2_comp, m2_eva]  # [kg/s]

df_endog_ed = pd.DataFrame(index=bc_c.index)
df_endog_ed["m2 [kg/s]"] = m2_vector
df_endog_ed["ED [kW]"] = bc_c["E_D"] * 1e-3
df_endog_ed["epsilon"] = bc_c["epsilon"]
df_endog_ed["ED_EN [kW]"] = ED_EN
df_endog_ed["ED_EX [kW]"] = df_endog_ed["ED [kW]"] - df_endog_ed["ED_EN [kW]"]
df_endog_ed["ED_EN share [-]"] = df_endog_ed["ED_EN [kW]"] / df_endog_ed["ED [kW]"]
df_endog_ed.round(3).to_csv("hp_endog.csv")


# --- ED_MEXO OF COMPRESSOR AND CONDENSER ------------------------------------------------------------------------------

hp_comp_cond = hp_open(['R245fa', 'water'])
hp_settings_open(hp_comp_cond, Q_cond*1e3, h1_id_delta_T, COMP=True, COND=True)
hp_comp_cond.solve(mode='design')
hp_comp_cond.results['Connection'].round(5).to_csv("adex_comp_cond.csv")
hp_comp_cond_s = hp_comp_cond.results['Connection']

m2_comp_cond, ED_COMP_comp_cond, ED_COND_comp_cond = endo_comp(hp_comp_cond_s, bc_s, bc_c, COND=True, ED_EN_COMP=ED_EN_COMP, ED_EN_COND=ED_EN_COND)


# --- ED_MEXO OF COMPRESSOR AND EVAPORATOR -----------------------------------------------------------------------------

hp_comp_eva = hp_open(['R245fa', 'water'])
hp_settings_open(hp_comp_eva, Q_cond * 1e3, h1_id_p_loss, COMP=True, EVA=True)
hp_comp_eva.solve(mode='design')
hp_comp_eva.results['Connection'].round(5).to_csv("adex_comp_eva.csv")
hp_comp_eva_s = hp_comp_eva.results['Connection']

m2_comp_eva, ED_COMP_comp_eva, ED_EVA_comp_eva = endo_comp(hp_comp_eva_s, bc_s, bc_c, EVA=True, ED_EN_COMP=ED_EN_COMP, ED_EN_EVA=ED_EN_EVA)


# --- ED_MEXO OF COMPRESSOR AND THROTTLE -----------------------------------------------------------------------------

hp_comp_thr = hp_open(['R245fa', 'water'])
hp_settings_open(hp_comp_thr, Q_cond * 1e3, h1_id_thr, COMP=True)
hp_comp_thr.solve(mode='design')
hp_comp_thr.results['Connection'].round(5).to_csv("adex_comp_thr.csv")
hp_comp_thr_s = hp_comp_thr.results['Connection']

m2_comp_thr, ED_COMP_comp_thr, ED_THR_comp_thr = endo_comp(hp_comp_thr_s, bc_s, bc_c, THR=True, ED_EN_COMP=ED_EN_COMP, ED_EN_THR=ED_EN_THR)


# --- ED_MEXO OF EVAPORATOR AND CONDENSER ------------------------------------------------------------------------------

hp_eva_cond = hp_open(['R245fa', 'water'])
hp_settings_open(hp_eva_cond, Q_cond*1e3, h1_id_delta_T_p_loss, EVA=True, COND=True)
hp_eva_cond.solve(mode='design')
hp_eva_cond.results['Connection'].round(5).to_csv("adex_eva_cond.csv")
hp_eva_cond_s = hp_eva_cond.results['Connection']

m2_eva_cond, ED_EVA_eva_cond, ED_COND_eva_cond = endo_eva(hp_eva_cond_s, bc_s, bc_c, COND=True, ED_EN_EVA=ED_EN_EVA, ED_EN_COND=ED_EN_COND)


# --- ED_MEXO OF EVAPORATOR AND CONDENSER ------------------------------------------------------------------------------

hp_eva_thr = hp_open(['R245fa', 'water'])
hp_settings_open(hp_eva_thr, Q_cond*1e3, h1_id_thr, EVA=True)
hp_eva_thr.solve(mode='design')
hp_eva_thr.results['Connection'].round(5).to_csv("adex_eva_thr.csv")
hp_eva_thr_s = hp_eva_thr.results['Connection']

m2_eva_thr, ED_EVA_eva_thr, ED_THR_eva_thr = endo_eva(hp_eva_cond_s, bc_s, bc_c, THR=True, ED_EN_EVA=ED_EN_EVA, ED_EN_THR=ED_EN_THR)


# --- ED_MEXO OF CONDENSER AND THROTTLE ------------------------------------------------------------------------------

hp_cond_thr = hp_open(['R245fa', 'water'])
hp_settings_open(hp_cond_thr, Q_cond*1e3, bc_s.loc[1, 'h'], COND=True)
hp_cond_thr.solve(mode='design')
hp_cond_thr.results['Connection'].round(5).to_csv("adex_eva_thr.csv")
hp_cond_thr_s = hp_cond_thr.results['Connection']

m2_cond_thr, ED_COND_cond_thr, ED_THR_cond_thr = endo_cond(hp_cond_thr_s, bc_s, bc_c, THR=True, ED_EN_COND=ED_EN_COND, ED_EN_THR=ED_EN_THR)


# --- RESULTS OF ED_MEXO -----------------------------------------------------------------------------------------------

ED_MEXO_COMP = bc_c.loc['compressor', 'E_D'] * 1e-3 - ED_EN_COMP - ((ED_COMP_comp_cond + ED_COMP_comp_eva + ED_COMP_comp_thr) - 3 * ED_EN_COMP)
ED_MEXO_EVA = bc_c.loc['evaporator hp', 'E_D'] * 1e-3 - ED_EN_EVA - ((ED_EVA_eva_thr + ED_EVA_comp_eva + ED_EVA_eva_cond) - 3 * ED_EN_EVA)
ED_MEXO_COND = bc_c.loc['condenser hp', 'E_D'] * 1e-3 - ED_EN_COND - ((ED_COND_comp_cond + ED_COND_eva_cond + ED_COND_cond_thr) - 3 * ED_EN_COND)
ED_MEXO_THR = bc_c.loc['expansion valve', 'E_D'] * 1e-3 - ED_EN_THR - ((ED_THR_eva_thr + ED_THR_comp_thr + ED_THR_cond_thr) - 3 * ED_EN_THR)

# The following data frame must be created in a more efficient way
df_mexo_ed = pd.DataFrame(index=pd.MultiIndex.from_tuples([('COMP', ' '), ('COMP', 'COND'), ('COMP', 'EVA'), ('COMP', 'THR'), ('EVA', ' '), ('EVA', 'COND'), ('EVA', 'COMP'), ('EVA', 'THR')]))
df_mexo_ed.loc[('COMP', ' '), 'ED [kW]'] = bc_c.loc['compressor', 'E_D'] * 1e-3
df_mexo_ed.loc[('COMP', ' '), 'ED_EN [kW]'] = ED_EN_COMP
df_mexo_ed.loc[('COMP', ' '), 'ED_EX [kW]'] = bc_c.loc['compressor', 'E_D'] * 1e-3 - ED_EN_COMP
df_mexo_ed.loc[('COMP', ' '), 'm2 [kg/s]'] = m2_comp
df_mexo_ed.loc[('COMP', 'COND'), 'ED_kl [kW]'] = ED_COMP_comp_cond
df_mexo_ed.loc[('COMP', 'COND'), 'ED_EX_l [kW]'] = ED_COMP_comp_cond - ED_EN_COMP
df_mexo_ed.loc[('COMP', 'COND'), 'm2 [kg/s]'] = m2_comp_cond
df_mexo_ed.loc[('COMP', 'EVA'), 'ED_EX_l [kW]'] = ED_COMP_comp_eva - ED_EN_COMP
df_mexo_ed.loc[('COMP', 'EVA'), 'ED_kl [kW]'] = ED_COMP_comp_eva
df_mexo_ed.loc[('COMP', 'EVA'), 'm2 [kg/s]'] = m2_comp_eva
df_mexo_ed.loc[('COMP', 'THR'), 'ED_EX_l [kW]'] = ED_COMP_comp_thr - ED_EN_COMP
df_mexo_ed.loc[('COMP', 'THR'), 'ED_kl [kW]'] = ED_COMP_comp_thr
df_mexo_ed.loc[('COMP', 'THR'), 'm2 [kg/s]'] = m2_comp_thr
df_mexo_ed.loc[('COMP', ' '), 'ED_MEXO [kW]'] = ED_MEXO_COMP
df_mexo_ed.loc[('EVA', ' '), 'ED [kW]'] = bc_c.loc['evaporator hp', 'E_D'] * 1e-3
df_mexo_ed.loc[('EVA', ' '), 'ED_EN [kW]'] = ED_EN_EVA
df_mexo_ed.loc[('EVA', ' '), 'ED_EX [kW]'] = bc_c.loc['evaporator hp', 'E_D'] * 1e-3 - ED_EN_EVA
df_mexo_ed.loc[('EVA', ' '), 'm2 [kg/s]'] = m2_eva
df_mexo_ed.loc[('EVA', 'COMP'), 'ED_EX_l [kW]'] = ED_EVA_comp_eva - ED_EN_EVA
df_mexo_ed.loc[('EVA', 'COMP'), 'ED_kl [kW]'] = ED_EVA_comp_eva
df_mexo_ed.loc[('EVA', 'COMP'), 'm2 [kg/s]'] = m2_comp_eva
df_mexo_ed.loc[('EVA', 'COND'), 'ED_EX_l [kW]'] = ED_EVA_eva_cond - ED_EN_EVA
df_mexo_ed.loc[('EVA', 'COND'), 'ED_kl [kW]'] = ED_EVA_eva_cond
df_mexo_ed.loc[('EVA', 'COND'), 'm2 [kg/s]'] = m2_eva_cond
df_mexo_ed.loc[('EVA', 'THR'), 'ED_EX_l [kW]'] = ED_EVA_eva_thr - ED_EN_EVA
df_mexo_ed.loc[('EVA', 'THR'), 'ED_kl [kW]'] = ED_EVA_eva_thr
df_mexo_ed.loc[('EVA', 'THR'), 'm2 [kg/s]'] = m2_eva_thr
df_mexo_ed.loc[('EVA', ' '), 'ED_MEXO [kW]'] = ED_MEXO_EVA
df_mexo_ed.loc[('COND', ' '), 'ED [kW]'] = bc_c.loc['condenser hp', 'E_D'] * 1e-3
df_mexo_ed.loc[('COND', ' '), 'ED_EN [kW]'] = ED_EN_COND
df_mexo_ed.loc[('COND', ' '), 'ED_EX [kW]'] = bc_c.loc['condenser hp', 'E_D'] * 1e-3 - ED_EN_COND
df_mexo_ed.loc[('COND', ' '), 'm2 [kg/s]'] = m2_cond
df_mexo_ed.loc[('COND', 'COMP'), 'ED_EX_l [kW]'] = ED_COND_comp_cond - ED_EN_COND
df_mexo_ed.loc[('COND', 'COMP'), 'ED_kl [kW]'] = ED_COND_comp_cond
df_mexo_ed.loc[('COND', 'COMP'), 'm2 [kg/s]'] = m2_comp_cond
df_mexo_ed.loc[('COND', 'EVA'), 'ED_EX_l [kW]'] = ED_COND_eva_cond - ED_EN_COND
df_mexo_ed.loc[('COND', 'EVA'), 'ED_kl [kW]'] = ED_COND_eva_cond
df_mexo_ed.loc[('COND', 'EVA'), 'm2 [kg/s]'] = m2_eva_cond
df_mexo_ed.loc[('COND', 'THR'), 'ED_EX_l [kW]'] = ED_COND_cond_thr - ED_EN_COND
df_mexo_ed.loc[('COND', 'THR'), 'ED_kl [kW]'] = ED_COND_cond_thr
df_mexo_ed.loc[('COND', 'THR'), 'm2 [kg/s]'] = m2_cond_thr
df_mexo_ed.loc[('COND', ' '), 'ED_MEXO [kW]'] = ED_MEXO_COND
df_mexo_ed.loc[('THR', ' '), 'ED [kW]'] = bc_c.loc['expansion valve', 'E_D'] * 1e-3
df_mexo_ed.loc[('THR', ' '), 'ED_EN [kW]'] = ED_EN_THR
df_mexo_ed.loc[('THR', ' '), 'ED_EX [kW]'] = bc_c.loc['expansion valve', 'E_D'] * 1e-3 - ED_EN_THR
df_mexo_ed.loc[('THR', ' '), 'm2 [kg/s]'] = m2_thr
df_mexo_ed.loc[('THR', 'COMP'), 'ED_EX_l [kW]'] = ED_THR_comp_thr - ED_EN_THR
df_mexo_ed.loc[('THR', 'COMP'), 'ED_kl [kW]'] = ED_THR_comp_thr
df_mexo_ed.loc[('THR', 'COMP'), 'm2 [kg/s]'] = m2_comp_thr
df_mexo_ed.loc[('THR', 'EVA'), 'ED_EX_l [kW]'] = ED_THR_eva_thr - ED_EN_THR
df_mexo_ed.loc[('THR', 'EVA'), 'ED_kl [kW]'] = ED_THR_eva_thr
df_mexo_ed.loc[('THR', 'EVA'), 'm2 [kg/s]'] = m2_eva_thr
df_mexo_ed.loc[('THR', 'COND'), 'ED_EX_l [kW]'] = ED_THR_cond_thr - ED_EN_THR
df_mexo_ed.loc[('THR', 'COND'), 'ED_kl [kW]'] = ED_THR_cond_thr
df_mexo_ed.loc[('THR', 'COND'), 'm2 [kg/s]'] = m2_cond_thr
df_mexo_ed.loc[('THR', ' '), 'ED_MEXO [kW]'] = ED_MEXO_THR
df_mexo_ed['ED/m2 [kJ/kg]'] = df_mexo_ed['ED [kW]'] / bc_s.loc[1, 'm']
df_mexo_ed['ED_ED/m2 [kJ/kg]'] = df_mexo_ed['ED_EN [kW]'] / df_mexo_ed['m2 [kg/s]']
df_mexo_ed['ED_kl/m2 [kJ/kg]'] = df_mexo_ed['ED_kl [kW]'] / df_mexo_ed['m2 [kg/s]']
df_mexo_ed.round(2).to_csv("hp_mexo.csv")



