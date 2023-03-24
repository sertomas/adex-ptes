import pandas as pd
import CoolProp.CoolProp as cp
from cb_set_pars import hp_settings_throttle, hp_settings_234, hp_settings_compressor, hp_settings_evaporator, \
    hp_settings_condenser
from cb_network import hp_network, hp_234, hp_open
from cb_exergy import exerg_an_hp

# the base case is read from the results of the simulation
bc_s = pd.read_csv("hp_base_case.csv", index_col=0)  # streams of base case
bc_c = pd.read_csv("hp_exan_components.csv", index_col=0)  # components of base case
# example: bc_s.loc[2, 's'] calls the entropy of stream 2

Q_cond = bc_s.loc[7, 'm'] * (bc_s.loc[7, 'h'] - bc_s.loc[8, 'h'])  # negative ==> heat extraction from the heat pump
delta_t_min = 5  # minimum temperature difference [K]
T_amb = 10  # temperature of ambient [°C]
p_amb = 1.013  # pressure of ambient [bar]

# --- ED_EN OF COMPRESSOR -----------------------------------------------------------------

hp_comp = hp_234(['R245fa', 'water'])
hp_settings_234(hp_comp, T_amb, p_amb, delta_t_min, Q_cond * 1e3)
hp_comp.solve(mode='design')
hp_comp.results['Connection'].round(5).to_csv("adex_comp.csv")
hp_comp_c = hp_comp.results['Connection']

m2_comp = bc_s.loc[7, 'm'] * (bc_s.loc[8, 's'] - bc_s.loc[7, 's']) / (hp_comp_c.loc['3', 's'] - hp_comp_c.loc['4', 's'])
print("COMPRESSOR:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2_comp, 3), "kg/s and not",
      round(bc_s.loc[2, 'm'], 3), "kg/s.")

e2_minus_e3 = hp_comp_c.loc['2', 'h'] - hp_comp_c.loc['3', 'h'] - (T_amb + 273.15) * (
            hp_comp_c.loc['2', 's'] - hp_comp_c.loc['3', 's'])
ED_EN_COMP = m2_comp * e2_minus_e3 + m2_comp * (hp_comp_c.loc['3', 'h'] - hp_comp_c.loc['2', 'h'])
ED_COMP = bc_c.loc['compressor', 'E_D']
print("The endogenous exergy destruction in the compressor is", round(ED_EN_COMP, 3),
      "W, compared to the total value of", round(ED_COMP, 3), "W.")
print("The endogenous exergy destruction in the compressor is", round(ED_EN_COMP / ED_COMP * 100, 1),
      "% of the total exergy destruction in the compressor.")

# --- ED_EN OF COMPRESSOR with new method that considers power flows from idealized components ---------------------

hp_comp_alt = hp_open(['R245fa', 'water'])
s4_id = cp.PropsSI('S', 'P', bc_s.loc[4, 'p'] * 1e5, 'T', bc_s.loc[7, 'T'] + 273.15,
                   'R245fa')  # entropy after condenser = entropy after fictive isentropic expander (s4=s1)
h1_id = cp.PropsSI('H', 'P', bc_s.loc[2, 'p'] * 1e5, 'S', s4_id,
                   'R245fa') * 1e-3  # enthalpy after fictive expander (h1=f(s4))
hp_settings_compressor(hp_comp_alt, T_amb, p_amb, delta_t_min, Q_cond * 1e3, h1_id)
hp_comp_alt.solve(mode='design')
hp_comp_alt.results['Connection'].round(5).to_csv("adex_comp_alt.csv")
hp_comp_c = hp_comp_alt.results['Connection']

# h3 and s3 are unknown, the simulation provides only h3_re and s3_re
# h3 and s3 are given as start value using a simple average (50%-50%) between the real and the ideal values
h3_open_id = cp.PropsSI('H', 'P', hp_comp_c.loc['3', 'p'] * 1e5, 'S', hp_comp_c.loc['2', 's'],
                        'R245fa') * 1e-3  # enthalpy after ideal compressor
h3_open = (hp_comp_c.loc['3', 'h'] + h3_open_id) / 2
s3_open = (hp_comp_c.loc['3', 's'] + hp_comp_c.loc['2', 's']) / 2
print("The start value of h3 is", round(h3_open, 3), "kJ/kg, between the value of h3_re (",
      round(hp_comp_c.loc['3', 'h'], 3), "kJ/kg) and h3_id (", round(h3_open_id, 3), "kJ/kg).")

toll = 1e-3
diff = 1
iter_step = 1
while diff > toll:
    m2_open = bc_s.loc[7, 'm'] * (bc_s.loc[8, 's'] - bc_s.loc[7, 's']) / (s3_open - hp_comp_c.loc['4', 's'])
    W_COND_open = m2_open * (hp_comp_c.loc['3', 'h'] - hp_comp_c.loc['4', 'h']) + bc_s.loc[7, 'm'] * (
                bc_s.loc[7, 'h'] - bc_s.loc[8, 'h'])
    W_THR_open = m2_open * (hp_comp_c.loc['4', 'h'] - hp_comp_c.loc['1', 'h'])
    s6_id = cp.PropsSI('S', 'P', p_amb * 1e5, 'T', T_amb - delta_t_min + 273.15,
                       'water')  # entropy of outgoing ambient water
    m5_open = m2_open * (hp_comp_c.loc['2', 's'] - hp_comp_c.loc['1', 's']) / (bc_s.loc[5, 's'] - s6_id)
    h6_id = cp.PropsSI('H', 'P', p_amb * 1e5, 'T', T_amb - delta_t_min + 273.15,
                       'water') * 1e-3  # enthalpy of outgoing ambient water
    W_EVA_open = m2_open * (hp_comp_c.loc['1', 'h'] - hp_comp_c.loc['2', 'h']) + m5_open * (bc_s.loc[5, 'h'] - h6_id)
    W_COMP_id = W_COND_open + W_THR_open + W_EVA_open
    m2_open_id = W_COMP_id / (h3_open_id - hp_comp_c.loc['2', 'h'])
    m2_open_re = m2_open - m2_open_id
    W_COMP_re = m2_open_re * (h3_open - hp_comp_c.loc['2', 'h'])
    ED_EN_COMP = m2_open_re * (hp_comp_c.loc['2', 'h'] - hp_comp_c.loc['3', 'h'] - (T_amb + 273.15) * (
                hp_comp_c.loc['2', 's'] - hp_comp_c.loc['3', 's']))

    # At the end h3 and s3 are calculated as used as new initial value
    # ==> iteration until the difference between the two values is lower than a tolerance value
    s3_open_new = (m2_open_re * hp_comp_c.loc['3', 's'] + m2_open_id * hp_comp_c.loc['2', 's']) / m2_open
    h3_open_new = (m2_open_re * hp_comp_c.loc['3', 'h'] + m2_open_id * hp_comp_c.loc['2', 'h']) / m2_open
    diff = abs(h3_open_new - h3_open)
    print("The iteration was conducted", iter_step,
          "times and the difference of the values of h3 between two iteration steps is", diff)
    iter_step += 1
    h3_open = h3_open_new
    s3_open = s3_open_new

print("COMPRESSOR:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2_open, 3), "kg/s and not",
      round(bc_s.loc[2, 'm'], 3), "kg/s.")
print("The power produced by the heat engine of the idealized condenser is", round(W_COND_open, 3), "kW.")
print("The power produced by the expander of the idealized throttle is", round(W_THR_open, 3), "kW.")
print("Compared to the ideal case, the mass flow of the ambient water is", round(m5_open, 3), "kg/s and not",
      round(bc_s.loc[5, 'm'], 3), "kg/s.")
print("The power produced by the heat engine of the idealized evaporator is", round(W_EVA_open, 3), "kW.")
print("The power produced by the idealized heat exchangers and throttle is", round(W_COMP_id, 3),
      "kW and is equal to the power of the ideal compressor.")
print("The mass flow going through the ideal compressor is equal to", round(m2_open_id, 3),
      "kg/s and is lower than the total mass flow of", round(m2_open, 3), "kg/s.")
print("The mass flow going through the real compressor is equal to", round(m2_open_re, 3),
      "kg/s and is lower than the total mass flow of", round(m2_open, 3), "kg/s.")
print("The power produced by the real compressor is", round(W_COMP_re, 3), "kW.")
print("The endogenous exergy destruction in the compressor is", round(ED_EN_COMP, 3),
      "W, compared to the total value of", round(ED_COMP, 3), "W.")
print("The endogenous exergy destruction in the compressor is", round(ED_EN_COMP / ED_COMP * 100, 1),
      "% of the total exergy destruction in the compressor.")

# --- ED_EN OF THROTTLE  -----------------------------------------------------------------

hp_thr = hp_network(['R245fa', 'water'])
hp_settings_throttle(hp_thr, T_amb, p_amb, delta_t_min, m7)
hp_thr.solve(mode='design')
hp_thr.results['Connection'].round(5).to_csv("adex_thr.csv")
hp_thr_c = hp_thr.results['Connection']

m2_thr = bc_s.loc[7, 'm'] * (bc_s.loc[8, 's'] - bc_s.loc[7, 's']) / (hp_thr_c.loc['3', 's'] - hp_thr_c.loc['4', 's'])
print("THROTTLE:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2_thr, 3), "kg/s and not",
      round(bc_s.loc[2, 'm'], 3), "kg/s.")

e4_minus_e1 = hp_thr_c.loc['4', 'h'] - hp_thr_c.loc['1', 'h'] - (T_amb + 273.15) * (
            hp_thr_c.loc['4', 's'] - hp_thr_c.loc['1', 's'])
ED_EN_THR = m2_thr * e4_minus_e1
ED_THR = bc_c.loc['expansion valve', 'E_D']
print("The endogenous exergy destruction in the throttle is", round(ED_EN_THR, 3), "compared to the total value of",
      round(ED_THR, 3), "W.")
print("The endogenous exergy destruction in the throttle is", round(ED_EN_THR / ED_THR * 100, 1),
      "% of the total exergy destruction in the throttle.")

# --- ED_EN OF EVAPORATOR  -----------------------------------------------------------------

hp_eva = hp_open(['R245fa', 'water'])
h1_id = cp.PropsSI('H', 'P', bc_s.loc[1, 'p'] * 1e5, 'S', s4_id,
                   'R245fa') * 1e-3  # enthalpy after fictive expander (h1=f(s4))
hp_settings_evaporator(hp_eva, T_amb, p_amb, delta_t_min, Q_cond * 1e3, h1_id)
hp_eva.solve(mode='design')
hp_eva.results['Connection'].round(5).to_csv("adex_eva.csv")
hp_eva_c = hp_eva.results['Connection']

m2_open = bc_s.loc[7, 'm'] * (bc_s.loc[8, 's'] - bc_s.loc[7, 's']) / (hp_eva_c.loc['3', 's'] - hp_eva_c.loc['4', 's'])
print("EVAPORATOR:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2_open, 3), "kg/s and not",
      round(bc_s.loc[2, 'm'], 3), "kg/s.")
m5_open = m2_open * (hp_eva_c.loc['2', 'h'] - hp_eva_c.loc['1', 'h']) / (bc_s.loc[5, 'h'] - bc_s.loc[6, 'h'])
ED_EN_EVA = m2_open * (hp_eva_c.loc['1', 'h'] - hp_eva_c.loc['2', 'h'] - (T_amb + 273.15) * (
            hp_eva_c.loc['1', 's'] - hp_eva_c.loc['2', 's'])) + m5_open * (
                        bc_s.loc[5, 'h'] - bc_s.loc[6, 'h'] - (T_amb + 273.15) * (bc_s.loc[5, 's'] - bc_s.loc[6, 's']))
ED_EVA = bc_c.loc['evaporator hp', 'E_D']
print("The endogenous exergy destruction in the evaporator is", round(ED_EN_EVA, 3),
      "W, compared to the total value of", round(ED_EVA, 3), "W.")
print("The endogenous exergy destruction in the evaporator is", round(ED_EN_EVA / ED_EVA * 100, 1),
      "% of the total exergy destruction in the evaporator.")

# --- ED_EN OF CONDENSER  -----------------------------------------------------------------

hp_cond = hp_open(['R245fa', 'water'])
h1_id = cp.PropsSI('H', 'P', bc_s.loc[1, 'p'] * 1e5, 'S', s4_id, 'R245fa') * 1e-3  # enthalpy after fictive expander (h1=f(s4))
hp_settings_condenser(hp_cond, T_amb, p_amb, delta_t_min, Q_cond*1e3, h1_id)
hp_cond.solve(mode='design')
hp_cond.results['Connection'].round(5).to_csv("adex_cond.csv")
hp_cond_c = hp_cond.results['Connection']

m2_open = bc_s.loc[7, 'm'] * (bc_s.loc[7, 'h'] - bc_s.loc[8, 'h']) / (hp_cond_c.loc['4', 'h'] - hp_cond_c.loc['3', 'h'])
print("CONDENSER:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2_open, 3), "kg/s and not",
      round(bc_s.loc[2, 'm'], 3), "kg/s.")
ED_EN_COND = m2_open * (hp_cond_c.loc['3', 'h'] - hp_cond_c.loc['4', 'h'] - (T_amb + 273.15) * (
            hp_cond_c.loc['3', 's'] - hp_cond_c.loc['4', 's'])) + bc_s.loc[7, 'm'] * (
            bc_s.loc[7, 'h'] - bc_s.loc[8, 'h'] - (T_amb + 273.15) * (bc_s.loc[7, 's'] - bc_s.loc[8, 's']))
ED_COND = bc_c.loc['condenser hp', 'E_D']
print("The endogenous exergy destruction in the condenser is", round(ED_EN_COND, 3),
      "W, compared to the total value of", round(ED_COND, 3), "W.")
print("The endogenous exergy destruction in the condenser is", round(ED_EN_COND / ED_COND * 100, 1),
      "% of the total exergy destruction in the condenser.")
