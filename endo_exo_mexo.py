import CoolProp.CoolProp as cp


def endo_comp(hp_comp_s, bc_s, bc_c, p_amb, T_amb, delta_t_min, COND=None, details=None, ED_EN_COMP=None, ED_EN_COND=None):

    # h3 and s3 are unknown, the simulation provides only h3_re and s3_re
    # h3 and s3 are given as start value using a simple average (50%-50%) between the real and the ideal values
    h3_id = cp.PropsSI('H', 'P', hp_comp_s.loc['3', 'p'] * 1e5, 'S', hp_comp_s.loc['2', 's'],'R245fa') * 1e-3  # enthalpy after ideal compressor
    h3 = (hp_comp_s.loc['3', 'h'] + h3_id) / 2
    s3 = (hp_comp_s.loc['3', 's'] + hp_comp_s.loc['2', 's']) / 2
    print("The start value of h3 is", round(h3, 3), "kJ/kg, between the value of h3_re (", round(hp_comp_s.loc['3', 'h'], 3), "kJ/kg) and h3_id (", round(h3_id, 3), "kJ/kg).")

    toll = 1e-3
    diff = 1
    iter_step = 1
    while diff > toll:
        m2_comp = bc_s.loc[7, 'm'] * (bc_s.loc[8, 's'] - bc_s.loc[7, 's']) / (s3 - hp_comp_s.loc['4', 's'])
        if COND is None: W_COND = m2_comp * (hp_comp_s.loc['3', 'h'] - hp_comp_s.loc['4', 'h']) + bc_s.loc[7, 'm'] * (bc_s.loc[7, 'h'] - bc_s.loc[8, 'h'])
        else: ED_COND_comp_cond = m2_comp * (hp_comp_s.loc['3', 'h'] - hp_comp_s.loc['4', 'h']) + bc_s.loc[7, 'm'] * (bc_s.loc[7, 'h'] - bc_s.loc[8, 'h'])
        W_THR = m2_comp * (hp_comp_s.loc['4', 'h'] - hp_comp_s.loc['1', 'h'])
        s6_id = cp.PropsSI('S', 'P', p_amb * 1e5, 'T', T_amb - delta_t_min + 273.15,'water')  # entropy of outgoing ambient water
        m5_comp = m2_comp * (hp_comp_s.loc['2', 's'] - hp_comp_s.loc['1', 's']) / (bc_s.loc[5, 's'] - s6_id)
        h6_id = cp.PropsSI('H', 'P', p_amb * 1e5, 'T', T_amb - delta_t_min + 273.15,'water') * 1e-3  # enthalpy of outgoing ambient water
        W_EVA = m2_comp * (hp_comp_s.loc['1', 'h'] - hp_comp_s.loc['2', 'h']) + m5_comp * (bc_s.loc[5, 'h'] - h6_id)
        if COND is None: W_COMP_id = W_COND + W_THR + W_EVA
        else: W_COMP_id = W_THR + W_EVA
        m2_comp_id = W_COMP_id / (h3_id - hp_comp_s.loc['2', 'h'])
        m2_comp_re = m2_comp - m2_comp_id
        W_COMP_re = m2_comp_re * (h3 - hp_comp_s.loc['2', 'h'])
        if COND is None: ED_EN_COMP = m2_comp_re * (hp_comp_s.loc['2', 'h'] - hp_comp_s.loc['3', 'h'] - (T_amb + 273.15) * (hp_comp_s.loc['2', 's'] - hp_comp_s.loc['3', 's']))
        else: ED_COMP_comp_cond = m2_comp_re * (hp_comp_s.loc['2', 'h'] - hp_comp_s.loc['3', 'h'] - (T_amb + 273.15) * (hp_comp_s.loc['2', 's'] - hp_comp_s.loc['3', 's']))

        # At the end h3 and s3 are calculated as used as new initial value
        # ==> iteration until the difference between the two values is lower than a tolerance value
        s3_new = (m2_comp_re * hp_comp_s.loc['3', 's'] + m2_comp_id * hp_comp_s.loc['2', 's']) / m2_comp
        h3_new = (m2_comp_re * hp_comp_s.loc['3', 'h'] + m2_comp_id * hp_comp_s.loc['2', 'h']) / m2_comp
        diff = abs(h3_new - h3)
        print("The iteration was conducted", iter_step, "times and the difference of the values of h3 between two iteration steps is", diff)
        iter_step += 1
        h3 = h3_new
        s3 = s3_new

    ED_COMP = bc_c.loc['compressor', 'E_D']
    ED_COND = bc_c.loc['condenser hp', 'E_D']

    if COND is None:
        print("COMPRESSOR:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2_comp, 3), "kg/s and not", round(bc_s.loc[2, 'm'], 3), "kg/s.")
        print("The endogenous exergy destruction in the compressor is", round(ED_EN_COMP, 3), "W, compared to the total value of", round(ED_COMP, 3), "W.")
        print("The endogenous exergy destruction in the compressor is", round(ED_EN_COMP / ED_COMP * 100, 1), "% of the total exergy destruction in the compressor.")
        if details:
            print("The power produced by the heat engine of the idealized condenser is", round(W_COND, 3), "kW.")
            print("The power produced by the expander of the idealized throttle is", round(W_THR, 3), "kW.")
            print("Compared to the ideal case, the mass flow of the ambient water is", round(m5_comp, 3), "kg/s and not", round(bc_s.loc[5, 'm'], 3), "kg/s.")
            print("The power produced by the heat engine of the idealized evaporator is", round(W_EVA, 3), "kW.")
            print("The power produced by the idealized heat exchangers and throttle is", round(W_COMP_id, 3), "kW and is equal to the power of the ideal compressor.")
            print("The mass flow going through the ideal compressor is equal to", round(m2_comp_id, 3), "kg/s and is lower than the total mass flow of", round(m2_comp, 3), "kg/s.")
            print("The mass flow going through the real compressor is equal to", round(m2_comp_re, 3),  "kg/s and is lower than the total mass flow of", round(m2_comp, 3), "kg/s.")
            print("The power produced by the real compressor is", round(W_COMP_re, 3), "kW.")
    else:
        print("COMPRESSOR-CONDESER:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2_comp, 3), "kg/s and not", round(bc_s.loc[2, 'm'], 3), "kg/s.")
        print("When the compressor and the condenser are real, the exergy destruction in the compressor is", round(ED_EN_COMP, 3), "W, compared to the total value of", round(ED_COMP, 3), "W.")
        print("When the compressor and the condenser are real, the exergy destruction in the condenser is", round(ED_EN_COND, 3), "W, compared to the total value of", round(ED_COND, 3), "W.")
        print("The endogenous exergy destruction in the compressor was", round(ED_EN_COMP, 3), "W.")
        print("The endogenous exergy destruction in the condenser was", round(ED_EN_COND, 3), "W.")
        print("The exogenous exergy destruction in the compressor caused by the interaction with the condenser is", round(ED_COMP_comp_cond - ED_EN_COMP, 3), "W.")
        print("The exogenous exergy destruction in the condenser caused by the interaction with the compressor is", round(ED_COND_comp_cond - ED_EN_COND, 3), "W.")

    if COND is None:
        m2_comp, ED_EN_COMP
    else:
        return m2_comp, ED_COMP_comp_cond, ED_COND_comp_cond


def endo_thr(hp_thr_s, bc_s, bc_c, T_amb):

    m2_thr = bc_s.loc[7, 'm'] * (bc_s.loc[8, 's'] - bc_s.loc[7, 's']) / (hp_thr_s.loc['3', 's'] - hp_thr_s.loc['4', 's'])
    e4_minus_e1 = hp_thr_s.loc['4', 'h'] - hp_thr_s.loc['1', 'h'] - (T_amb + 273.15) * (hp_thr_s.loc['4', 's'] - hp_thr_s.loc['1', 's'])
    ED_EN_THR = m2_thr * e4_minus_e1
    ED_THR = bc_c.loc['expansion valve', 'E_D']
    print("THROTTLE:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2_thr, 3), "kg/s and not", round(bc_s.loc[2, 'm'], 3), "kg/s.")
    print("The endogenous exergy destruction in the throttle is", round(ED_EN_THR, 3), "compared to the total value of", round(ED_THR, 3), "W.")
    print("The endogenous exergy destruction in the throttle is", round(ED_EN_THR / ED_THR * 100, 1), "% of the total exergy destruction in the throttle.")

    return m2_thr, ED_EN_THR


def endo_eva(hp_eva_s, bc_s, bc_c, T_amb):

    m2_eva = bc_s.loc[7, 'm'] * (bc_s.loc[8, 's'] - bc_s.loc[7, 's']) / (hp_eva_s.loc['3', 's'] - hp_eva_s.loc['4', 's'])
    print("EVAPORATOR:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2_eva, 3), "kg/s and not",round(bc_s.loc[2, 'm'], 3), "kg/s.")
    m5_eva = m2_eva * (hp_eva_s.loc['2', 'h'] - hp_eva_s.loc['1', 'h']) / (bc_s.loc[5, 'h'] - bc_s.loc[6, 'h'])
    ED_EN_EVA = m2_eva * (hp_eva_s.loc['1', 'h'] - hp_eva_s.loc['2', 'h'] - (T_amb + 273.15) * (hp_eva_s.loc['1', 's'] - hp_eva_s.loc['2', 's'])) + m5_eva * (bc_s.loc[5, 'h'] - bc_s.loc[6, 'h'] - (T_amb + 273.15) * (bc_s.loc[5, 's'] - bc_s.loc[6, 's']))
    ED_EVA = bc_c.loc['evaporator hp', 'E_D']
    print("The endogenous exergy destruction in the evaporator is", round(ED_EN_EVA, 3), "W, compared to the total value of", round(ED_EVA, 3), "W.")
    print("The endogenous exergy destruction in the evaporator is", round(ED_EN_EVA / ED_EVA * 100, 1), "% of the total exergy destruction in the evaporator.")

    return m2_eva, ED_EN_EVA

def endo_cond(hp_cond_s, bc_s, bc_c, T_amb):

    m2_cond = bc_s.loc[7, 'm'] * (bc_s.loc[7, 'h'] - bc_s.loc[8, 'h']) / ( hp_cond_s.loc['4', 'h'] - hp_cond_s.loc['3', 'h'])
    print("CONDENSER:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2_cond, 3), "kg/s and not", round(bc_s.loc[2, 'm'], 3), "kg/s.")
    ED_EN_COND = m2_cond * (hp_cond_s.loc['3', 'h'] - hp_cond_s.loc['4', 'h'] - (T_amb + 273.15) * (hp_cond_s.loc['3', 's'] - hp_cond_s.loc['4', 's'])) + bc_s.loc[7, 'm'] * (bc_s.loc[7, 'h'] - bc_s.loc[8, 'h'] - (T_amb + 273.15) * (bc_s.loc[7, 's'] - bc_s.loc[8, 's']))
    ED_COND = bc_c.loc['condenser hp', 'E_D']
    print("The endogenous exergy destruction in the condenser is", round(ED_EN_COND, 3), "W, compared to the total value of", round(ED_COND, 3), "W.")
    print("The endogenous exergy destruction in the condenser is", round(ED_EN_COND / ED_COND * 100, 1), "% of the total exergy destruction in the condenser.")

    return m2_cond, ED_EN_COND