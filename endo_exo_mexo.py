import CoolProp.CoolProp as cp


def endo_comp(hp_comp_s, bc_s, bc_c, T_amb, COND=None, EVA=None, THR=None, details=None, ED_EN_COMP=None, ED_EN_COND=None, ED_EN_EVA=None, ED_EN_THR=None):

    # h3 and s3 are unknown, the simulation provides only h3_re and s3_re
    # h3 and s3 are given as start value using a simple average (50%-50%) between the real and the ideal values
    h3_id = cp.PropsSI('H', 'P', hp_comp_s.loc['3', 'p'] * 1e5, 'S', hp_comp_s.loc['2', 's'] * 1e3, 'R245fa') * 1e-3  # [kJ/kg] enthalpy after ideal compressor
    h3 = (hp_comp_s.loc['3', 'h'] + h3_id) / 2  # [kJ/kg]
    s3 = (hp_comp_s.loc['3', 's'] + hp_comp_s.loc['2', 's']) / 2  # [kJ/kgK]
    print("The start value of h3 is", round(h3, 3), "kJ/kg, between the value of h3_re (", round(hp_comp_s.loc['3', 'h'], 3), "kJ/kg) and h3_id (", round(h3_id, 3), "kJ/kg).")

    toll = 1e-3
    diff = 1
    iter_step = 1
    while diff > toll:
        m2 = bc_s.loc[7, 'm'] * (bc_s.loc[8, 's'] - bc_s.loc[7, 's']) / (s3 - hp_comp_s.loc['4', 's'])  # [kg/s]
        if THR is None:
            W_THR = m2 * (hp_comp_s.loc['4', 'h'] - hp_comp_s.loc['1', 'h'])  # [kW]
        else:
            s4_thr = cp.PropsSI('S', 'P', bc_s.loc[4, 'p'] * 1e5, 'T', bc_s.loc[7, 'T'] + 273.15, 'R245fa') * 1e-3  # [kJ/kgK]
            ED_THR_comp_thr = m2 * (- (T_amb + 273.15) * (s4_thr - hp_comp_s.loc['1', 's']))  # [kW]
        s6_id = cp.PropsSI('S', 'P', bc_s.loc[5, 'p'] * 1e5, 'T', bc_s.loc[6, 'T'] + 273.15, 'water') * 1e-3  # [kJ/kgK] entropy of idealized outgoing ambient water
        h6_id = cp.PropsSI('H', 'P', bc_s.loc[5, 'p'] * 1e5, 'T', bc_s.loc[6, 'T'] + 273.15, 'water') * 1e-3  # [kJ/kg] enthalpy of idealized outgoing ambient water
        s6_re = cp.PropsSI('S', 'P', bc_s.loc[6, 'p'] * 1e5, 'T', bc_s.loc[6, 'T'] + 273.15, 'water') * 1e-3  # [kJ/kgK] entropy of real outgoing ambient water
        h6_re = cp.PropsSI('H', 'P', bc_s.loc[6, 'p'] * 1e5, 'T', bc_s.loc[6, 'T'] + 273.15, 'water') * 1e-3  # [kJ/kg] enthalpy of real outgoing ambient water
        if EVA is None:
            m5 = m2 * (hp_comp_s.loc['2', 's'] - hp_comp_s.loc['1', 's']) / (bc_s.loc[5, 's'] - s6_id)  # [kg/s]
            W_EVA = m2 * (hp_comp_s.loc['1', 'h'] - hp_comp_s.loc['2', 'h']) + m5 * (bc_s.loc[5, 'h'] - h6_re)  # [kW]
        else:
            m5 = m2 * (hp_comp_s.loc['2', 'h'] - hp_comp_s.loc['1', 'h']) / (bc_s.loc[5, 'h'] - h6_id)  # [kg/s]
            ED_EVA_comp_eva = m5 * (bc_s.loc[5, 'h'] - h6_re - (T_amb + 273.15) * (bc_s.loc[5, 's'] - s6_re)) + m2 * (hp_comp_s.loc['1', 'h'] - hp_comp_s.loc['2', 'h'] - (T_amb + 273.15) * (hp_comp_s.loc['1', 's'] - hp_comp_s.loc['2', 's']))  # [kW]
        if COND is None:
            W_COND = m2 * (hp_comp_s.loc['3', 'h'] - hp_comp_s.loc['4', 'h']) + bc_s.loc[7, 'm'] * (bc_s.loc[7, 'h'] - bc_s.loc[8, 'h'])  # [kW]
        else:
            ED_COND_comp_cond = m2 * (hp_comp_s.loc['3', 'h'] - hp_comp_s.loc['4', 'h']) + bc_s.loc[7, 'm'] * (bc_s.loc[7, 'h'] - bc_s.loc[8, 'h'])  # [kW]
        if EVA is None and COND is None and THR is None:
            W_COMP_id = W_COND + W_THR + W_EVA  # [kW]
        if COND is not None:
            W_COMP_id = W_THR + W_EVA  # [kW]
        if EVA is not None:
            W_COMP_id = W_THR + W_COND  # [kW]
        if THR is not None:
            W_COMP_id = W_COND + W_EVA  # [kW]
        m2_id = W_COMP_id / (h3_id - hp_comp_s.loc['2', 'h'])  # [kg/s]
        m2_re = m2 - m2_id  # [kg/s]
        W_COMP_re = m2_re * (h3 - hp_comp_s.loc['2', 'h'])  # [kW]
        if COND is None and EVA is None and THR is None:
            ED_EN_COMP = m2_re * (hp_comp_s.loc['2', 'h'] - hp_comp_s.loc['3', 'h'] - (T_amb + 273.15) * (hp_comp_s.loc['2', 's'] - hp_comp_s.loc['3', 's'])) + W_COMP_re  # [kW]
        if EVA is not None:
            ED_COMP_comp_eva = m2_re * (hp_comp_s.loc['2', 'h'] - hp_comp_s.loc['3', 'h'] - (T_amb + 273.15) * (hp_comp_s.loc['2', 's'] - hp_comp_s.loc['3', 's'])) + W_COMP_re  # [kW]
        if COND is not None:
            ED_COMP_comp_cond = m2_re * (hp_comp_s.loc['2', 'h'] - hp_comp_s.loc['3', 'h'] - (T_amb + 273.15) * (hp_comp_s.loc['2', 's'] - hp_comp_s.loc['3', 's'])) + W_COMP_re  # [kW]
        if THR is not None:
            ED_COMP_comp_thr = m2_re * (hp_comp_s.loc['2', 'h'] - hp_comp_s.loc['3', 'h'] - (T_amb + 273.15) * (hp_comp_s.loc['2', 's'] - hp_comp_s.loc['3', 's'])) + W_COMP_re  # [kW]

        # At the end h3 and s3 are calculated as used as new initial value
        # ==> iteration until the difference between the two values is lower than a tolerance value
        h3_new = (m2_re * hp_comp_s.loc['3', 'h'] + m2_id * h3_id) / m2  # [kJ/kg]
        s3_new = cp.PropsSI('S', 'P', hp_comp_s.loc['3', 'p'] * 1e5, 'H', h3_new * 1e3, 'R245fa') * 1e-3  # [kJ/kgK]
        diff = abs(h3_new - h3)  # [kJ/kg]
        print("The iteration was conducted", iter_step, "times and the difference of the values of h3 between two iteration steps is", diff)
        iter_step += 1
        h3 = h3_new
        s3 = s3_new

    ED_COMP = bc_c.loc['compressor', 'E_D'] * 1e-3  # [kW]
    ED_COND = bc_c.loc['condenser hp', 'E_D'] * 1e-3  # [kW]
    ED_EVA = bc_c.loc['evaporator hp', 'E_D'] * 1e-3  # [kW]
    ED_THR = bc_c.loc['expansion valve', 'E_D'] * 1e-3  # [kW]

    if COND is None and EVA is None:
        print("COMPRESSOR:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2, 3), "kg/s and not", round(bc_s.loc[2, 'm'], 3), "kg/s.")
        print("The endogenous exergy destruction in the compressor is", round(ED_EN_COMP, 3), "kW, compared to the total value of", round(ED_COMP, 3), "kW.")
        print("The endogenous exergy destruction in the compressor is", round(ED_EN_COMP / ED_COMP * 100, 1), "% of the total exergy destruction in the compressor.")
        if details:
            print("The power produced by the heat engine of the idealized condenser is", round(W_COND, 3), "kW.")
            print("The power produced by the expander of the idealized throttle is", round(W_THR, 3), "kW.")
            print("Compared to the ideal case, the mass flow of the ambient water is", round(m5, 3), "kg/s and not", round(bc_s.loc[5, 'm'], 3), "kg/s.")
            print("The power produced by the heat engine of the idealized evaporator is", round(W_EVA, 3), "kW.")
            print("The power produced by the idealized heat exchangers and throttle is", round(W_COMP_id, 3), "kW and is equal to the power of the ideal compressor.")
            print("The mass flow going through the ideal compressor is equal to", round(m2_id, 3), "kg/s and is lower than the total mass flow of", round(m2, 3), "kg/s.")
            print("The mass flow going through the real compressor is equal to", round(m2_re, 3),  "kg/s and is lower than the total mass flow of", round(m2, 3), "kg/s.")
            print("The power produced by the real compressor is", round(W_COMP_re, 3), "kW.")
    if COND is not None:
        print("COMPRESSOR-CONDESER:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2, 3), "kg/s and not", round(bc_s.loc[2, 'm'], 3), "kg/s.")
        print("When the compressor and the condenser are real, the exergy destruction in the compressor is", round(ED_COMP_comp_cond, 3), "kW, compared to the total exergy destruction in the compressor of", round(ED_COMP, 3), "kW.")
        print("When the compressor and the condenser are real, the exergy destruction in the condenser is", round(ED_COND_comp_cond, 3), "kW, compared to the total exergy destruction in the condenser of", round(ED_COND, 3), "kW.")
        print("The endogenous exergy destruction in the compressor was", round(ED_EN_COMP, 3), "kW,", round(ED_EN_COMP / ED_COMP * 100, 1), "% of the total.")
        print("The endogenous exergy destruction in the condenser was", round(ED_EN_COND, 3), "kW,", round(ED_EN_COND / ED_COND * 100, 1), "% of the total.")
        print("The exogenous exergy destruction in the compressor caused by the interaction with the condenser is", round(ED_COMP_comp_cond - ED_EN_COMP, 3), "kW.")
        print("The exogenous exergy destruction in the condenser caused by the interaction with the compressor is", round(ED_COND_comp_cond - ED_EN_COND, 3), "kW.")
    if EVA is not None:
        print(ED_EN_EVA)
        print(ED_EVA)
        print("COMPRESSOR-EVAPORATOR:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2, 3), "kg/s and not", round(bc_s.loc[2, 'm'], 3), "kg/s.")
        print("When the compressor and the evaporator are real, the exergy destruction in the compressor is", round(ED_COMP_comp_eva, 3), "kW, compared to the total exergy destruction in the compressor of", round(ED_COMP, 3), "kW.")
        print("When the compressor and the evaporator are real, the exergy destruction in the evaporator is", round(ED_EVA_comp_eva, 3), "kW, compared to the total exergy destruction in the evaporator of", round(ED_EVA, 3), "kW.")
        print("The endogenous exergy destruction in the compressor was", round(ED_EN_COMP, 3), "kW,", round(ED_EN_COMP / ED_COMP * 100, 1), "% of the total.")
        print("The endogenous exergy destruction in the evaporator was", round(ED_EN_EVA, 3), "kW,", round(ED_EN_EVA / ED_EVA * 100, 1), "% of the total.")
        print("The exogenous exergy destruction in the compressor caused by the interaction with the evaporator is", round(ED_COMP_comp_eva - ED_EN_COMP, 3), "kW.")
        print("The exogenous exergy destruction in the evaporator caused by the interaction with the compressor is", round(ED_EVA_comp_eva - ED_EN_EVA, 3), "kW.")
    if THR is not None:
        print("COMPRESSOR-THROTTLE:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2, 3), "kg/s and not", round(bc_s.loc[2, 'm'], 3), "kg/s.")
        print("When the compressor and the throttle are real, the exergy destruction in the compressor is", round(ED_COMP_comp_thr, 3), "kW, compared to the total exergy destruction in the compressor of", round(ED_COMP, 3), "kW.")
        print("When the compressor and the throttle are real, the exergy destruction in the throttle is", round(ED_THR_comp_thr, 3), "kW, compared to the total exergy destruction in the throttle of", round(ED_THR, 3), "kW.")
        print("The endogenous exergy destruction in the compressor was", round(ED_EN_COMP, 3), "kW,", round(ED_EN_COMP / ED_COMP * 100, 1), "% of the total.")
        print("The endogenous exergy destruction in the throttle was", round(ED_EN_THR, 3), "kW,", round(ED_EN_THR / ED_THR * 100, 1), "% of the total.")
        print("The exogenous exergy destruction in the compressor caused by the interaction with the throttle is", round(ED_COMP_comp_thr - ED_EN_COMP, 3), "kW.")
        print("The exogenous exergy destruction in the throttle caused by the interaction with the compressor is", round(ED_THR_comp_thr - ED_EN_THR, 3), "kW.")

    if COND is None and EVA is None and THR is None:
        return m2, ED_EN_COMP
    if COND is not None:
        return m2, ED_COMP_comp_cond, ED_COND_comp_cond
    if EVA is not None:
        return m2, ED_COMP_comp_eva, ED_EVA_comp_eva
    if THR is not None:
        return m2, ED_COMP_comp_thr, ED_THR_comp_thr


def endo_thr(hp_thr_s, bc_s, bc_c, T_amb):

    m2 = bc_s.loc[7, 'm'] * (bc_s.loc[8, 's'] - bc_s.loc[7, 's']) / (hp_thr_s.loc['3', 's'] - hp_thr_s.loc['4', 's'])  # [kg/s]
    e4_minus_e1 = hp_thr_s.loc['4', 'h'] - hp_thr_s.loc['1', 'h'] - (T_amb + 273.15) * (hp_thr_s.loc['4', 's'] - hp_thr_s.loc['1', 's'])  # [kJ/kg]
    ED_EN_THR = m2 * e4_minus_e1  # [kW]
    ED_THR = bc_c.loc['expansion valve', 'E_D'] * 1e-3  # [kW]
    print("THROTTLE:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2, 3), "kg/s and not", round(bc_s.loc[2, 'm'], 3), "kg/s.")
    print("The endogenous exergy destruction in the throttle is", round(ED_EN_THR, 3), "compared to the total value of", round(ED_THR, 3), "kW.")
    print("The endogenous exergy destruction in the throttle is", round(ED_EN_THR / ED_THR * 100, 1), "% of the total exergy destruction in the throttle.")

    return m2, ED_EN_THR


def endo_eva(hp_eva_s, bc_s, bc_c, T_amb, COND=None, ED_EN_COND=None, ED_EN_EVA=None):
    
    if COND is None:
        m2 = bc_s.loc[7, 'm'] * (bc_s.loc[8, 's'] - bc_s.loc[7, 's']) / (hp_eva_s.loc['3', 's'] - hp_eva_s.loc['4', 's'])  # [kW]
    else:
        m2 = bc_s.loc[7, 'm'] * (bc_s.loc[7, 'h'] - bc_s.loc[8, 'h']) / (hp_eva_s.loc['4', 'h'] - hp_eva_s.loc['3', 'h'])  # [kg/s]
        ED_COND_eva_cond = m2 * (hp_eva_s.loc['3', 'h'] - hp_eva_s.loc['4', 'h'] - (T_amb + 273.15) * (hp_eva_s.loc['3', 's'] - hp_eva_s.loc['4', 's'])) + bc_s.loc[7, 'm'] * (bc_s.loc[7, 'h'] - bc_s.loc[8, 'h'] - (T_amb + 273.15) * (bc_s.loc[7, 's'] - bc_s.loc[8, 's']))
    m5 = m2 * (hp_eva_s.loc['2', 'h'] - hp_eva_s.loc['1', 'h']) / (bc_s.loc[5, 'h'] - bc_s.loc[6, 'h'])  # [kg/s]
    ED_EVA = bc_c.loc['evaporator hp', 'E_D'] * 1e-3  # [kW]
    ED_COND = bc_c.loc['condenser hp', 'E_D'] * 1e-3  # [kW]
    if COND is None:
        ED_EN_EVA = m2 * (hp_eva_s.loc['1', 'h'] - hp_eva_s.loc['2', 'h'] - (T_amb + 273.15) * (hp_eva_s.loc['1', 's'] - hp_eva_s.loc['2', 's'])) + m5 * (bc_s.loc[5, 'h'] - bc_s.loc[6, 'h'] - (T_amb + 273.15) * (bc_s.loc[5, 's'] - bc_s.loc[6, 's']))  # [kW]
        print("EVAPORATOR:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2, 3), "kg/s and not", round(bc_s.loc[2, 'm'], 3), "kg/s.")
        print("The endogenous exergy destruction in the evaporator is", round(ED_EN_EVA, 3), "kW, compared to the total value of", round(ED_EVA, 3), "kW.")
        print("The endogenous exergy destruction in the evaporator is", round(ED_EN_EVA / ED_EVA * 100, 1), "% of the total exergy destruction in the evaporator.")
    else:
        ED_EVA_eva_cond = m2 * (hp_eva_s.loc['1', 'h'] - hp_eva_s.loc['2', 'h'] - (T_amb + 273.15) * (hp_eva_s.loc['1', 's'] - hp_eva_s.loc['2', 's'])) + m5 * (bc_s.loc[5, 'h'] - bc_s.loc[6, 'h'] - (T_amb + 273.15) * (bc_s.loc[5, 's'] - bc_s.loc[6, 's']))  # [kW]
        print("EVAPORATOR-CONDENSER:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2, 3), "kg/s and not", round(bc_s.loc[2, 'm'], 3), "kg/s.")
        print("When the evaporator and the condenser are real, the exergy destruction in the evaporator is", round(ED_EVA_eva_cond, 3), "kW, compared to the total exergy destruction in the evaporator of", round(ED_EVA, 3), "kW.")
        print("When the evaporator and the condenser are real, the exergy destruction in the condenser is", round(ED_COND_eva_cond, 3), "kW, compared to the total exergy destruction in the condenser of", round(ED_COND, 3), "kW.")
        print("The endogenous exergy destruction in the evaporator was", round(ED_EN_EVA, 3), "kW,", round(ED_EN_EVA / ED_EVA * 100, 1), "% of the total.")
        print("The endogenous exergy destruction in the condenser was", round(ED_EN_COND, 3), "kW,", round(ED_EN_COND / ED_COND * 100, 1), "% of the total.")
        print("The exogenous exergy destruction in the evaporator caused by the interaction with the condenser is", round(ED_EVA_eva_cond - ED_EN_EVA, 3), "kW.")
        print("The exogenous exergy destruction in the condenser caused by the interaction with the evaporator is", round(ED_COND_eva_cond - ED_EN_COND, 3), "kW.")

    if COND is None:
        return m2, ED_EN_EVA
    else:
        return m2, ED_EVA_eva_cond, ED_COND_eva_cond

def endo_cond(hp_cond_s, bc_s, bc_c, T_amb):

    m2 = bc_s.loc[7, 'm'] * (bc_s.loc[7, 'h'] - bc_s.loc[8, 'h']) / (hp_cond_s.loc['4', 'h'] - hp_cond_s.loc['3', 'h'])  # [kg/s]
    print("CONDENSER:\nCompared to the ideal case, the mass flow in the HP cycle is", round(m2, 3), "kg/s and not", round(bc_s.loc[2, 'm'], 3), "kg/s.")
    ED_EN_COND = m2 * (hp_cond_s.loc['3', 'h'] - hp_cond_s.loc['4', 'h'] - (T_amb + 273.15) * (hp_cond_s.loc['3', 's'] - hp_cond_s.loc['4', 's'])) + bc_s.loc[7, 'm'] * (bc_s.loc[7, 'h'] - bc_s.loc[8, 'h'] - (T_amb + 273.15) * (bc_s.loc[7, 's'] - bc_s.loc[8, 's']))  # [kW]
    ED_COND = bc_c.loc['condenser hp', 'E_D'] * 1e-3  # [kW]
    print("The endogenous exergy destruction in the condenser is", round(ED_EN_COND, 3), "kW, compared to the total value of", round(ED_COND, 3), "kW.")
    print("The endogenous exergy destruction in the condenser is", round(ED_EN_COND / ED_COND * 100, 1), "% of the total exergy destruction in the condenser.")

    return m2, ED_EN_COND
