import numpy as np
from CoolProp.CoolProp import PropsSI as PSI
import pandas as pd
from func_fix import (pr_func, pr_deriv, eta_s_compressor_func, eta_s_compressor_deriv, turbo_func, turbo_deriv,
                      he_func, he_deriv, ihx_func, ihx_deriv, temperature_func, temperature_deriv, valve_func,
                      valve_deriv, x_saturation_func, x_saturation_deriv, qt_diagram, eps_compressor_func,
                      eps_compressor_deriv, eps_real_he_func, eps_real_he_deriv, eps_real_ihx_func, eps_real_ihx_deriv,
                      simple_he_func, simple_he_deriv, ideal_ihx_entropy_func, ideal_ihx_entropy_deriv,
                      ideal_he_entropy_func, ideal_he_entropy_deriv, ideal_valve_entropy_func,
                      ideal_valve_entropy_deriv, he_with_p_func, he_with_p_deriv, same_temperature_func,
                      same_temperature_deriv, eta_s_expander_func,  eta_s_expander_deriv, ttd_func, ttd_deriv, 
                      eps_expander_func, eps_expander_deriv)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def orc_simultaneous(target_p63, print_results, config, label, adex=False, plot=False):

    try:
        epsilon = pd.read_csv('outputs/adex_orc/orc_comps_real.csv', index_col=0)['epsilon']  # from base case
        # Additional code to process epsilon if needed
    except FileNotFoundError:
        print('File not found. Please run base case model first!')
        # Handle the exception (e.g., set epsilon to None or a default value)
        epsilon = None

    wf = 'REFPROP::R134a'
    fluid_tes = 'REFPROP::water'

    # TES
    t51 = 140 + 273.15  # input is known
    p51 = 5e5           # input is known
    t52 = 70 + 273.15   # output temperature is set
    m51 = 10            # dimensioning of the system (full load)

    # PRESSURE DROPS
    if config['cond']:
        pr_cond_hot = 0.95
    else:
        pr_cond_hot = 1
    if config['ihx']:
        pr_ihx_hot = 0.985
        pr_ihx_cold = 0.985
    else:
        pr_ihx_hot = 1
        pr_ihx_cold = 1
    if config['eva']:
        pr_eva_hot = 1
        pr_eva_cold = 0.95
    else:
        pr_eva_hot = 1
        pr_eva_cold = 1

    # AMBIENT
    t0 = 10 + 273.15    # K
    p0 = 1.013e5        # bar
    
    # HEAT PUMP
    if config['cond']:
        ttd_l_cond = 5  # K
    else:
        ttd_l_cond = 0  # K
        
    if config['ihx']:
        ttd_l_ihx = 5   # K
    else:
        ttd_l_ihx = 0   # K

    if config['eva']:
        ttd_u_eva = 5   # K
    else:
        ttd_u_eva = 0   # K
        
    pr_eva_part_cold = np.cbrt(pr_eva_cold)  # eva pressure drop is split equally (geom, mean) between ECO-EVA-SH
    pr_eva_part_hot = np.cbrt(pr_eva_hot)    # eva pressure drop is split equally (geom, mean) between ECO-EVA-SH

    # TECHNICAL PARAMETERS
    eta_s_pump = 0.85
    eta_s_exp = 0.9
        
    # PRE-CALCULATION
    t61 = t0 + ttd_l_cond
    t64 = t51 - ttd_u_eva

    # STARTING VALUES
    h61 = 218e3
    h62 = 220e3
    h63 = 260e3
    h64 = 500e3
    h65 = 455e3
    h66 = 410e3
    m61 = 12.3
    h51 = 589e3
    h52 = 293e3
    p61 = 4.4e5
    p62 = 25e5
    p64 = 25e5
    p68 = 25e5
    p69 = 25e5
    p65 = 4.7e5
    p66 = 4.7e5
    p52 = 5e5
    h58 = 485e3
    h59 = 332e3
    h68 = 330e3
    h69 = 425e3
    p58 = 5e5
    p59 = 5e5

    variables = np.array([h61, p61, h62, h66, p66, p62, p64, h64, p65, h65, h63, h51, h52, p52, m61,
                          h68, h69, h58, h59, p68, p69, p58, p59])
    residual = np.ones(len(variables))

    iter_step = 0

    while np.linalg.norm(residual) > 1e-4:
        # TODO [h61, p61, h62, h66, p66, p62, p64, h64, p65, h65, h63, h51, h52, p52, m61
        # TODO   0    1    2    3    4    5    6    7    8    9    10   11   12   13   14
        #       h68, h69, h58, h59, p68, p69, p58, p59])]
        #        15   16   17   18   19   20   21   22   
        # 0
        t61_set = temperature_func(t61, variables[0], variables[1], wf)
        # 1
        p61_calc_cond = x_saturation_func(0, variables[0], variables[1], wf)
        # 2
        if adex and not config['pump']:  # ideal pump
            t62_calc_pump = eps_compressor_func(1, variables[0], variables[1], variables[2], variables[5], wf)
        elif adex and config['pump']:  # real pump for adv. exergy analysis
            t62_calc_pump = eps_compressor_func(epsilon['pump'], variables[0], variables[1], variables[2], variables[5], wf)
        else:  # real pump for base design
            t62_calc_pump = eta_s_compressor_func(eta_s_pump, variables[0], variables[1], variables[2], variables[5], wf)
        # 3
        if adex and not config['ihx']:  # ideal ihx
            t66_set = same_temperature_func(variables[2], variables[5], wf, variables[3], variables[4], wf)
        elif adex and config['ihx']:  # real ihx for adv. exergy analysis
            t66_set = eps_real_ihx_func(epsilon['ihx'], variables[9], variables[8], variables[3], variables[4], wf,
                                        variables[2], variables[5], variables[10], target_p63, wf)
        else:  # real ihx for base design
            t66_pre_calc_ihx = ttd_func(variables[2], variables[5], ttd_l_ihx, wf)  # NOT part of equation system
            t66_set = temperature_func(t66_pre_calc_ihx, variables[3], variables[4], wf)  # t66 = t62
        # 4
        p66_set = pr_func(pr_cond_hot, variables[4], variables[1])
        # 5
        p62_set = pr_func(pr_ihx_cold, variables[5], target_p63)
        # 6
        p64_set = pr_func(pr_eva_cold, target_p63, variables[6])
        # 7
        t64_set = temperature_func(t64, variables[7], variables[6], wf)
        # 8
        p65_set = pr_func(pr_ihx_hot, variables[8], variables[4])
        # 9
        if adex and not config['exp']:  # ideal exp
            t65_calc_exp = eps_expander_func(1, variables[7], variables[6], variables[9], variables[8], wf)
        elif adex and config['exp']:  # real exp for adv. exergy analysis
            t65_calc_exp = eps_expander_func(epsilon['exp'], variables[7], variables[6], variables[9], variables[8], wf)
        else:  # real exp for base design
            t65_calc_exp = eta_s_expander_func(eta_s_exp, variables[7], variables[6], variables[9], variables[8], wf)
        # 10
        if adex and not config['ihx']:  # ideal ihx
            if not config['eva']:  # ideal eva
                t63_calc = ideal_ihx_entropy_func(variables[9], variables[8], variables[3], variables[4], wf,
                                                  variables[2], variables[5], variables[10], target_p63, wf)
            else:  # real eva for adv. exergy analysis
                t63_calc = eps_real_he_func(epsilon['eva'], variables[11], p51, variables[12], variables[13], m51, fluid_tes,
                                            variables[10], target_p63, variables[7], variables[6], variables[14], wf)
        else:  # real ihx for base design or for adv. exergy analysis
            t63_calc = ihx_func(variables[9], variables[3], variables[2], variables[10])
        # 11
        t51_set = temperature_func(t51, variables[11], p51, fluid_tes)
        # 12
        t52_set = temperature_func(t52, variables[12], variables[13], fluid_tes)
        # 13
        p52_set = pr_func(pr_eva_hot, p51, variables[13])
        # 14
        if adex and not config['eva']:  # ideal eva
            m61_calc_eva = ideal_he_entropy_func(variables[11], p51, variables[12], variables[13], m51, fluid_tes,
                                                 variables[10], target_p63, variables[7], variables[6], variables[14], wf)
        else:  # real eva for base design or for adv. exergy analysis
            m61_calc_eva = he_func(m51, variables[11], variables[12], variables[14], variables[10], variables[7])
        # 15
        eva_eco_outlet_sat = x_saturation_func(0, variables[15], variables[19], wf)
        # 16
        eva_eco_en_bal = he_func(m51, variables[18], variables[12], variables[14], variables[10], variables[15])
        # 17
        eva_sh_inlet_sat = x_saturation_func(1, variables[16], variables[20], wf)
        # 18
        eva_sh_en_bal = he_func(m51, variables[11], variables[17], variables[14], variables[16], variables[7])
        # 19
        p68_set = pr_func(pr_eva_part_cold, target_p63, variables[19])
        # 20
        p69_set = pr_func(pr_eva_part_cold, variables[19], variables[20])
        # 21
        p58_set = pr_func(pr_eva_part_hot, p51, variables[21])
        # 22
        p59_set = pr_func(pr_eva_part_hot, variables[21], variables[22])

        residual = np.array([t61_set, p61_calc_cond, t62_calc_pump, t66_set, p66_set, p62_set, p64_set, t64_set, 
                             p65_set, t65_calc_exp, t63_calc, t51_set, t52_set, p52_set, m61_calc_eva,
                             eva_eco_outlet_sat, eva_eco_en_bal, eva_sh_inlet_sat, eva_sh_en_bal, p68_set, p69_set,
                             p58_set, p59_set])
        jacobian = np.zeros((len(variables), len(variables)))

        # 0
        t61_set_j = temperature_deriv(t61, variables[0], variables[1], wf)
        # 1
        p61_calc_cond_j = x_saturation_deriv(0, variables[0], variables[1], wf)
        # 2
        if adex and not config['pump']:  # ideal pump
            t62_calc_pump_j = eps_compressor_deriv(1, variables[0], variables[1], variables[2], variables[5], wf)
        elif adex and config['pump']:  # real pump for adv. exergy analysis
            t62_calc_pump_j = eps_compressor_deriv(epsilon['pump'], variables[0], variables[1], variables[2], variables[5], wf)
        else:  # real pump for base design
            t62_calc_pump_j = eta_s_compressor_deriv(eta_s_pump, variables[0], variables[1], variables[2], variables[5], wf)
        # 3
        if adex and not config['ihx']:  # ideal ihx
            t66_set_j = same_temperature_deriv(variables[2], variables[5], wf, variables[3], variables[4], wf)
        elif adex and config['ihx']:  # real ihx for adv. exergy analysis
            t66_set_j = eps_real_ihx_deriv(epsilon['ihx'], variables[9], variables[8], variables[3], variables[4], wf,
                                        variables[2], variables[5], variables[10], target_p63, wf)
        else:  # real ihx for base design
            t66_pre_calc_ihx = ttd_func(variables[2], variables[5], ttd_l_ihx, wf)  # NOT part of equation system
            t66_set_j = temperature_deriv(t66_pre_calc_ihx, variables[3], variables[4], wf)  # t66_j = t62
        # 4
        p66_set_j = pr_deriv(pr_cond_hot, variables[4], variables[1])
        # 5
        p62_set_j = pr_deriv(pr_ihx_cold, variables[5], target_p63)
        # 6
        p64_set_j = pr_deriv(pr_eva_cold, target_p63, variables[6])
        # 7
        t64_set_j = temperature_deriv(t64, variables[7], variables[6], wf)
        # 8
        p65_set_j = pr_deriv(pr_ihx_hot, variables[8], variables[4])
        # 9
        if adex and not config['exp']:  # ideal exp
            t65_calc_exp_j = eps_expander_deriv(1, variables[7], variables[6], variables[9], variables[8], wf)
        elif adex and config['exp']:  # real exp for adv. exergy analysis
            t65_calc_exp_j = eps_expander_deriv(epsilon['exp'], variables[7], variables[6], variables[9], variables[8], wf)
        else:  # real exp for base design
            t65_calc_exp_j = eta_s_expander_deriv(eta_s_exp, variables[7], variables[6], variables[9], variables[8], wf)
        # 10
        if adex and not config['ihx']:  # ideal ihx
            if not config['eva']:  # ideal eva
                t63_calc_j = ideal_ihx_entropy_deriv(variables[9], variables[8], variables[3], variables[4], wf,
                                                     variables[2], variables[5], variables[10], target_p63, wf)
            else:  # real eva for adv. exergy analysis
                t63_calc_j = eps_real_he_deriv(epsilon['eva'], variables[11], p51, variables[12], variables[13], m51, fluid_tes,
                                               variables[10], target_p63, variables[7], variables[6], variables[14], wf)
        else:  # real ihx for base design or for adv. exergy analysis
            t63_calc_j = ihx_deriv(variables[9], variables[3], variables[2], variables[10])
        # 11
        t51_set_j = temperature_deriv(t51, variables[11], p51, fluid_tes)
        # 12
        t52_set_j = temperature_deriv(t52, variables[12], variables[13], fluid_tes)
        # 13
        p52_set_j = pr_deriv(pr_eva_hot, p51, variables[13])
        # 14
        if adex and not config['eva']:  # ideal eva
            m61_calc_eva_j = ideal_he_entropy_deriv(variables[11], p51, variables[12], variables[13], m51, fluid_tes,
                                                    variables[10], target_p63, variables[7], variables[6], variables[14], wf)
        else:  # real eva for base design or for adv. exergy analysis
            m61_calc_eva_j = he_deriv(m51, variables[11], variables[12], variables[14], variables[10], variables[7])
        # 15
        eva_eco_outlet_sat_j = x_saturation_deriv(0, variables[15], variables[19], wf)
        # 16
        eva_eco_en_bal_j = he_deriv(m51, variables[18], variables[12], variables[14], variables[10], variables[15])
        # 17
        eva_sh_inlet_sat_j = x_saturation_deriv(1, variables[16], variables[20], wf)
        # 18
        eva_sh_en_bal_j = he_deriv(m51, variables[11], variables[17], variables[14], variables[16], variables[7])
        # 19
        p68_set_j = pr_deriv(pr_eva_part_cold, target_p63, variables[19])
        # 20
        p69_set_j = pr_deriv(pr_eva_part_cold, variables[19], variables[20])
        # 21
        p58_set_j = pr_deriv(pr_eva_part_hot, p51, variables[21])
        # 22
        p59_set_j = pr_deriv(pr_eva_part_hot, variables[21], variables[22])
        
        # TODO [h61, p61, h62, h66, p66, p62, p64, h64, p65, h65, h63, h51, h52, p52, m61
        # TODO   0    1    2    3    4    5    6    7    8    9    10   11   12   13   14
        #       h68, h69, h58, h59, p68, p69, p58, p59])]
        #        15   16   17   18   19   20   21   22   
        jacobian[0, 0] = t61_set_j['h']  # derivative of t61_set with respect to h61
        jacobian[0, 1] = t61_set_j['p']  # derivative of t61_set with respect to p61
        jacobian[1, 0] = p61_calc_cond_j['h']  # derivative of p61_calc_cond with respect to h61
        jacobian[1, 1] = p61_calc_cond_j['p']  # derivative of p61_calc_cond with respect to p61
        jacobian[2, 0] = t62_calc_pump_j['h_1']  # derivative of t62_calc_pump with respect to h61
        jacobian[2, 1] = t62_calc_pump_j['p_1']  # derivative of t62_calc_pump with respect to p61
        jacobian[2, 2] = t62_calc_pump_j['h_2']  # derivative of t62_calc_pump with respect to h62
        jacobian[2, 5] = t62_calc_pump_j['p_2']  # derivative of t62_calc_pump with respect to p62
        if adex and not config['ihx']:
            jacobian[3, 2] = t66_set_j['h_copy']  # derivative of t66_set_j with respect to h65
            jacobian[3, 5] = t66_set_j['p_copy']  # derivative of t66_set_j with respect to p65
            jacobian[3, 3] = t66_set_j['h_paste']  # derivative of t66_set_j with respect to h66
            jacobian[3, 4] = t66_set_j['p_paste']  # derivative of t66_set_j with respect to p66
        elif adex and config['ihx']:
            jacobian[3, 9] = t66_set_j['h_hot_in']  # derivative of t66_set_j with respect to h65
            jacobian[3, 8] = t66_set_j['p_hot_in']  # derivative of t66_set_j with respect to p65
            jacobian[3, 3] = t66_set_j['h_hot_out']  # derivative of t66_set_j with respect to h66
            jacobian[3, 4] = t66_set_j['p_hot_out']  # derivative of t66_set_j with respect to p66
            jacobian[3, 2] = t66_set_j['h_cold_in']  # derivative of t66_set_j with respect to h62
            jacobian[3, 5] = t66_set_j['p_cold_in']  # derivative of t66_set_j with respect to p62
            jacobian[3, 10] = t66_set_j['h_cold_out']  # derivative of t66_set_j with respect to h63
        else:
            jacobian[3, 3] = t66_set_j['h']  # derivative of t66_set with respect to h66
            jacobian[3, 4] = t66_set_j['p']  # derivative of t66_set with respect to p66
        jacobian[4, 4] = p66_set_j['p_1']  # derivative of p66_set with respect to p66
        jacobian[4, 1] = p66_set_j['p_2']  # derivative of p66_set with respect to p61
        jacobian[5, 5] = p62_set_j['p_1']  # derivative of p62_set with respect to p63
        jacobian[6, 6] = p64_set_j['p_2']  # derivative of p64_set with respect to p64
        jacobian[7, 7] = t64_set_j['h']  # derivative of t64_set with respect to h64
        jacobian[7, 6] = t64_set_j['p']  # derivative of t64_set with respect to p64
        jacobian[8, 8] = p65_set_j['p_1']  # derivative of p64_set with respect to p65
        jacobian[8, 4] = p65_set_j['p_2']  # derivative of p64_set with respect to p66
        jacobian[9, 7] = t65_calc_exp_j['h_1']  # derivative of t65_calc_exp with respect to h64
        jacobian[9, 6] = t65_calc_exp_j['p_1']  # derivative of t65_calc_exp with respect to p64
        jacobian[9, 9] = t65_calc_exp_j['h_2']  # derivative of t65_calc_exp with respect to h65
        jacobian[9, 8] = t65_calc_exp_j['p_2']  # derivative of t65_calc_exp with respect to p65
        if adex and not config['ihx']:  # ideal ihx
            if not config['eva']:  # ideal eva
                jacobian[10, 9] = t63_calc_j['h_hot_in']  # derivative of t63_calc_ihx_j with respect to h65
                jacobian[10, 8] = t63_calc_j['p_hot_in']  # derivative of t63_calc_ihx_j with respect to p65
                jacobian[10, 3] = t63_calc_j['h_hot_out']  # derivative of t63_calc_ihx_j with respect to h66
                jacobian[10, 4] = t63_calc_j['p_hot_out']  # derivative of t63_calc_ihx_j with respect to p66
                jacobian[10, 2] = t63_calc_j['h_cold_in']  # derivative of t63_calc_ihx_j with respect to h62
                jacobian[10, 5] = t63_calc_j['p_cold_in']  # derivative of t63_calc_ihx_j with respect to p62
                jacobian[10, 10] = t63_calc_j['h_cold_out']  # derivative of t63_calc_ihx_j with respect to h63
            else:
                jacobian[10, 11] = t63_calc_j['h_hot_in']  # derivative of m61_calc_eva with respect to h51
                jacobian[10, 12] = t63_calc_j['h_hot_out']  # derivative of m61_calc_eva with respect to h52
                jacobian[10, 13] = t63_calc_j['p_hot_out']  # derivative of m61_calc_eva with respect to p52
                jacobian[10, 10] = t63_calc_j['h_cold_in']  # derivative of m61_calc_eva with respect to h63
                jacobian[10, 7] = t63_calc_j['h_cold_out']  # derivative of m61_calc_eva with respect to h64
                jacobian[10, 6] = t63_calc_j['p_cold_out']  # derivative of m61_calc_eva with respect to p64
                jacobian[10, 14] = t63_calc_j['m_cold']  # derivative of m61_calc_eva with respect to m61
        else:
            jacobian[10, 9] = t63_calc_j['h_1']  # derivative of t63_calc_ihx with respect to h65
            jacobian[10, 3] = t63_calc_j['h_2']  # derivative of t63_calc_ihx with respect to h66
            jacobian[10, 2] = t63_calc_j['h_3']  # derivative of t63_calc_ihx with respect to h62
            jacobian[10, 10] = t63_calc_j['h_4']  # derivative of t63_calc_ihx with respect to h63
        jacobian[11, 11] = t51_set_j['h']  # derivative of t51_set with respect to h51
        jacobian[12, 12] = t52_set_j['h']  # derivative of t52_set with respect to h52
        jacobian[12, 13] = t52_set_j['p']  # derivative of t52_set with respect to p52
        jacobian[13, 13] = p52_set_j['p_2']  # derivative of p52_set with respect to p52
        if adex and not config['eva']:
            jacobian[14, 11] = m61_calc_eva_j['h_hot_in']  # derivative of m61_calc_eva with respect to h51
            jacobian[14, 12] = m61_calc_eva_j['h_hot_out']  # derivative of m61_calc_eva with respect to h52
            jacobian[14, 13] = m61_calc_eva_j['p_hot_out']  # derivative of m61_calc_eva with respect to p52
            jacobian[14, 10] = m61_calc_eva_j['h_cold_in']  # derivative of m61_calc_eva with respect to h63
            jacobian[14, 7] = m61_calc_eva_j['h_cold_out']  # derivative of m61_calc_eva with respect to h64
            jacobian[14, 6] = m61_calc_eva_j['p_cold_out']  # derivative of m61_calc_eva with respect to p64
            jacobian[14, 14] = m61_calc_eva_j['m_cold']  # derivative of m61_calc_eva with respect to m61
        else:
            jacobian[14, 11] = m61_calc_eva_j['h_1']  # derivative of m61_calc_eva with respect to h51
            jacobian[14, 12] = m61_calc_eva_j['h_2']  # derivative of m61_calc_eva with respect to h52
            jacobian[14, 14] = m61_calc_eva_j['m_cold']  # derivative of m61_calc_eva with respect to m61
            jacobian[14, 10] = m61_calc_eva_j['h_3']  # derivative of m61_calc_eva with respect to h63
            jacobian[14, 7] = m61_calc_eva_j['h_4']  # derivative of m61_calc_eva with respect to h64
        jacobian[15, 15] = eva_eco_outlet_sat_j['h']
        jacobian[15, 19] = eva_eco_outlet_sat_j['p']
        jacobian[16, 18] = eva_eco_en_bal_j['h_1']
        jacobian[16, 12] = eva_eco_en_bal_j['h_2']
        jacobian[16, 14] = eva_eco_en_bal_j['m_cold']
        jacobian[16, 10] = eva_eco_en_bal_j['h_3']
        jacobian[16, 15] = eva_eco_en_bal_j['h_4']
        jacobian[17, 16] = eva_sh_inlet_sat_j['h']
        jacobian[17, 20] = eva_sh_inlet_sat_j['p']
        jacobian[18, 11] = eva_sh_en_bal_j['h_1']
        jacobian[18, 17] = eva_sh_en_bal_j['h_2']
        jacobian[18, 14] = eva_sh_en_bal_j['m_cold']
        jacobian[18, 16] = eva_sh_en_bal_j['h_3']
        jacobian[18, 7] = eva_sh_en_bal_j['h_4']
        jacobian[19, 19] = p68_set_j['p_2']
        jacobian[20, 19] = p69_set_j['p_1']
        jacobian[20, 20] = p69_set_j['p_2']
        jacobian[21, 21] = p58_set_j['p_2']
        jacobian[22, 21] = p59_set_j['p_1']
        jacobian[22, 22] = p59_set_j['p_2']
        # Save the DataFrame as a CSV file
        # pd.DataFrame(jacobian).round(4).to_csv('jacobian_orc.csv', index=False)

        variables -= np.linalg.inv(jacobian).dot(residual)

        iter_step += 1
        cond_number = np.linalg.cond(jacobian)
        # print('Condition number: ', cond_number, ' and residual: ', np.linalg.norm(residual))

    # TODO [h61, p61, h62, h66, p66, p62, p64, h64, p65, h65, h63, h51, h52, p52, m61
    # TODO   0    1    2    3    4    5    6    7    8    9    10   11   12   13   14
    #       h68, h69, h58, h59, p68, p69, p58, p59])]
    #        15   16   17   18   19   20   21   22   
    p61 = variables[1]
    p62 = variables[5]
    p63 = target_p63
    p64 = variables[6]
    p65 = variables[8]
    p66 = variables[4]
    p51 = p51
    p52 = variables[13]
    p68 = variables[19]
    p69 = variables[20]
    p58 = variables[21]
    p59 = variables[22]

    h61 = variables[0]
    h62 = variables[2]
    h63 = variables[10]
    h64 = variables[7]
    h65 = variables[9]
    h66 = variables[3]
    h51 = variables[11]
    h52 = variables[12]
    h68 = variables[15]
    h69 = variables[16]
    h58 = variables[17]
    h59 = variables[18]
    
    t61 = PSI('T', 'H', h61, 'P', p61, wf)
    t62 = PSI('T', 'H', h62, 'P', p62, wf)
    t63 = PSI('T', 'H', h63, 'P', p63, wf)
    t64 = PSI('T', 'H', h64, 'P', p64, wf)
    t65 = PSI('T', 'H', h65, 'P', p65, wf)
    t66 = PSI('T', 'H', h66, 'P', p66, wf)
    t68 = PSI('T', 'H', h68, 'P', p68, wf)
    t69 = PSI('T', 'H', h69, 'P', p69, wf)
    t51 = PSI('T', 'H', h51, 'P', p51, fluid_tes)
    t52 = PSI('T', 'H', h52, 'P', p52, fluid_tes)
    t58 = PSI('T', 'H', h58, 'P', p58, fluid_tes)
    t59 = PSI('T', 'H', h59, 'P', p59, fluid_tes)

    s61 = PSI('S', 'H', h61, 'P', p61, wf)
    s62 = PSI('S', 'H', h62, 'P', p62, wf)
    s63 = PSI('S', 'H', h63, 'P', p63, wf)
    s64 = PSI('S', 'H', h64, 'P', p64, wf)
    s65 = PSI('S', 'H', h65, 'P', p65, wf)
    s66 = PSI('S', 'H', h66, 'P', p66, wf)
    s68 = PSI('S', 'H', h68, 'P', p68, wf)
    s69 = PSI('S', 'H', h69, 'P', p69, wf)
    s51 = PSI('S', 'H', h51, 'P', p51, fluid_tes)
    s52 = PSI('S', 'H', h52, 'P', p52, fluid_tes)
    s58 = PSI('S', 'H', h58, 'P', p58, fluid_tes)
    s59 = PSI('S', 'H', h59, 'P', p59, fluid_tes)
    
    m61 = variables[14]

    df_streams = pd.DataFrame(index=[61, 62, 63, 64, 65, 66, 51, 52, 68, 69, 58, 59],
                      columns=['m [kg/s]', 'T [°C]', 'h [kJ/kg]', 'p [bar]', 's [J/kgK]', 'fluid'])
    df_streams.loc[61] = [m61, t61, h61, p61, s61, wf]
    df_streams.loc[62] = [m61, t62, h62, p62, s62, wf]
    df_streams.loc[63] = [m61, t63, h63, p63, s63, wf]
    df_streams.loc[64] = [m61, t64, h64, p64, s64, wf]
    df_streams.loc[65] = [m61, t65, h65, p65, s65, wf]
    df_streams.loc[66] = [m61, t66, h66, p66, s66, wf]
    df_streams.loc[68] = [m61, t68, h68, p68, s68, wf]
    df_streams.loc[69] = [m61, t69, h69, p69, s69, wf]
    df_streams.loc[51] = [m51, t51, h51, p51, s51, fluid_tes]
    df_streams.loc[52] = [m51, t52, h52, p52, s52, fluid_tes]
    df_streams.loc[58] = [m51, t58, h58, p58, s58, fluid_tes]
    df_streams.loc[59] = [m51, t59, h59, p59, s59, fluid_tes]
    
    df_streams['T [°C]'] = df_streams['T [°C]'] - 273.15
    df_streams['h [kJ/kg]'] = df_streams['h [kJ/kg]'] * 1e-3
    df_streams['p [bar]'] = df_streams['p [bar]'] * 1e-5

    if print_results:
        print('-------------------------------------------------------------\n', 'case:', label)
        print(df_streams.iloc[:, :5])
        print('-------------------------------------------------------------')

    power_pump = df_streams.loc[61, 'm [kg/s]'] * (df_streams.loc[62, 'h [kJ/kg]'] - df_streams.loc[61, 'h [kJ/kg]'])
    power_exp = df_streams.loc[61, 'm [kg/s]'] * (df_streams.loc[64, 'h [kJ/kg]'] - df_streams.loc[65, 'h [kJ/kg]'])
    heat_eva = df_streams.loc[51, 'm [kg/s]'] * (df_streams.loc[51, 'h [kJ/kg]'] - df_streams.loc[52, 'h [kJ/kg]'])
    heat_cond = df_streams.loc[61, 'm [kg/s]'] * (df_streams.loc[66, 'h [kJ/kg]'] - df_streams.loc[61, 'h [kJ/kg]'])
    power_ihx = df_streams.loc[61, 'm [kg/s]'] * (df_streams.loc[65, 'h [kJ/kg]'] - df_streams.loc[66, 'h [kJ/kg]'] +
                                                  df_streams.loc[62, 'h [kJ/kg]'] - df_streams.loc[63, 'h [kJ/kg]'])
    power_eva = (df_streams.loc[51, 'm [kg/s]'] * (df_streams.loc[51, 'h [kJ/kg]'] - df_streams.loc[52, 'h [kJ/kg]'])
                 + df_streams.loc[61, 'm [kg/s]'] * (df_streams.loc[63, 'h [kJ/kg]'] - df_streams.loc[64, 'h [kJ/kg]']))
    if abs(heat_eva+power_pump-power_exp-heat_cond-power_ihx-power_eva) > 1e-4:
        print('Energy balances are not fulfilled! :(')

    if plot:
        qt_diagram(df_streams, 'IHX', 65, 66, 62, 63,
                   ttd_l_ihx, 'ORC', plot=plot, case=f'{label} with ttd_l_ihx={ttd_l_ihx}')
        [min_td_eva, _] = qt_diagram(df_streams, 'EVA', 51, 52, 63, 64,
                                     ttd_u_eva, 'ORC', plot=plot, case=f'{label} with ttd_u_eva={ttd_u_eva}')

    t_pinch_eva_sh = t59 - t68

    efficiency = (power_exp+power_eva+power_eva-power_pump)/heat_eva

    return df_streams, t_pinch_eva_sh, efficiency


def exergy_analysis_orc(t0, p0, df):
    
    h0_wf = PSI('H', 'T', t0, 'P', p0, df.loc[61, 'fluid']) * 1e-3
    s0_wf = PSI('S', 'T', t0, 'P', p0, df.loc[61, 'fluid'])
    h0_tes = PSI('H', 'T', t0, 'P', p0, df.loc[51, 'fluid']) * 1e-3
    s0_tes = PSI('S', 'T', t0, 'P', p0, df.loc[51, 'fluid'])

    for i in [61, 62, 63, 64, 65, 66, 68, 69]:
        df.loc[i, 'e^PH [kJ/kg]'] = df.loc[i, 'h [kJ/kg]'] - h0_wf - t0 * (df.loc[i, 's [J/kgK]'] - s0_wf) * 1e-3
    for i in [51, 52, 58, 59]:
        df.loc[i, 'e^PH [kJ/kg]'] = df.loc[i, 'h [kJ/kg]'] - h0_tes - t0 * (df.loc[i, 's [J/kgK]'] - s0_tes) * 1e-3

    # ED_PUMP = T0 * (S62 - S61)
    ed_pump = t0 * (df.loc[61, 'm [kg/s]'] * (df.loc[62, 's [J/kgK]'] - df.loc[61, 's [J/kgK]'])) * 1e-3
    # ED_EVA = T0 * (S63 - S62 + S52 - S51)
    ed_eva = t0 * (df.loc[61, 'm [kg/s]'] * (df.loc[64, 's [J/kgK]'] - df.loc[63, 's [J/kgK]']) +
                   df.loc[51, 'm [kg/s]'] * (df.loc[52, 's [J/kgK]'] - df.loc[51, 's [J/kgK]'])) * 1e-3
    # ED_IHX = T0 * (S63 - S62 + S66 - S65)
    ed_ihx = t0 * (df.loc[61, 'm [kg/s]'] * (df.loc[63, 's [J/kgK]'] - df.loc[62, 's [J/kgK]']) +
                   df.loc[61, 'm [kg/s]'] * (df.loc[66, 's [J/kgK]'] - df.loc[65, 's [J/kgK]'])) * 1e-3
    # ED_EXP = T0 * (S65 - S64)
    ed_exp = t0 * (df.loc[61, 'm [kg/s]'] * (df.loc[65, 's [J/kgK]'] - df.loc[64, 's [J/kgK]'])) * 1e-3
    # ED_EVA = Q * (1 - T0/Tb)
    temp_boundary = ((df.loc[61, 'h [kJ/kg]'] - df.loc[66, 'h [kJ/kg]'])
                     / (df.loc[61, 's [J/kgK]'] - df.loc[66, 's [J/kgK]']) * 1e3)
    ed_cond = (df.loc[61, 'm [kg/s]'] * (df.loc[66, 'h [kJ/kg]'] - df.loc[61, 'h [kJ/kg]'])) * (
                1 - t0 / temp_boundary)

    ed = {
        'pump': ed_pump,
        'eva': ed_eva,
        'ihx': ed_ihx,
        'exp': ed_exp,
        'cond': ed_cond,
        'tot': ed_pump+ed_cond+ed_ihx+ed_exp+ed_eva
    }

    ef_pump = df.loc[61, 'm [kg/s]'] * (df.loc[62, 'h [kJ/kg]'] - df.loc[61, 'h [kJ/kg]'])  # inlet power
    ef_ihx = df.loc[61, 'm [kg/s]'] * (df.loc[65, 'e^PH [kJ/kg]'] - df.loc[66, 'e^PH [kJ/kg]'])
    ef_eva = df.loc[51, 'm [kg/s]'] * (df.loc[51, 'e^PH [kJ/kg]'] - df.loc[52, 'e^PH [kJ/kg]'])
    ef_exp = df.loc[61, 'm [kg/s]'] * (df.loc[64, 'e^PH [kJ/kg]'] - df.loc[65, 'e^PH [kJ/kg]'])

    epsilon_pump = 1 - ed_pump / ef_pump
    epsilon_eva = 1 - ed_eva / ef_eva
    epsilon_exp = 1 - ed_exp / ef_exp
    epsilon_ihx = 1 - ed_ihx / ef_ihx

    heat_eva = df.loc[51, 'm [kg/s]'] * (df.loc[51, 'h [kJ/kg]'] - df.loc[52, 'h [kJ/kg]'])
    heat_cond = df.loc[61, 'm [kg/s]'] * (df.loc[66, 'h [kJ/kg]'] - df.loc[61, 'h [kJ/kg]'])
    heat_ihx = df.loc[61, 'm [kg/s]'] * (df.loc[63, 'h [kJ/kg]'] - df.loc[62, 'h [kJ/kg]'])
    power_pump = df.loc[61, 'm [kg/s]'] * (df.loc[62, 'h [kJ/kg]'] - df.loc[61, 'h [kJ/kg]'])
    power_exp = df.loc[61, 'm [kg/s]'] * (df.loc[64, 'h [kJ/kg]'] - df.loc[65, 'h [kJ/kg]'])
    power_eva = (df.loc[51, 'm [kg/s]'] * (df.loc[51, 'h [kJ/kg]'] - df.loc[52, 'h [kJ/kg]'])
                 + df.loc[61, 'm [kg/s]'] * (df.loc[63, 'h [kJ/kg]'] - df.loc[64, 'h [kJ/kg]']))
    power_ihx = df.loc[61, 'm [kg/s]'] * (df.loc[65, 'h [kJ/kg]'] - df.loc[66, 'h [kJ/kg]'] +
                                          df.loc[62, 'h [kJ/kg]'] - df.loc[63, 'h [kJ/kg]'])

    exergy_efficiency = ((- power_pump + power_ihx + power_exp + power_eva)
                         / (df.loc[51, 'm [kg/s]'] * (df.loc[51, 'e^PH [kJ/kg]'] - df.loc[52, 'e^PH [kJ/kg]'])))

    ef = {
        'pump': ef_pump,
        'eva': ef_eva,
        'ihx': ef_ihx,
        'exp': ef_exp,
        'cond': np.nan,
        'tot': ef_eva
    }

    epsilon = {
        'pump': epsilon_pump,
        'eva': epsilon_eva,
        'ihx': epsilon_ihx,
        'exp': epsilon_exp,
        'cond': np.nan,
        'tot': exergy_efficiency
    }

    df_comps = pd.DataFrame(index=['pump', 'eva', 'ihx', 'exp', 'cond', 'tot'],
                            columns=['EF [kW]', 'EP [kW]', 'ED [kW]', 'epsilon', 'P [kW]', 'Q [kW]'])

    for k in ['pump', 'eva', 'ihx', 'exp', 'cond', 'tot']:
        df_comps.loc[k, 'epsilon'] = epsilon[k]
        df_comps.loc[k, 'EF [kW]'] = ef[k]
        if df_comps.loc[k, 'EF [kW]'] is not None or df_comps.loc[k, 'epsilon'] is not None:
            df_comps.loc[k, 'EP [kW]'] = df_comps.loc[k, 'EF [kW]'] * df_comps.loc[k, 'epsilon']
        df_comps.loc[k, 'ED [kW]'] = ed[k]

    df_comps.loc['pump', 'P [kW]'] = power_pump
    df_comps.loc['eva', 'P [kW]'] = power_eva
    df_comps.loc['ihx', 'P [kW]'] = power_ihx
    df_comps.loc['exp', 'P [kW]'] = power_exp
    df_comps.loc['cond', 'Q [kW]'] = heat_cond
    df_comps.loc['ihx', 'Q [kW]'] = heat_ihx
    df_comps.loc['eva', 'Q [kW]'] = heat_eva
    df_comps.loc['tot', 'P [kW]'] = - power_pump + power_eva + power_exp + power_ihx

    return df_comps


def set_adex_orc_config(*args):
    # Define all keys with a default value of False
    config_keys = ['pump', 'eva', 'exp', 'ihx', 'cond']
    config = {key: False for key in config_keys}

    # Set the keys provided in args to True
    for arg in args:
        if arg in config:
            config[arg] = True
        else:
            print(f'Warning: {arg} is not a valid key. It has been ignored.')

    # Generate the label
    label_parts = [key for key, value in config.items() if value]
    label = '_'.join(label_parts)

    # Assign special labels for certain conditions
    if label == '':
        label = 'ideal'
    elif all(config.values()):
        label = 'real'

    return config, label


def find_opt_p63(p63_opt_start, min_t_diff_eva_start, config, label, adex=False, output_buffer=None):
    if config['eva']:
        target_min_td_eva = 5
    else:
        target_min_td_eva = 0
    min_t_diff_eva = target_min_td_eva
    p63_opt = p63_opt_start
    tolerance = 1e-3  # relative to min temperature difference
    learning_rate = 5e4  # relative to p63
    diff = abs(min_t_diff_eva_start - target_min_td_eva)
    step = 0

    while diff > tolerance:
        if p63_opt > 38e5-0.5e5:  # because otherwise too close to critical point  # TODO: take pressure of step before
            print('Optimization interrupted because steam pressure to close to critical point.')
            break
        # Adjust p63 based on the difference
        adjustment = -(target_min_td_eva - min_t_diff_eva) * learning_rate
        # adjustment is the smaller, the smaller the difference target_min_td_eva - min_td_eva
        p63_opt += adjustment

        [_, min_t_diff_eva, eta] = orc_simultaneous(p63_opt, print_results=False, label=label, config=config, adex=adex)
        diff_new = abs(min_t_diff_eva - target_min_td_eva)
        if diff_new > diff:
            print(f'Optimization interrupted because diff is increasing (new diff = {round(diff_new,6)}). No local solution found.')
            break  # because otherwise min. temp. diff. not reached (non-linear behaviour & no solution possible)
        diff = diff_new

        step += 1
        print(f'Optimization in progress for {label}: step = {step}, diff = {round(diff,6)}, p63 = {round(p63_opt*1e-5, 4)} bar, efficiency = {round(eta, 4)}.')

    print(f'Optimization completed successfully in {step} steps!\n')
    print(f'Optimal p63: {round(p63_opt*1e-5, 4)}, bar.\n')

    return p63_opt


t0 = 283.15  # K
p0 = 1.013e5  # Pa

'''[config_real, label_real] = set_adex_orc_config('pump', 'eva', 'ihx', 'exp', 'cond')
p63_start = 23e5  # bar
[_, target_diff, _] = orc_simultaneous(p63_start, print_results=True, config=config_real, label=label_real)
p63_opt = find_opt_p63(p63_start, target_diff, config=config_real, label=label_real, adex=False)
[df_opt, _, _] = orc_simultaneous(p63_opt, print_results=True, config=config_real, label=f'optimal {label_real}')
[df_opt, _, _] = orc_simultaneous(p63_opt, print_results=True, config=config_real, label=f'optimal {label_real}')
df_components = exergy_analysis_orc(t0, p0, df_opt)
for col in df_components.columns[:5]:
    df_components[col] = pd.to_numeric(df_components[col], errors='coerce')
df_components.to_csv(f'outputs/adex_orc/orc_comps_real.csv')'''

[config_ideal, label_ideal] = set_adex_orc_config('cond')
p63_start = 29e5  # bar
[df_opt, target_diff, _] = orc_simultaneous(p63_start, print_results=True, config=config_ideal, label=label_ideal, adex=True)
p63_opt = find_opt_p63(p63_start, target_diff, config=config_ideal, label=label_ideal, adex=True)
[df_opt, _, _] = orc_simultaneous(p63_opt, print_results=True, config=config_ideal, label=f'optimal {label_ideal}', adex=True)
df_components = exergy_analysis_orc(t0, p0, df_opt)
for col in df_components.columns[:5]:
    df_components[col] = pd.to_numeric(df_components[col], errors='coerce')
df_components.round(4).to_csv(f'outputs/adex_orc/orc_comps_ideal.csv')

