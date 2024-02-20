import numpy as np
import itertools
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
                      eps_expander_func, eps_expander_deriv, ttd_temperature_func, ttd_temperature_deriv)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def orc_simultaneous(target_p33, print_results, config, label, adex=False, plot=False):

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
    t41 = 140 + 273.15  # input is known
    p41 = 5e5           # input is known
    t42 = 70 + 273.15   # output temperature is set
    m41 = 10            # dimensioning of the system (full load)

    # PRESSURE DROPS
    if config['cond']:  # real cond
        pr_cond_hot = 0.95
    else:
        pr_cond_hot = 1
    if config['ihx']:  # real ihx
        pr_ihx_hot = 0.985
        pr_ihx_cold = 0.985
    else:
        pr_ihx_hot = 1
        pr_ihx_cold = 1
    if config['eva']:  # real eva
        pr_eva_hot = 1
        pr_eva_cold = 0.95
    else:
        pr_eva_hot = 1
        pr_eva_cold = 1

    # AMBIENT
    t0 = 10 + 273.15    # K
    p0 = 1.013e5        # bar

    # HEAT PUMP
    if config['cond']:  # real cond
        ttd_l_cond = 5  # K
    else:
        ttd_l_cond = 0  # K

    if config['ihx']:  # real ihx
        ttd_l_ihx = 5   # K
    else:
        ttd_l_ihx = 0   # K

    if config['eva']:  # real eva
        ttd_u_eva = 5   # K
    else:
        ttd_u_eva = 0   # K

    pr_eva_part_cold = np.cbrt(pr_eva_cold)  # eva pressure drop is split equally (geom, mean) between ECO-EVA-SH
    pr_eva_part_hot = np.cbrt(pr_eva_hot)    # eva pressure drop is split equally (geom, mean) between ECO-EVA-SH

    # TECHNICAL PARAMETERS
    eta_s_pump = 0.85
    eta_s_exp = 0.9

    # PRE-CALCULATION
    t31 = t0 + ttd_l_cond
    t34 = t41 - ttd_u_eva

    # STARTING VALUES
    if config['cond']:  # real cond
        h31 = 220e3
        h35 = 465e3
        h36 = 410e3
    else:
        h31 = 213e3
        h35 = 466e3
        h36 = 405e3
    h32 = 215e3
    h33 = 275e3
    h34 = 508e3
    m31 = 12.75
    h41 = 589e3
    h42 = 293e3
    if config['cond']:  # real cond
        p31 = 5e5
        p35 = 5e5
        p36 = 5e5
    else:
        p31 = 4.14e5
        p35 = 4.14e5
        p36 = 4.14e5
    p32 = target_p33/0.97
    p34 = target_p33*0.97
    p38 = target_p33*0.95
    p39 = target_p33*0.94
    p42 = 5e5
    h48 = 487e3
    h49 = 358e3
    h38 = 327e3
    h39 = 428e3
    p48 = 5e5
    p49 = 5e5

    variables = np.array([h31, p31, h32, h36, p36, p32, p34, h34, p35, h35, h33, h41, h42, p42, m31,
                          h38, h39, h48, h49, p38, p39, p48, p49])
    residual = np.ones(len(variables))

    iter_step = 0

    while np.linalg.norm(residual) > 1e-4:
        # TODO [h31, p31, h32, h36, p36, p32, p34, h34, p35, h35, h33, h41, h42, p42, m31
        # TODO   0    1    2    3    4    5    6    7    8    9    10   11   12   13   14
        #       h38, h39, h48, h49, p38, p39, p48, p49])]
        #        15   16   17   18   19   20   21   22
        # 0
        t31_set = temperature_func(t31, variables[0], variables[1], wf)
        # 1
        p31_calc_cond = x_saturation_func(0, variables[0], variables[1], wf)
        # 2
        if adex and not config['pump']:  # ideal pump
            t32_calc_pump = eps_compressor_func(1, variables[0], variables[1], variables[2], variables[5], wf)
        elif adex and config['pump']:  # real pump for adv. exergy analysis
            t32_calc_pump = eps_compressor_func(epsilon['pump'], variables[0], variables[1], variables[2], variables[5], wf)
        else:  # real pump for base design
            t32_calc_pump = eta_s_compressor_func(eta_s_pump, variables[0], variables[1], variables[2], variables[5], wf)
        # 3
        if adex and not config['ihx']:  # ideal ihx
            t36_set = same_temperature_func(variables[2], variables[5], wf, variables[3], variables[4], wf)
        elif adex and config['ihx']:  # real ihx for adv. exergy analysis
            t36_set = eps_real_ihx_func(epsilon['ihx'], variables[9], variables[8], variables[3], variables[4], wf,
                                        variables[2], variables[5], variables[10], target_p33, wf)
        else:  # real ihx for base design
            t36_set = ttd_temperature_func(ttd_l_ihx, variables[2], variables[5], wf, variables[3], variables[4], wf)
        # 4
        p36_set = pr_func(pr_cond_hot, variables[4], variables[1])
        # 5
        p32_set = pr_func(pr_ihx_cold, variables[5], target_p33)
        # 6
        p34_set = pr_func(pr_eva_cold, target_p33, variables[6])
        # 7
        t34_set = temperature_func(t34, variables[7], variables[6], wf)
        # 8
        p35_set = pr_func(pr_ihx_hot, variables[8], variables[4])
        # 9
        if adex and not config['exp']:  # ideal exp
            t35_calc_exp = eps_expander_func(1, variables[7], variables[6], variables[9], variables[8], wf)
        elif adex and config['exp']:  # real exp for adv. exergy analysis
            t35_calc_exp = eps_expander_func(epsilon['exp'], variables[7], variables[6], variables[9], variables[8], wf)
        else:  # real exp for base design
            t35_calc_exp = eta_s_expander_func(eta_s_exp, variables[7], variables[6], variables[9], variables[8], wf)
        # 10
        if adex and not config['ihx']:  # ideal ihx
            if not config['eva']:  # ideal eva
                t33_calc = ideal_ihx_entropy_func(variables[9], variables[8], variables[3], variables[4], wf,
                                                  variables[2], variables[5], variables[10], target_p33, wf)
            else:  # real eva for adv. exergy analysis
                t33_calc = eps_real_he_func(epsilon['eva'], variables[11], p41, variables[12], variables[13], m41, fluid_tes,
                                            variables[10], target_p33, variables[7], variables[6], variables[14], wf)
        else:  # real ihx for base design or for adv. exergy analysis
            t33_calc = ihx_func(variables[9], variables[3], variables[2], variables[10])
        # 11
        t41_set = temperature_func(t41, variables[11], p41, fluid_tes)
        # 12
        t42_set = temperature_func(t42, variables[12], variables[13], fluid_tes)
        # 13
        p42_set = pr_func(pr_eva_hot, p41, variables[13])
        # 14
        if adex and not config['eva']:  # ideal eva
            m31_calc_eva = ideal_he_entropy_func(variables[11], p41, variables[12], variables[13], m41, fluid_tes,
                                                 variables[10], target_p33, variables[7], variables[6], variables[14], wf)
        else:  # real eva for base design or for adv. exergy analysis
            m31_calc_eva = he_func(m41, variables[11], variables[12], variables[14], variables[10], variables[7])
        # 15
        eva_eco_outlet_sat = x_saturation_func(0, variables[15], variables[19], wf)
        # 16
        eva_eco_en_bal = he_func(m41, variables[18], variables[12], variables[14], variables[10], variables[15])
        # 17
        eva_sh_inlet_sat = x_saturation_func(1, variables[16], variables[20], wf)
        # 18
        eva_sh_en_bal = he_func(m41, variables[11], variables[17], variables[14], variables[16], variables[7])
        # 19
        p38_set = pr_func(pr_eva_part_cold, target_p33, variables[19])
        # 20
        p39_set = pr_func(pr_eva_part_cold, variables[19], variables[20])
        # 21
        p48_set = pr_func(pr_eva_part_hot, p41, variables[21])
        # 22
        p49_set = pr_func(pr_eva_part_hot, variables[21], variables[22])

        residual = np.array([t31_set, p31_calc_cond, t32_calc_pump, t36_set, p36_set, p32_set, p34_set, t34_set,
                             p35_set, t35_calc_exp, t33_calc, t41_set, t42_set, p42_set, m31_calc_eva,
                             eva_eco_outlet_sat, eva_eco_en_bal, eva_sh_inlet_sat, eva_sh_en_bal, p38_set, p39_set,
                             p48_set, p49_set])
        jacobian = np.zeros((len(variables), len(variables)))

        # 0
        t31_set_j = temperature_deriv(t31, variables[0], variables[1], wf)
        # 1
        p31_calc_cond_j = x_saturation_deriv(0, variables[0], variables[1], wf)
        # 2
        if adex and not config['pump']:  # ideal pump
            t32_calc_pump_j = eps_compressor_deriv(1, variables[0], variables[1], variables[2], variables[5], wf)
        elif adex and config['pump']:  # real pump for adv. exergy analysis
            t32_calc_pump_j = eps_compressor_deriv(epsilon['pump'], variables[0], variables[1], variables[2], variables[5], wf)
        else:  # real pump for base design
            t32_calc_pump_j = eta_s_compressor_deriv(eta_s_pump, variables[0], variables[1], variables[2], variables[5], wf)
        # 3
        if adex and not config['ihx']:  # ideal ihx
            t36_set_j = same_temperature_deriv(variables[2], variables[5], wf, variables[3], variables[4], wf)
        elif adex and config['ihx']:  # real ihx for adv. exergy analysis
            t36_set_j = eps_real_ihx_deriv(epsilon['ihx'], variables[9], variables[8], variables[3], variables[4], wf,
                                           variables[2], variables[5], variables[10], target_p33, wf)
        else:  # real ihx for base design
            t36_set_j = ttd_temperature_deriv(ttd_l_ihx, variables[2], variables[5], wf, variables[3], variables[4], wf)
        # 4
        p36_set_j = pr_deriv(pr_cond_hot, variables[4], variables[1])
        # 5
        p32_set_j = pr_deriv(pr_ihx_cold, variables[5], target_p33)
        # 6
        p34_set_j = pr_deriv(pr_eva_cold, target_p33, variables[6])
        # 7
        t34_set_j = temperature_deriv(t34, variables[7], variables[6], wf)
        # 8
        p35_set_j = pr_deriv(pr_ihx_hot, variables[8], variables[4])
        # 9
        if adex and not config['exp']:  # ideal exp
            t35_calc_exp_j = eps_expander_deriv(1, variables[7], variables[6], variables[9], variables[8], wf)
        elif adex and config['exp']:  # real exp for adv. exergy analysis
            t35_calc_exp_j = eps_expander_deriv(epsilon['exp'], variables[7], variables[6], variables[9], variables[8], wf)
        else:  # real exp for base design
            t35_calc_exp_j = eta_s_expander_deriv(eta_s_exp, variables[7], variables[6], variables[9], variables[8], wf)
        # 10
        if adex and not config['ihx']:  # ideal ihx
            if not config['eva']:  # ideal eva
                t33_calc_j = ideal_ihx_entropy_deriv(variables[9], variables[8], variables[3], variables[4], wf,
                                                     variables[2], variables[5], variables[10], target_p33, wf)
            else:  # real eva for adv. exergy analysis
                t33_calc_j = eps_real_he_deriv(epsilon['eva'], variables[11], p41, variables[12], variables[13], m41, fluid_tes,
                                               variables[10], target_p33, variables[7], variables[6], variables[14], wf)
        else:  # real ihx for base design or for adv. exergy analysis
            t33_calc_j = ihx_deriv(variables[9], variables[3], variables[2], variables[10])
        # 11
        t41_set_j = temperature_deriv(t41, variables[11], p41, fluid_tes)
        # 12
        t42_set_j = temperature_deriv(t42, variables[12], variables[13], fluid_tes)
        # 13
        p42_set_j = pr_deriv(pr_eva_hot, p41, variables[13])
        # 14
        if adex and not config['eva']:  # ideal eva
            m31_calc_eva_j = ideal_he_entropy_deriv(variables[11], p41, variables[12], variables[13], m41, fluid_tes,
                                                    variables[10], target_p33, variables[7], variables[6], variables[14], wf)
        else:  # real eva for base design or for adv. exergy analysis
            m31_calc_eva_j = he_deriv(m41, variables[11], variables[12], variables[14], variables[10], variables[7])
        # 15
        eva_eco_outlet_sat_j = x_saturation_deriv(0, variables[15], variables[19], wf)
        # 16
        eva_eco_en_bal_j = he_deriv(m41, variables[18], variables[12], variables[14], variables[10], variables[15])
        # 17
        eva_sh_inlet_sat_j = x_saturation_deriv(1, variables[16], variables[20], wf)
        # 18
        eva_sh_en_bal_j = he_deriv(m41, variables[11], variables[17], variables[14], variables[16], variables[7])
        # 19
        p38_set_j = pr_deriv(pr_eva_part_cold, target_p33, variables[19])
        # 20
        p39_set_j = pr_deriv(pr_eva_part_cold, variables[19], variables[20])
        # 21
        p48_set_j = pr_deriv(pr_eva_part_hot, p41, variables[21])
        # 22
        p49_set_j = pr_deriv(pr_eva_part_hot, variables[21], variables[22])

        # TODO [h31, p31, h32, h36, p36, p32, p34, h34, p35, h35, h33, h41, h42, p42, m31
        # TODO   0    1    2    3    4    5    6    7    8    9    10   11   12   13   14
        #       h38, h39, h48, h49, p38, p39, p48, p49])]
        #        15   16   17   18   19   20   21   22
        jacobian[0, 0] = t31_set_j['h']  # derivative of t31_set with respect to h31
        jacobian[0, 1] = t31_set_j['p']  # derivative of t31_set with respect to p31
        jacobian[1, 0] = p31_calc_cond_j['h']  # derivative of p31_calc_cond with respect to h31
        jacobian[1, 1] = p31_calc_cond_j['p']  # derivative of p31_calc_cond with respect to p31
        jacobian[2, 0] = t32_calc_pump_j['h_1']  # derivative of t32_calc_pump with respect to h31
        jacobian[2, 1] = t32_calc_pump_j['p_1']  # derivative of t32_calc_pump with respect to p31
        jacobian[2, 2] = t32_calc_pump_j['h_2']  # derivative of t32_calc_pump with respect to h32
        jacobian[2, 5] = t32_calc_pump_j['p_2']  # derivative of t32_calc_pump with respect to p32
        if adex and not config['ihx']:
            jacobian[3, 2] = t36_set_j['h_copy']  # derivative of t36_set_j with respect to h35
            jacobian[3, 5] = t36_set_j['p_copy']  # derivative of t36_set_j with respect to p35
            jacobian[3, 3] = t36_set_j['h_paste']  # derivative of t36_set_j with respect to h36
            jacobian[3, 4] = t36_set_j['p_paste']  # derivative of t36_set_j with respect to p36
        elif adex and config['ihx']:
            jacobian[3, 9] = t36_set_j['h_hot_in']  # derivative of t36_set_j with respect to h35
            jacobian[3, 8] = t36_set_j['p_hot_in']  # derivative of t36_set_j with respect to p35
            jacobian[3, 3] = t36_set_j['h_hot_out']  # derivative of t36_set_j with respect to h36
            jacobian[3, 4] = t36_set_j['p_hot_out']  # derivative of t36_set_j with respect to p36
            jacobian[3, 2] = t36_set_j['h_cold_in']  # derivative of t36_set_j with respect to h32
            jacobian[3, 5] = t36_set_j['p_cold_in']  # derivative of t36_set_j with respect to p32
            jacobian[3, 10] = t36_set_j['h_cold_out']  # derivative of t36_set_j with respect to h33
        else:
            jacobian[3, 2] = t36_set_j['h_copy']  # derivative of t36_set_j with respect to h35
            jacobian[3, 5] = t36_set_j['p_copy']  # derivative of t36_set_j with respect to p35
            jacobian[3, 3] = t36_set_j['h_paste']  # derivative of t36_set_j with respect to h36
            jacobian[3, 4] = t36_set_j['p_paste']  # derivative of t36_set_j with respect to p36
        jacobian[4, 4] = p36_set_j['p_1']  # derivative of p36_set with respect to p36
        jacobian[4, 1] = p36_set_j['p_2']  # derivative of p36_set with respect to p31
        jacobian[5, 5] = p32_set_j['p_1']  # derivative of p32_set with respect to p33
        jacobian[6, 6] = p34_set_j['p_2']  # derivative of p34_set with respect to p34
        jacobian[7, 7] = t34_set_j['h']  # derivative of t34_set with respect to h34
        jacobian[7, 6] = t34_set_j['p']  # derivative of t34_set with respect to p34
        jacobian[8, 8] = p35_set_j['p_1']  # derivative of p34_set with respect to p35
        jacobian[8, 4] = p35_set_j['p_2']  # derivative of p34_set with respect to p36
        jacobian[9, 7] = t35_calc_exp_j['h_1']  # derivative of t35_calc_exp with respect to h34
        jacobian[9, 6] = t35_calc_exp_j['p_1']  # derivative of t35_calc_exp with respect to p34
        jacobian[9, 9] = t35_calc_exp_j['h_2']  # derivative of t35_calc_exp with respect to h35
        jacobian[9, 8] = t35_calc_exp_j['p_2']  # derivative of t35_calc_exp with respect to p35
        if adex and not config['ihx']:  # ideal ihx
            if not config['eva']:  # ideal eva
                jacobian[10, 9] = t33_calc_j['h_hot_in']  # derivative of t33_calc_ihx_j with respect to h35
                jacobian[10, 8] = t33_calc_j['p_hot_in']  # derivative of t33_calc_ihx_j with respect to p35
                jacobian[10, 3] = t33_calc_j['h_hot_out']  # derivative of t33_calc_ihx_j with respect to h36
                jacobian[10, 4] = t33_calc_j['p_hot_out']  # derivative of t33_calc_ihx_j with respect to p36
                jacobian[10, 2] = t33_calc_j['h_cold_in']  # derivative of t33_calc_ihx_j with respect to h32
                jacobian[10, 5] = t33_calc_j['p_cold_in']  # derivative of t33_calc_ihx_j with respect to p32
                jacobian[10, 10] = t33_calc_j['h_cold_out']  # derivative of t33_calc_ihx_j with respect to h33
            else:
                jacobian[10, 11] = t33_calc_j['h_hot_in']  # derivative of m31_calc_eva with respect to h41
                jacobian[10, 12] = t33_calc_j['h_hot_out']  # derivative of m31_calc_eva with respect to h42
                jacobian[10, 13] = t33_calc_j['p_hot_out']  # derivative of m31_calc_eva with respect to p42
                jacobian[10, 10] = t33_calc_j['h_cold_in']  # derivative of m31_calc_eva with respect to h33
                jacobian[10, 7] = t33_calc_j['h_cold_out']  # derivative of m31_calc_eva with respect to h34
                jacobian[10, 6] = t33_calc_j['p_cold_out']  # derivative of m31_calc_eva with respect to p34
                jacobian[10, 14] = t33_calc_j['m_cold']  # derivative of m31_calc_eva with respect to m31
        else:
            jacobian[10, 9] = t33_calc_j['h_1']  # derivative of t33_calc_ihx with respect to h35
            jacobian[10, 3] = t33_calc_j['h_2']  # derivative of t33_calc_ihx with respect to h36
            jacobian[10, 2] = t33_calc_j['h_3']  # derivative of t33_calc_ihx with respect to h32
            jacobian[10, 10] = t33_calc_j['h_4']  # derivative of t33_calc_ihx with respect to h33
        jacobian[11, 11] = t41_set_j['h']  # derivative of t41_set with respect to h41
        jacobian[12, 12] = t42_set_j['h']  # derivative of t42_set with respect to h42
        jacobian[12, 13] = t42_set_j['p']  # derivative of t42_set with respect to p42
        jacobian[13, 13] = p42_set_j['p_2']  # derivative of p42_set with respect to p42
        if adex and not config['eva']:
            jacobian[14, 11] = m31_calc_eva_j['h_hot_in']  # derivative of m31_calc_eva with respect to h41
            jacobian[14, 12] = m31_calc_eva_j['h_hot_out']  # derivative of m31_calc_eva with respect to h42
            jacobian[14, 13] = m31_calc_eva_j['p_hot_out']  # derivative of m31_calc_eva with respect to p42
            jacobian[14, 10] = m31_calc_eva_j['h_cold_in']  # derivative of m31_calc_eva with respect to h33
            jacobian[14, 7] = m31_calc_eva_j['h_cold_out']  # derivative of m31_calc_eva with respect to h34
            jacobian[14, 6] = m31_calc_eva_j['p_cold_out']  # derivative of m31_calc_eva with respect to p34
            jacobian[14, 14] = m31_calc_eva_j['m_cold']  # derivative of m31_calc_eva with respect to m31
        else:
            jacobian[14, 11] = m31_calc_eva_j['h_1']  # derivative of m31_calc_eva with respect to h41
            jacobian[14, 12] = m31_calc_eva_j['h_2']  # derivative of m31_calc_eva with respect to h42
            jacobian[14, 14] = m31_calc_eva_j['m_cold']  # derivative of m31_calc_eva with respect to m31
            jacobian[14, 10] = m31_calc_eva_j['h_3']  # derivative of m31_calc_eva with respect to h33
            jacobian[14, 7] = m31_calc_eva_j['h_4']  # derivative of m31_calc_eva with respect to h34
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
        jacobian[19, 19] = p38_set_j['p_2']
        jacobian[20, 19] = p39_set_j['p_1']
        jacobian[20, 20] = p39_set_j['p_2']
        jacobian[21, 21] = p48_set_j['p_2']
        jacobian[22, 21] = p49_set_j['p_1']
        jacobian[22, 22] = p49_set_j['p_2']
        # Save the DataFrame as a CSV file
        # pd.DataFrame(jacobian).round(4).to_csv('jacobian_orc.csv', index=False)

        variables -= np.linalg.inv(jacobian).dot(residual)
        iter_step += 1
        cond_number = np.linalg.cond(jacobian)
        # print('Condition number:', cond_number, '- residual:', np.linalg.norm(residual), '- step:', iter_step)
        # print(np.linalg.norm(residual))
    # TODO [h31, p31, h32, h36, p36, p32, p34, h34, p35, h35, h33, h41, h42, p42, m31
    # TODO   0    1    2    3    4    5    6    7    8    9    10   11   12   13   14
    #       h38, h39, h48, h49, p38, p39, p48, p49])]
    #        15   16   17   18   19   20   21   22
    p31 = variables[1]
    p32 = variables[5]
    p33 = target_p33
    p34 = variables[6]
    p35 = variables[8]
    p36 = variables[4]
    p41 = p41
    p42 = variables[13]
    p38 = variables[19]
    p39 = variables[20]
    p48 = variables[21]
    p49 = variables[22]

    h31 = variables[0]
    h32 = variables[2]
    h33 = variables[10]
    h34 = variables[7]
    h35 = variables[9]
    h36 = variables[3]
    h41 = variables[11]
    h42 = variables[12]
    h38 = variables[15]
    h39 = variables[16]
    h48 = variables[17]
    h49 = variables[18]

    t31 = PSI('T', 'H', h31, 'P', p31, wf)
    t32 = PSI('T', 'H', h32, 'P', p32, wf)
    t33 = PSI('T', 'H', h33, 'P', p33, wf)
    t34 = PSI('T', 'H', h34, 'P', p34, wf)
    t35 = PSI('T', 'H', h35, 'P', p35, wf)
    t36 = PSI('T', 'H', h36, 'P', p36, wf)
    t38 = PSI('T', 'H', h38, 'P', p38, wf)
    t39 = PSI('T', 'H', h39, 'P', p39, wf)
    t41 = PSI('T', 'H', h41, 'P', p41, fluid_tes)
    t42 = PSI('T', 'H', h42, 'P', p42, fluid_tes)
    t48 = PSI('T', 'H', h48, 'P', p48, fluid_tes)
    t49 = PSI('T', 'H', h49, 'P', p49, fluid_tes)

    s31 = PSI('S', 'H', h31, 'P', p31, wf)
    s32 = PSI('S', 'H', h32, 'P', p32, wf)
    s33 = PSI('S', 'H', h33, 'P', p33, wf)
    s34 = PSI('S', 'H', h34, 'P', p34, wf)
    s35 = PSI('S', 'H', h35, 'P', p35, wf)
    s36 = PSI('S', 'H', h36, 'P', p36, wf)
    s38 = PSI('S', 'H', h38, 'P', p38, wf)
    s39 = PSI('S', 'H', h39, 'P', p39, wf)
    s41 = PSI('S', 'H', h41, 'P', p41, fluid_tes)
    s42 = PSI('S', 'H', h42, 'P', p42, fluid_tes)
    s48 = PSI('S', 'H', h48, 'P', p48, fluid_tes)
    s49 = PSI('S', 'H', h49, 'P', p49, fluid_tes)

    m31 = variables[14]

    df_streams = pd.DataFrame(index=[31, 32, 33, 34, 35, 36, 41, 42, 38, 39, 48, 49],
                      columns=['m [kg/s]', 'T [°C]', 'h [kJ/kg]', 'p [bar]', 's [J/kgK]', 'fluid'])
    df_streams.loc[31] = [m31, t31, h31, p31, s31, wf]
    df_streams.loc[32] = [m31, t32, h32, p32, s32, wf]
    df_streams.loc[33] = [m31, t33, h33, p33, s33, wf]
    df_streams.loc[34] = [m31, t34, h34, p34, s34, wf]
    df_streams.loc[35] = [m31, t35, h35, p35, s35, wf]
    df_streams.loc[36] = [m31, t36, h36, p36, s36, wf]
    df_streams.loc[38] = [m31, t38, h38, p38, s38, wf]
    df_streams.loc[39] = [m31, t39, h39, p39, s39, wf]
    df_streams.loc[41] = [m41, t41, h41, p41, s41, fluid_tes]
    df_streams.loc[42] = [m41, t42, h42, p42, s42, fluid_tes]
    df_streams.loc[48] = [m41, t48, h48, p48, s48, fluid_tes]
    df_streams.loc[49] = [m41, t49, h49, p49, s49, fluid_tes]

    df_streams['T [°C]'] = df_streams['T [°C]'] - 273.15
    df_streams['h [kJ/kg]'] = df_streams['h [kJ/kg]'] * 1e-3
    df_streams['p [bar]'] = df_streams['p [bar]'] * 1e-5

    if print_results:
        print('-------------------------------------------------------------\n', 'case:', label)
        print(df_streams.iloc[:, :5])
        print('-------------------------------------------------------------')

    power_pump = df_streams.loc[31, 'm [kg/s]'] * (df_streams.loc[32, 'h [kJ/kg]'] - df_streams.loc[31, 'h [kJ/kg]'])
    power_exp = df_streams.loc[31, 'm [kg/s]'] * (df_streams.loc[34, 'h [kJ/kg]'] - df_streams.loc[35, 'h [kJ/kg]'])
    heat_eva = df_streams.loc[41, 'm [kg/s]'] * (df_streams.loc[41, 'h [kJ/kg]'] - df_streams.loc[42, 'h [kJ/kg]'])
    heat_cond = df_streams.loc[31, 'm [kg/s]'] * (df_streams.loc[36, 'h [kJ/kg]'] - df_streams.loc[31, 'h [kJ/kg]'])
    power_ihx = df_streams.loc[31, 'm [kg/s]'] * (df_streams.loc[35, 'h [kJ/kg]'] - df_streams.loc[36, 'h [kJ/kg]'] +
                                                  df_streams.loc[32, 'h [kJ/kg]'] - df_streams.loc[33, 'h [kJ/kg]'])
    power_eva = (df_streams.loc[41, 'm [kg/s]'] * (df_streams.loc[41, 'h [kJ/kg]'] - df_streams.loc[42, 'h [kJ/kg]'])
                 + df_streams.loc[31, 'm [kg/s]'] * (df_streams.loc[33, 'h [kJ/kg]'] - df_streams.loc[34, 'h [kJ/kg]']))
    if abs(heat_eva+power_pump-power_exp-heat_cond-power_ihx-power_eva) > 1e-4:
        print('Energy balances are not fulfilled! :(')

    if plot:
        qt_diagram(df_streams, 'IHX', 35, 36, 32, 33,
                    ttd_l_ihx, 'ORC', plot=plot, case=f'{label} with ttd_l_ihx={ttd_l_ihx}')
        [min_td_eva, _] = qt_diagram(df_streams, 'EVA', 41, 42, 33, 34,
                                     ttd_u_eva, 'ORC', plot=plot, case=f'{label} with ttd_u_eva={ttd_u_eva}')

    t_pinch_eva_sh = t49 - t38

    efficiency = (power_exp+power_eva+power_eva-power_pump)/heat_eva

    return df_streams, t_pinch_eva_sh, efficiency


def exergy_analysis_orc(t0, p0, df):

    h0_wf = PSI('H', 'T', t0, 'P', p0, df.loc[31, 'fluid']) * 1e-3
    s0_wf = PSI('S', 'T', t0, 'P', p0, df.loc[31, 'fluid'])
    h0_tes = PSI('H', 'T', t0, 'P', p0, df.loc[41, 'fluid']) * 1e-3
    s0_tes = PSI('S', 'T', t0, 'P', p0, df.loc[41, 'fluid'])

    for i in [31, 32, 33, 34, 35, 36, 38, 39]:
        df.loc[i, 'e^PH [kJ/kg]'] = df.loc[i, 'h [kJ/kg]'] - h0_wf - t0 * (df.loc[i, 's [J/kgK]'] - s0_wf) * 1e-3
    for i in [41, 42, 48, 49]:
        df.loc[i, 'e^PH [kJ/kg]'] = df.loc[i, 'h [kJ/kg]'] - h0_tes - t0 * (df.loc[i, 's [J/kgK]'] - s0_tes) * 1e-3

    # ED_PUMP = T0 * (s32 - s31)
    ed_pump = t0 * (df.loc[31, 'm [kg/s]'] * (df.loc[32, 's [J/kgK]'] - df.loc[31, 's [J/kgK]'])) * 1e-3
    # ED_EVA = T0 * (s33 - s32 + s42 - s41)
    ed_eva = t0 * (df.loc[31, 'm [kg/s]'] * (df.loc[34, 's [J/kgK]'] - df.loc[33, 's [J/kgK]']) +
                   df.loc[41, 'm [kg/s]'] * (df.loc[42, 's [J/kgK]'] - df.loc[41, 's [J/kgK]'])) * 1e-3
    # ED_IHX = T0 * (s33 - s32 + s36 - s35)
    ed_ihx = t0 * (df.loc[31, 'm [kg/s]'] * (df.loc[33, 's [J/kgK]'] - df.loc[32, 's [J/kgK]']) +
                   df.loc[31, 'm [kg/s]'] * (df.loc[36, 's [J/kgK]'] - df.loc[35, 's [J/kgK]'])) * 1e-3
    # ED_EXP = T0 * (s35 - s34)
    ed_exp = t0 * (df.loc[31, 'm [kg/s]'] * (df.loc[35, 's [J/kgK]'] - df.loc[34, 's [J/kgK]'])) * 1e-3
    # ED_EVA = Q * (1 - T0/Tb)
    temp_boundary = ((df.loc[31, 'h [kJ/kg]'] - df.loc[36, 'h [kJ/kg]'])
                     / (df.loc[31, 's [J/kgK]'] - df.loc[36, 's [J/kgK]']) * 1e3)
    ed_cond = (df.loc[31, 'm [kg/s]'] * (df.loc[36, 'h [kJ/kg]'] - df.loc[31, 'h [kJ/kg]'])) * (
                1 - t0 / temp_boundary)

    ed = {
        'pump': ed_pump,
        'eva': ed_eva,
        'ihx': ed_ihx,
        'exp': ed_exp,
        'cond': ed_cond,
        'tot': ed_pump+ed_cond+ed_ihx+ed_exp+ed_eva
    }

    ef_pump = df.loc[31, 'm [kg/s]'] * (df.loc[32, 'h [kJ/kg]'] - df.loc[31, 'h [kJ/kg]'])  # inlet power
    ef_ihx = df.loc[31, 'm [kg/s]'] * (df.loc[35, 'e^PH [kJ/kg]'] - df.loc[36, 'e^PH [kJ/kg]'])
    ef_eva = df.loc[41, 'm [kg/s]'] * (df.loc[41, 'e^PH [kJ/kg]'] - df.loc[42, 'e^PH [kJ/kg]'])
    ef_exp = df.loc[31, 'm [kg/s]'] * (df.loc[34, 'e^PH [kJ/kg]'] - df.loc[35, 'e^PH [kJ/kg]'])

    epsilon_pump = 1 - ed_pump / ef_pump
    epsilon_eva = 1 - ed_eva / ef_eva
    epsilon_exp = 1 - ed_exp / ef_exp
    epsilon_ihx = 1 - ed_ihx / ef_ihx

    heat_eva = df.loc[41, 'm [kg/s]'] * (df.loc[41, 'h [kJ/kg]'] - df.loc[42, 'h [kJ/kg]'])
    heat_cond = df.loc[31, 'm [kg/s]'] * (df.loc[36, 'h [kJ/kg]'] - df.loc[31, 'h [kJ/kg]'])
    heat_ihx = df.loc[31, 'm [kg/s]'] * (df.loc[33, 'h [kJ/kg]'] - df.loc[32, 'h [kJ/kg]'])
    power_pump = df.loc[31, 'm [kg/s]'] * (df.loc[32, 'h [kJ/kg]'] - df.loc[31, 'h [kJ/kg]'])
    power_exp = df.loc[31, 'm [kg/s]'] * (df.loc[34, 'h [kJ/kg]'] - df.loc[35, 'h [kJ/kg]'])
    power_eva = (df.loc[41, 'm [kg/s]'] * (df.loc[41, 'h [kJ/kg]'] - df.loc[42, 'h [kJ/kg]'])
                 + df.loc[31, 'm [kg/s]'] * (df.loc[33, 'h [kJ/kg]'] - df.loc[34, 'h [kJ/kg]']))
    power_ihx = df.loc[31, 'm [kg/s]'] * (df.loc[35, 'h [kJ/kg]'] - df.loc[36, 'h [kJ/kg]'] +
                                          df.loc[32, 'h [kJ/kg]'] - df.loc[33, 'h [kJ/kg]'])

    exergy_efficiency = ((- power_pump + power_ihx + power_exp + power_eva)
                         / (df.loc[41, 'm [kg/s]'] * (df.loc[41, 'e^PH [kJ/kg]'] - df.loc[42, 'e^PH [kJ/kg]'])))

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


def find_opt_p33(p33_opt_start, min_t_diff_eva_start, config, label, adex=False, output_buffer=None):
    if config['eva']:
        target_min_td_eva = 5
    else:
        target_min_td_eva = 0
    min_t_diff_eva = target_min_td_eva
    p33_opt = p33_opt_start
    tolerance = 1e-3  # relative to min temperature difference
    learning_rate = 5e4  # relative to p33
    diff = min_t_diff_eva_start - target_min_td_eva
    step = 0

    while abs(diff) > tolerance:
        if p33_opt > 40.5e5:  # because otherwise too close to critical point  # TODO: take pressure of step before
            print('Optimization interrupted because steam above critical point.')
            break
        # Adjust p33 based on the difference
        adjustment = -(target_min_td_eva - min_t_diff_eva) * learning_rate
        # adjustment is the smaller, the smaller the difference target_min_td_eva - min_td_eva
        p33_opt += adjustment

        [_, min_t_diff_eva, eta] = orc_simultaneous(p33_opt, print_results=False, label=label, config=config, adex=adex)
        diff_new = abs(min_t_diff_eva - target_min_td_eva)
        if diff_new > diff:
            print(f'Optimization interrupted because diff is increasing (new diff = {round(diff_new,6)}). No local solution found.')
            break  # because otherwise min. temp. diff. not reached (non-linear behaviour & no solution possible)
        diff = diff_new

        step += 1
        print(f'Optimization in progress for {label}: step = {step}, diff = {round(diff,6)}, p33 = {round(p33_opt*1e-5, 4)} bar, efficiency = {round(eta, 4)}.')

    print(f'Optimization completed successfully in {step} steps!\n')
    print(f'Optimal p33: {round(p33_opt*1e-5, 4)}, bar.\n')

    return p33_opt


t0 = 283.15  # K
p0 = 1.013e5  # Pa

'''[config_real, label_real] = set_adex_orc_config('pump', 'eva', 'ihx', 'exp', 'cond')
p33_start = 23e5  # bar
[_, target_diff, _] = orc_simultaneous(p33_start, print_results=True, config=config_real, label=label_real)
p33_opt = find_opt_p33(p33_start, target_diff, config=config_real, label=label_real, adex=False)
[df_opt, _, _] = orc_simultaneous(p33_opt, print_results=True, config=config_real, label=f'optimal {label_real}')
df_components = exergy_analysis_orc(t0, p0, df_opt)
for col in df_components.columns[:5]:
    df_components[col] = pd.to_numeric(df_components[col], errors='coerce')
df_components.to_csv(f'outputs/adex_orc/orc_comps_real.csv')

[config_ideal, label_ideal] = set_adex_orc_config()
p33_start = 28e5  # bar
[df_opt, target_diff, _] = orc_simultaneous(p33_start, print_results=True, config=config_ideal, label=label_ideal, adex=True)
p33_opt = find_opt_p33(p33_start, target_diff, config=config_ideal, label=label_ideal, adex=True)
[df_opt, _, _] = orc_simultaneous(p33_opt, print_results=True, config=config_ideal, label=f'optimal {label_ideal}', adex=True, plot=True)
df_components = exergy_analysis_orc(t0, p0, df_opt)
for col in df_components.columns[:5]:
    df_components[col] = pd.to_numeric(df_components[col], errors='coerce')
df_components.round(4).to_csv(f'outputs/adex_orc/orc_comps_ideal.csv')'''

'''[config_real, label_real] = set_adex_orc_config('pump', 'eva', 'ihx', 'exp', 'cond')
p33_start = 23.6e5  # bar
[_, target_diff, _] = orc_simultaneous(p33_start, print_results=True, config=config_real, label=label_real)
p33_opt = find_opt_p33(p33_start, target_diff, config=config_real, label=label_real, adex=False)
[df_opt, _, _] = orc_simultaneous(p33_opt, print_results=True, config=config_real, label=f'optimal {label_real}')
df_components = exergy_analysis_orc(t0, p0, df_opt)
for col in df_components.columns[:5]:
    df_components[col] = pd.to_numeric(df_components[col], errors='coerce')
df_components.to_csv(f'outputs/adex_orc/orc_comps_real.csv')

p = 27.5e5

[config_real, label_real] = set_adex_orc_config()
[_, target_diff, _] = orc_simultaneous(p, print_results=True, config=config_real, label=label_real, plot=True)
print(target_diff)

[config_real, label_real] = set_adex_orc_config('ihx', 'cond')
[_, target_diff, _] = orc_simultaneous(p, print_results=True, config=config_real, label=label_real, plot=True)
print(target_diff)'''


# [config_real, label_real] = set_adex_orc_config('pump', 'exp')
# [_, target_diff, _] = orc_simultaneous(27.5e5, print_results=True, config=config_real, label=label_real, plot=True)

'''for start_value in range(200, 300, 1):
    try:
        [config_real, label_real] = set_adex_orc_config('cond')
        try:
            [_, _, eff] = orc_simultaneous(start_value*1e4, print_results=True, config=config_real, label=label_real, plot=False)
        except Exception as e:
            print(f"An error occurred in orc_simultaneous: {e}")
    except Exception as e:
        print(f"An error occurred in set_adex_orc_config: {e}")'''


'''[config_real, label_real] = set_adex_orc_config('pump', 'eva', 'ihx', 'exp', 'cond')
p33_start = 23.6e5  # bar
[_, target_diff, _] = orc_simultaneous(p33_start, print_results=True, config=config_real, label=label_real)

[config_ideal, label_ideal] = set_adex_orc_config()
p33_start = 23.6e5  # bar
[_, target_diff, _] = orc_simultaneous(p33_start, print_results=True, config=config_ideal, label=label_ideal)'''

p33_start = 40.5e5

components = ['pump', 'eva', 'ihx', 'exp', 'cond']
for component in components:
    config_i, label_i = set_adex_orc_config(component)
    [_, diff, eta] = orc_simultaneous(p33_start, print_results=True, config=config_i, label=label_i, adex=True, plot=True)
    print(eta)
    if component == 'eva':
        minimum = 5
    else:
        minimum = 0
    if diff < minimum:
        print('NOT GOOD')
        print(diff)

# Advanced Exergy Analysis -- pair of components
component_pairs = list(itertools.combinations(components, 2))
for pair in component_pairs:
    config_i, label_i = set_adex_orc_config(*pair)
    [_, diff, eta] = orc_simultaneous(p33_start, print_results=True, config=config_i, label=label_i, adex=True, plot=True)
    print(eta)
    if component == 'eva':
        minimum = 5
    else:
        minimum = 0
    if diff < minimum:
        print('NOT GOOD')
        print(diff)


[config_real, label_real] = set_adex_orc_config('pump', 'eva', 'ihx', 'exp', 'cond')
p33_start = 40.5e5  # bar
[_, target_diff, _] = orc_simultaneous(p33_start, print_results=True, config=config_real, label=label_real)
p33_opt = find_opt_p33(p33_start, target_diff, config=config_real, label=label_real, adex=False)
[df_opt, diff, eta] = orc_simultaneous(p33_opt, print_results=True, config=config_real, label=f'optimal {label_real}', plot=True)
df_components = exergy_analysis_orc(t0, p0, df_opt)
for col in df_components.columns[:5]:
    df_components[col] = pd.to_numeric(df_components[col], errors='coerce')
df_components.to_csv(f'outputs/adex_orc/orc_comps_real.csv')
print(eta)
print(diff)
