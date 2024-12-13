import numpy as np
from CoolProp.CoolProp import PropsSI as PSI
import itertools
import multiprocessing
import time
import io
import json
import pandas as pd
from functions import (pr_func, pr_deriv, eta_s_compressor_func, eta_s_compressor_deriv, turbo_func, turbo_deriv,
                       he_func, he_deriv, ihx_func, ihx_deriv, temperature_func, temperature_deriv, valve_func,
                       valve_deriv, x_saturation_func, x_saturation_deriv, qt_diagram, eps_compressor_func,
                       eps_compressor_deriv, eps_real_he_func, eps_real_he_deriv, eps_real_ihx_func, eps_real_ihx_deriv,
                       simple_he_func, simple_he_deriv, ideal_ihx_entropy_func, ideal_ihx_entropy_deriv,
                       ideal_he_entropy_func, ideal_he_entropy_deriv, ideal_valve_entropy_func,
                       ideal_valve_entropy_deriv, he_with_p_func, he_with_p_deriv, same_temperature_func,
                       same_temperature_deriv, eta_s_expander_func, eta_s_expander_deriv, ttd_func, ttd_deriv,
                       eps_expander_func, eps_expander_deriv, ttd_temperature_func, ttd_temperature_deriv)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def solve_orc(target_p33, print_results, config, label, adex=False, plot=False):
    """
    Simulates the performance of an Organic Rankine Cycle (ORC) system under specified conditions and configurations.

    This function performs a simultaneous solution approach to determine the operating conditions of an ORC system.
    It takes into account thermodynamic properties, pressure drops, and heat exchange efficiencies to solve for the state
    points in the cycle, including the pump, condenser, expander, and evaporator. The function can optionally account for
    advanced exergetic analysis (adex) parameters and plot the results.

    Parameters:
    - target_p33 (float): Target pressure at state point 33 in Pa.
    - print_results (bool): If True, prints the results of the simulation.
    - config (dict): Configuration dictionary specifying which components are included in the simulation (condenser, IHX, evaporator).
    - label (str): A label for the simulation case, used in printing and plotting.
    - adex (bool, optional): If True, uses advanced exergetic analysis parameters. Defaults to False.
    - plot (bool, optional): If True, plots the results of the simulation. Defaults to False.

    Returns:
    - df_streams (DataFrame): A DataFrame containing the mass flow rates, temperatures, enthalpies, pressures, entropies,
      and fluids for each state point in the cycle.
    - t_pinch_eva_sh (float): The pinch temperature difference in the superheater section of the evaporator.
    - efficiency (float): The efficiency of the ORC system calculated as the net power output divided by the heat input.
    """

    # Load configurations from JSON file
    with open('inputs/orc_ihx_R152a_unavoid.json', 'r') as file:
        orc_config_unavoid = json.load(file)
    with open('inputs/orc_ihx_R152a.json', 'r') as file:
        orc_config = json.load(file)
    try:
        epsilon = pd.read_csv('outputs/adex_orc/orc_comps_all_real.csv', index_col=0)['epsilon']  # from base case
        # Additional code to process epsilon if needed
    except FileNotFoundError:
        print('File not found. Please run base case model first!')
        # Handle the exception (e.g., set epsilon to None or a default value)
        epsilon = None

    # Access values directly from the loaded JSON configuration
    wf = orc_config['fluid_properties']['wf']
    fluid_tes = orc_config['fluid_properties']['fluid_tes']

    # TES
    t41 = orc_config['tes']['t41']  # K
    t42 = orc_config['tes']['t42']  # K
    p41 = orc_config['tes']['p41']  # Pa
    m41 = orc_config['tes']['m41']  # kg/s

    # PRESSURE DROPS
    if config['cond'] == 'unavoid':
        pr_cond_hot = orc_config_unavoid['pressure_drops']['cond']['pr_cond_hot']
    elif config['cond'] == 'real':
        pr_cond_hot = orc_config['pressure_drops']['cond']['pr_cond_hot']
    else:
        pr_cond_hot = 1

    if config['ihx'] == 'unavoid':
        pr_ihx_hot = orc_config_unavoid['pressure_drops']['ihx']['pr_ihx_hot']
        pr_ihx_cold = orc_config_unavoid['pressure_drops']['ihx']['pr_ihx_cold']
    elif config['ihx'] == 'real':
        pr_ihx_hot = orc_config['pressure_drops']['ihx']['pr_ihx_hot']
        pr_ihx_cold = orc_config['pressure_drops']['ihx']['pr_ihx_cold']
    else:
        pr_ihx_hot = 1
        pr_ihx_cold = 1

    if config['eva'] == 'unavoid':
        pr_eva_hot = orc_config_unavoid['pressure_drops']['eva']['pr_eva_hot']
        pr_eva_cold = orc_config_unavoid['pressure_drops']['eva']['pr_eva_cold']
    elif config['eva'] == 'real':
        pr_eva_hot = orc_config['pressure_drops']['eva']['pr_eva_hot']
        pr_eva_cold = orc_config['pressure_drops']['eva']['pr_eva_cold']
    else:
        pr_eva_hot = 1
        pr_eva_cold = 1

    pr_eva_part_cold = np.cbrt(pr_eva_cold)  # eva pressure drop is split equally (geom, mean) between ECO-EVA-SH
    pr_eva_part_hot = np.cbrt(pr_eva_hot)    # eva pressure drop is split equally (geom, mean) between ECO-EVA-SH
    
    # AMBIENT
    t0 = orc_config['ambient']['t0']  # K
    p0 = orc_config['ambient']['p0']  # Pa

    # HEAT PUMP
    if config['cond'] == 'unavoid':
        ttd_l_cond = orc_config_unavoid['orc']['ttd_l_cond']  # K
    elif config['cond'] == 'real':
        ttd_l_cond = orc_config['orc']['ttd_l_cond']  # K
    else:
        ttd_l_cond = 0  # K

    if config['ihx'] == 'unavoid':
        ttd_l_ihx = orc_config_unavoid['orc']['ttd_l_ihx']  # K
    elif config['ihx'] == 'real':
        ttd_l_ihx = orc_config['orc']['ttd_l_ihx']   # K
    else:
        ttd_l_ihx = 0   # K

    if config['eva'] == 'unavoid':
        ttd_u_eva = orc_config_unavoid['orc']['ttd_u_eva']  # K
    elif config['eva'] == 'real':
        ttd_u_eva = orc_config['orc']['ttd_u_eva']   # K
    else:
        ttd_u_eva = 0   # K

    # TECHNICAL PARAMETERS
    if config['pump'] == 'unavoid':
        eta_s_pump = orc_config_unavoid['technical_parameters']['eta_s_pump']  # -
    else:
        eta_s_pump = orc_config['technical_parameters']['eta_s_pump']  # -

    if config['exp'] == 'unavoid':
        eta_s_exp = orc_config_unavoid['technical_parameters']['eta_s_exp']    # -
    else:
        eta_s_exp = orc_config['technical_parameters']['eta_s_exp']  # -

    # PRE-CALCULATION
    t31 = t0 + ttd_l_cond
    t34 = t41 - ttd_u_eva

    # STARTING VALUES
    h_values = orc_config['starting_values']['enthalpies']
    p_values = orc_config['starting_values']['pressures']
    m31 = orc_config['starting_values']['mass_flows']['m31']

    variables = np.array([h_values['h31'],  p_values['p31'], h_values['h32'], h_values['h36'], p_values['p36'],
                          p_values['p32'],  p_values['p34'], h_values['h34'], p_values['p35'], h_values['h35'],
                          h_values['h33'], h_values['h41'], h_values['h42'], p_values['p42'], m31,
                          h_values['h38'], h_values['h39'], h_values['h48'], h_values['h49'],  p_values['p38'],
                          p_values['p39'], p_values['p48'], p_values['p49']])
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
        if adex and config['pump'] == 'ideal':  # ideal pump: epsilon = 1
            t32_calc_pump = eps_compressor_func(1, variables[0], variables[1], variables[2], variables[5], wf)
        elif adex and config['pump'] == 'real':  # adex real pump: epsilon from base case
            t32_calc_pump = eps_compressor_func(epsilon['pump'], variables[0], variables[1], variables[2], variables[5], wf)
        else:  # real or unavoid pump: given eta_s
            t32_calc_pump = eta_s_compressor_func(eta_s_pump, variables[0], variables[1], variables[2], variables[5], wf)
        # 3
        if adex and config['ihx'] == 'ideal':
            if config['cond'] == 'real':
                t36_set = ttd_temperature_func(1, variables[2], variables[5], wf, variables[3], variables[4], wf)
            elif config['cond'] == 'ideal':
                t36_set = same_temperature_func(variables[2], variables[5], wf, variables[3], variables[4], wf)
        # elif adex and config['ihx']:  # real ihx for adv. exergy analysis
        #     t36_set = eps_real_ihx_func(epsilon['ihx'], variables[9], variables[8], variables[3], variables[4], wf,
        #                                 variables[2], variables[5], variables[10], target_p33, wf)
        # PROBLEM: otherwise is temp. diff. in IHX negative
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
        if adex and config['exp'] == 'ideal':
            t35_calc_exp = eps_expander_func(1, variables[7], variables[6], variables[9], variables[8], wf)
        elif adex and config['exp'] == 'real':
            t35_calc_exp = eps_expander_func(epsilon['exp'], variables[7], variables[6], variables[9], variables[8], wf)
        else:
            t35_calc_exp = eta_s_expander_func(eta_s_exp, variables[7], variables[6], variables[9], variables[8], wf)
        # 10
        if adex and config['ihx'] == 'ideal':
            # if config['eva'] and not config['pump'] and not config['exp'] and not config['cond']:  # only real eva
            #     t33_calc = eps_real_he_func(epsilon['eva'], variables[11], p41, variables[12], variables[13], m41,
            #                                 fluid_tes, variables[10], target_p33, variables[7], variables[6], variables[14], wf)
            # else:  # ideal eva
            # PROBLEM: if epsilon['eva'] is set for the other cases, the IHX has a negative entropy production
                t33_calc = ideal_ihx_entropy_func(variables[9], variables[8], variables[3], variables[4], wf,
                                                  variables[2], variables[5], variables[10], target_p33, wf)
        else:  # real ihx for base design or for adv. exergy analysis
            t33_calc = ihx_func(variables[9], variables[3], variables[2], variables[10]) 
        # 11
        t41_set = temperature_func(t41, variables[11], p41, fluid_tes)
        # 12
        t42_set = temperature_func(t42, variables[12], variables[13], fluid_tes)
        # 13
        p42_set = pr_func(pr_eva_hot, p41, variables[13])
        # 14
        if adex and config['eva'] == 'ideal':  # ideal eva
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
        if adex and config['pump'] == 'ideal':
            t32_calc_pump_j = eps_compressor_deriv(1, variables[0], variables[1], variables[2], variables[5], wf)
        elif adex and config['pump'] == 'real':
            t32_calc_pump_j = eps_compressor_deriv(epsilon['pump'], variables[0], variables[1], variables[2], variables[5], wf)
        else:
            t32_calc_pump_j = eta_s_compressor_deriv(eta_s_pump, variables[0], variables[1], variables[2], variables[5], wf)
        # 3
        if adex and config['ihx'] == 'ideal':
            if config['cond'] == 'real':
                t36_set_j = ttd_temperature_deriv(1, variables[2], variables[5], wf, variables[3], variables[4], wf)
            elif config['cond'] == 'ideal':
                t36_set_j = same_temperature_deriv(variables[2], variables[5], wf, variables[3], variables[4], wf)
        # elif adex and config['ihx']:  # real ihx for adv. exergy analysis
        #     t36_set_j = eps_real_ihx_deriv(epsilon['ihx'], variables[9], variables[8], variables[3], variables[4], wf,
        #                                    variables[2], variables[5], variables[10], target_p33, wf)
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
        if adex and config['exp'] == 'ideal':
            t35_calc_exp_j = eps_expander_deriv(1, variables[7], variables[6], variables[9], variables[8], wf)
        elif adex and config['exp'] == 'real':
            t35_calc_exp_j = eps_expander_deriv(epsilon['exp'], variables[7], variables[6], variables[9], variables[8], wf)
        else:
            t35_calc_exp_j = eta_s_expander_deriv(eta_s_exp, variables[7], variables[6], variables[9], variables[8], wf)
        # 10
        if adex and config['ihx'] == 'ideal':
            # if config['eva'] and not config['pump'] and not config['exp'] and not config['cond']:  # only real eva
            #     t33_calc_j = eps_real_he_deriv(epsilon['eva'], variables[11], p41, variables[12], variables[13], m41,
            #                                    fluid_tes, variables[10], target_p33, variables[7], variables[6], variables[14], wf)
            # else:  # ideal eva
                t33_calc_j = ideal_ihx_entropy_deriv(variables[9], variables[8], variables[3], variables[4], wf,
                                                     variables[2], variables[5], variables[10], target_p33, wf)
        else:  # real ihx for base design or for adv. exergy analysis
            t33_calc_j = ihx_deriv(variables[9], variables[3], variables[2], variables[10]) 
        # 11
        t41_set_j = temperature_deriv(t41, variables[11], p41, fluid_tes)
        # 12
        t42_set_j = temperature_deriv(t42, variables[12], variables[13], fluid_tes)
        # 13
        p42_set_j = pr_deriv(pr_eva_hot, p41, variables[13])
        # 14
        if adex and config['eva'] == 'ideal':
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
        if adex and config['ihx'] == 'ideal':
            jacobian[3, 2] = t36_set_j['h_copy']  # derivative of t36_set_j with respect to h35
            jacobian[3, 5] = t36_set_j['p_copy']  # derivative of t36_set_j with respect to p35
            jacobian[3, 3] = t36_set_j['h_paste']  # derivative of t36_set_j with respect to h36
            jacobian[3, 4] = t36_set_j['p_paste']  # derivative of t36_set_j with respect to p36
        # elif adex and config['ihx']:
        #     jacobian[3, 9] = t36_set_j['h_hot_in']  # derivative of t36_set_j with respect to h35
        #     jacobian[3, 8] = t36_set_j['p_hot_in']  # derivative of t36_set_j with respect to p35
        #     jacobian[3, 3] = t36_set_j['h_hot_out']  # derivative of t36_set_j with respect to h36
        #     jacobian[3, 4] = t36_set_j['p_hot_out']  # derivative of t36_set_j with respect to p36
        #     jacobian[3, 2] = t36_set_j['h_cold_in']  # derivative of t36_set_j with respect to h32
        #     jacobian[3, 5] = t36_set_j['p_cold_in']  # derivative of t36_set_j with respect to p32
        #     jacobian[3, 10] = t36_set_j['h_cold_out']  # derivative of t36_set_j with respect to h33
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
        if adex and config['ihx'] == 'ideal':  # ideal ihx
            # if config['eva'] and not config['pump'] and not config['exp'] and not config['cond']:  # only real eva
            # jacobian[10, 11] = t33_calc_j['h_hot_in']  # derivative of m31_calc_eva with respect to h41
            # jacobian[10, 12] = t33_calc_j['h_hot_out']  # derivative of m31_calc_eva with respect to h42
            # jacobian[10, 13] = t33_calc_j['p_hot_out']  # derivative of m31_calc_eva with respect to p42
            # jacobian[10, 10] = t33_calc_j['h_cold_in']  # derivative of m31_calc_eva with respect to h33
            # jacobian[10, 7] = t33_calc_j['h_cold_out']  # derivative of m31_calc_eva with respect to h34
            # jacobian[10, 6] = t33_calc_j['p_cold_out']  # derivative of m31_calc_eva with respect to p34
            # jacobian[10, 14] = t33_calc_j['m_cold']  # derivative of m31_calc_eva with respect to m31
            # else:
                jacobian[10, 9] = t33_calc_j['h_hot_in']  # derivative of t33_calc_ihx_j with respect to h35
                jacobian[10, 8] = t33_calc_j['p_hot_in']  # derivative of t33_calc_ihx_j with respect to p35
                jacobian[10, 3] = t33_calc_j['h_hot_out']  # derivative of t33_calc_ihx_j with respect to h36
                jacobian[10, 4] = t33_calc_j['p_hot_out']  # derivative of t33_calc_ihx_j with respect to p36
                jacobian[10, 2] = t33_calc_j['h_cold_in']  # derivative of t33_calc_ihx_j with respect to h32
                jacobian[10, 5] = t33_calc_j['p_cold_in']  # derivative of t33_calc_ihx_j with respect to p32
                jacobian[10, 10] = t33_calc_j['h_cold_out']  # derivative of t33_calc_ihx_j with respect to h33
        else:
            jacobian[10, 9] = t33_calc_j['h_1']  # derivative of t33_calc_ihx with respect to h35
            jacobian[10, 3] = t33_calc_j['h_2']  # derivative of t33_calc_ihx with respect to h36
            jacobian[10, 2] = t33_calc_j['h_3']  # derivative of t33_calc_ihx with respect to h32
            jacobian[10, 10] = t33_calc_j['h_4']  # derivative of t33_calc_ihx with respect to h33
        jacobian[11, 11] = t41_set_j['h']  # derivative of t41_set with respect to h41
        jacobian[12, 12] = t42_set_j['h']  # derivative of t42_set with respect to h42
        jacobian[12, 13] = t42_set_j['p']  # derivative of t42_set with respect to p42
        jacobian[13, 13] = p42_set_j['p_2']  # derivative of p42_set with respect to p42
        if adex and config['eva'] == 'ideal':
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

    '''if plot:
        qt_diagram(df_streams, 'IHX', 35, 36, 32, 33,
                   ttd_l_ihx, 'ORC', plot=plot, case=f'{label} with ttd_l_ihx={ttd_l_ihx}', step_number=100,
                   path=f'outputs/diagrams/adex_orc_qt_ihx_{label}.png')
        [min_td_eva, _] = qt_diagram(df_streams, 'EVA', 41, 42, 33, 34,
                                     ttd_u_eva, 'ORC', plot=plot, case=f'{label} with ttd_u_eva={ttd_u_eva}',
                                     step_number=100, path=f'outputs/diagrams/adex_orc_qt_eva_{label}.png')'''

    t_pinch_eva_sh = t49 - t38

    efficiency = (power_exp+power_eva+power_eva-power_pump)/heat_eva

    return df_streams, t_pinch_eva_sh, efficiency


def exergy_analysis_orc(t0, p0, df):
    """
    Conducts an exergy analysis on an Organic Rankine Cycle (ORC) system using the thermodynamic properties of the cycle's state points.

    This function calculates the physical exergy, exergy destruction, and exergy efficiency of each component within an ORC system,
    based on the inlet temperatures, pressures, and the thermodynamic states of the working fluid and the secondary fluid in thermal
    energy storage (TES). The analysis helps in identifying the major sources of inefficiency within the system.

    Parameters:
    - t0 (float): Ambient temperature in Kelvin.
    - p0 (float): Ambient pressure in Pa.
    - df (DataFrame): A DataFrame containing the mass flow rates (m [kg/s]), temperatures (T [°C]),
      enthalpies (h [kJ/kg]), pressures (p [bar]), entropies (s [J/kgK]), and fluids for each state point in the cycle.

    Returns:
    - df_comps (DataFrame): A DataFrame with the exergy flow (EF [kW]), exergy destruction (ED [kW]), exergy
      performance (EP [kW]), exergy efficiency (epsilon), power (P [kW]), and heat (Q [kW]) for each component of the ORC system.
    """

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
    # ED_EVA = E36 - E31
    ed_cond = (df.loc[31, 'm [kg/s]'] * (df.loc[36, 'e^PH [kJ/kg]'] - df.loc[31, 'e^PH [kJ/kg]']))

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
                            columns=['EF [kW]', 'EP [kW]', 'ED [kW]', 'epsilon', 'P [kW]', 'Q [kW]', 'unavoid ratio'])

    for k in ['pump', 'eva', 'ihx', 'exp', 'cond', 'tot']:
        df_comps.loc[k, 'epsilon'] = epsilon[k]
        df_comps.loc[k, 'EF [kW]'] = ef[k]
        if df_comps.loc[k, 'EF [kW]'] is not None or df_comps.loc[k, 'epsilon'] is not None:
            df_comps.loc[k, 'EP [kW]'] = df_comps.loc[k, 'EF [kW]'] * df_comps.loc[k, 'epsilon']
        df_comps.loc[k, 'ED [kW]'] = ed[k]
        if df_comps.loc[k, 'EP [kW]'] is not None:
            df_comps.loc[k, 'unavoid ratio'] = df_comps.loc[k, 'ED [kW]'] / df_comps.loc[k, 'EP [kW]']

    df_comps.loc['cond', 'unavoid ratio'] = df_comps.loc['cond', 'ED [kW]'] / df.loc[31, 'm [kg/s]']

    df_comps.loc['pump', 'P [kW]'] = power_pump
    df_comps.loc['eva', 'P [kW]'] = power_eva
    df_comps.loc['ihx', 'P [kW]'] = power_ihx
    df_comps.loc['exp', 'P [kW]'] = power_exp
    df_comps.loc['cond', 'Q [kW]'] = heat_cond
    df_comps.loc['ihx', 'Q [kW]'] = heat_ihx
    df_comps.loc['eva', 'Q [kW]'] = heat_eva
    df_comps.loc['tot', 'P [kW]'] = - power_pump + power_eva + power_exp + power_ihx

    return df_comps


def set_adex_orc_config(pump='real', eva='real', ihx='real', exp='real', cond='real'):
    """
    Generates a configuration dictionary and label for an advanced exergy analysis of a ORC system based on the state of its components.

    This version of the function constructs the label with the following rules:
    1. If a component is 'real', it's mentioned by name in the label unless another component is 'unavoid'.
    2. If a component is 'ideal', it's not mentioned in the label.
    3. If a component is 'unavoid', it's mentioned by name with '_unavoid' appended.
    4. If there's any 'unavoid' component, 'real' components are not included in the label.
    5. If all components share the same state, the label is 'all_real', 'all_ideal', or 'all_unavoid' accordingly.

    Parameters:
    - pump, eva, ihx, exp, cond (str): States of the pump, evaporator, internal heat exchanger, expander, and condenser respectively.

    Returns:
    - config (dict): States of heat pump components.
    - label (str): Identifier for the configuration.
    """

    config = {
        'pump': pump,
        'eva': eva,
        'ihx': ihx,
        'exp': exp,
        'cond': cond
    }

    if all(state == 'real' for state in config.values()):
        return config, 'all_real'
    elif all(state == 'ideal' for state in config.values()):
        return config, 'all_ideal'
    elif all(state == 'unavoid' for state in config.values()):
        return config, 'all_unavoid'

    label_parts = []
    has_unavoid = any(state == 'unavoid' for state in config.values())

    for component, state in config.items():
        if state == 'unavoid':
            label_parts.append(f'{component}_unavoid')
        elif state == 'real' and not has_unavoid:
            label_parts.append(component)

    label = '_'.join(label_parts)

    return config, label


def find_opt_p33(p33_opt_start, min_t_diff_eva_start, config, label, adex=False, output_buffer=None):
    """
    Finds the optimal pressure at state point 33 to achieve a target minimum temperature difference in the evaporator of an ORC system.

    This function iteratively adjusts the pressure at state point 33 to minimize the difference between the actual minimum temperature
    difference across the evaporator and a target value. It uses a simple optimization approach, adjusting the pressure based on the
    observed temperature difference and a predetermined learning rate.

    Parameters:
    - p33_opt_start (float): Initial guess for the optimal pressure at state point 33 in Pa.
    - min_t_diff_eva_start (float): Initial minimum temperature difference in the evaporator in Kelvin.
    - config (dict): Configuration dictionary specifying which components are included in the ORC system simulation.
    - label (str): A label for the simulation case, used in logging and output.
    - adex (bool, optional): If True, uses advanced exergetic analysis parameters. Defaults to False.
    - output_buffer (io.StringIO, optional): A buffer to capture the optimization process output. If None, output is not captured.

    Returns:
    - p33_opt (float): The optimized pressure at state point 33 in Pa that achieves or comes closest to the target minimum temperature
                       difference in the evaporator.

    Note:
    - The function logs the optimization progress, including the current step, the difference between the target and actual minimum
      temperature differences, the current guess for the optimal pressure, and the system's efficiency at each step.
    - If an output_buffer is provided, the log is appended to it; otherwise, the log is printed or discarded.
    """

    buffer = io.StringIO()  # Create a new buffer

    with open('inputs/orc_ihx_R152a_unavoid.json', 'r') as file:
        orc_config_unavoid = json.load(file)
    with open('inputs/orc_ihx_R152a.json', 'r') as file:
        orc_config = json.load(file)
        
    if config['eva'] == 'unavoid':
        target_min_td_eva = orc_config_unavoid['orc']['ttd_u_eva']  # K
    elif config['eva'] == 'real':
        target_min_td_eva = orc_config['orc']['ttd_u_eva']  # K
    else:
        target_min_td_eva = 0
        
    min_t_diff_eva = target_min_td_eva
    p33_opt = p33_opt_start
    tolerance = 1e-3  # relative to min temperature difference
    learning_rate = 5e4  # relative to p33
    diff = min_t_diff_eva_start - target_min_td_eva
    step = 0

    while abs(diff) > tolerance:
        # Adjust p33 based on the difference
        adjustment = -(target_min_td_eva - min_t_diff_eva) * learning_rate
        # adjustment is the smaller, the smaller the difference target_min_td_eva - min_td_eva
        p33_opt += adjustment

        [_, min_t_diff_eva, eta] = solve_orc(p33_opt, print_results=False, label=label, config=config, adex=adex)
        diff = abs(min_t_diff_eva - target_min_td_eva)

        step += 1
        buffer.write(f'Optimization in progress for {label}: step = {step}, diff = {round(diff, 6)}, p33 = {round(p33_opt * 1e-5, 4)} bar, eta = {round(eta, 4)}.\n')

    buffer.write(f'Optimization completed successfully in {step} steps!\n')
    buffer.write(f'Optimal p33: {round(p33_opt * 1e-5, 4)}, bar.\n')
    
    if output_buffer is not None:
        output_buffer.append(buffer.getvalue())

    return p33_opt


def perform_adex_orc(p33_start, config, label, df_ed, adex=False, save=False, print_results=False, calc_epsilon=False, output_buffer=None):
    """
    Executes an advanced exergy analysis for an Organic Rankine Cycle (ORC) system, optimizing a specific target condition, and evaluating system performance.

    This function automates the process of optimizing the pressure at state point 33 based on a target condition, performing
    an ORC system simulation under the optimized conditions, conducting exergy analysis to identify inefficiencies, and
    optionally calculating and saving exergy efficiency results.

    Parameters:
    - p33_start (float): Starting pressure at state point 33 in Pa, used as the initial guess for optimization.
    - config (dict): Configuration dictionary specifying which components are included in the simulation.
    - label (str): A label for the simulation case, used in printing and saving results.
    - df_ed (DataFrame): A DataFrame to store the exergy destruction (ED) results for each component of the system.
    - adex (bool, optional): If True, uses advanced exergetic analysis parameters. Defaults to False.
    - save (bool, optional): If True, saves the simulation results to CSV files. Defaults to False.
    - print_results (bool, optional): If True, prints the results of the simulation. Defaults to False.
    - calc_epsilon (bool, optional): If True, calculates the exergy efficiency of the system components. Defaults to False.
    - output_buffer (io.StringIO, optional): A buffer to capture and return the optimization process output. If None, output is not captured.

    Returns:
    - df_ed (DataFrame): Updated DataFrame with the exergy destruction (ED) results for each component.

    Raises:
    - Error messages are printed if there are negative temperature differences in the evaporator or internal heat exchanger (IHX),
      indicating an issue with the optimization or simulation setup.
    """

    # Run first simulation and get pinch point in evaporator (target_diff)
    [_, target_diff, _] = solve_orc(p33_start, print_results=print_results, config=config, label=label, adex=adex)
    # Search for optimal p33 according to minimum pinch point in evaporator
    p33_opt = find_opt_p33(p33_start, target_diff, config=config, label=label, adex=adex, output_buffer=output_buffer)
    # Run simulation with optimal p33 and minimum pinch point in evaporator
    [df_opt, _, _] = solve_orc(p33_opt, print_results=print_results, config=config, label=f'optimal {label}', adex=adex)

    # Run conventional exergy analysis to get exergetic efficiencies and exergy destruction rates
    with open('inputs/orc_ihx_R152a.json', 'r') as file:
        orc_config = json.load(file)
    t0 = orc_config['ambient']['t0']  # K
    p0 = orc_config['ambient']['p0']  # Pa
    df_components = exergy_analysis_orc(t0, p0, df_opt)
    
    for component in df_components.index:
        df_ed.loc[(label, component), 'ED [kW]'] = df_components.loc[component, 'ED [kW]']
    if calc_epsilon:
        for col in df_components.columns[:5]:
            df_components[col] = pd.to_numeric(df_components[col], errors='coerce')
        if label == "all_real":
            df_components.to_csv(f'outputs/adex_orc/orc_comps_{label}.csv')
        else:
            df_components.round(4).to_csv(f'outputs/adex_orc/orc_comps_{label}.csv')
    if df_opt.loc[41, 'T [°C]'] < df_opt.loc[34, 'T [°C]'] - 1e-3:
        print('Error: Upper temperature difference in EVA is negative!')
    if df_opt.loc[42, 'T [°C]'] < df_opt.loc[33, 'T [°C]'] - 1e-3:
        print('Error: Lower temperature difference in EVA is negative!')
    if df_opt.loc[35, 'T [°C]'] < df_opt.loc[33, 'T [°C]'] - 1e-3:
        print('Error: Upper temperature difference in IHX is negative!')
    if df_opt.loc[36, 'T [°C]'] < df_opt.loc[32, 'T [°C]'] - 1e-3:
        print('Error: Lower temperature difference in IHX is negative!')
    if save:
        round(df_opt, 5).to_csv(f'outputs/adex_orc/orc_streams_{label}.csv')

    return df_ed


def main_serial():
    """
    The main function for running a serial (sequential) execution of advanced exergy analysis (ADEX) on an Organic Rankine Cycle (ORC) system.

    This function sequentially executes the ADEX analysis for different configurations of an ORC system, starting with the base case,
    followed by the ideal case, single components in real conditions, and pairs of components in real conditions. Each configuration's
    exergy destruction is calculated, and the results are saved and optionally printed. Avoidable/Unavoidable is not
    considered. In this case, please run the multiprocess function. This function should be used only for debugging.

    The function:
    - Sets up configurations for the base case, ideal case, individual components, and pairs of components.
    - Executes the perform_adex_orc function for each configuration to perform the ADEX analysis.
    - Saves the results in a DataFrame which includes exergy destruction values for each component and configuration.
    - Optionally prints the results and calculates exergy efficiency for further analysis.

    Parameters:
    None

    Returns:
    None

    Note:
    This function does not accept any arguments and does not return any values. Instead, it directly modifies global variables
    and performs file I/O operations to save the analysis results. It also measures and prints the total execution time.
    """

    start = time.perf_counter()

    columns = ['ED']
    multi_index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['Label', 'Key'])
    df_ed = pd.DataFrame(columns=columns, index=multi_index)

    # BASE CASE
    [config_base, label_base] = set_adex_orc_config('pump', 'eva', 'ihx', 'exp', 'cond')
    p33_start_real = 22e5  # bar
    perform_adex_orc(p33_start_real, config_base, label_base, df_ed, adex=False, save=True, print_results=True, calc_epsilon=True)

    # IDEAL CASE
    [config_ideal, label_ideal] = set_adex_orc_config()
    p33_start_ideal = 28e5  # bar
    perform_adex_orc(p33_start_ideal, config_ideal, label_ideal, df_ed, adex=True, save=True, print_results=True, calc_epsilon=True)

    # ADVANCED EXERGY ANALYSIS -- single components
    components = ['pump', 'eva', 'ihx', 'exp', 'cond']
    for component in components:
        config_i, label_i = set_adex_orc_config(component)
        p33_start_ideal = 28e5  # bar
        perform_adex_orc(p33_start_ideal, config_i, label_i, df_ed, adex=True, save=True, print_results=True, calc_epsilon=True)

    # ADVANCED EXERGY ANALYSIS -- pair of components
    component_pairs = list(itertools.combinations(components, 2))
    for pair in component_pairs:
        config_i, label_i = set_adex_orc_config(*pair)
        p33_start_ideal = 28e5  # bar
        perform_adex_orc(p33_start_ideal, config_i, label_i, df_ed, adex=True, save=True, print_results=True, calc_epsilon=True)

    end = time.perf_counter()
    print(f'Elapsed time: {round(end - start, 3)} seconds.')

    # Generate combinations
    combinations = []
    for component in components:
        combinations.append((component, ''))  # Component considered alone
        for other_component in components:
            if component != other_component:
                combinations.append((component, other_component))

    multi_index = pd.MultiIndex.from_tuples(combinations, names=['k', 'l'])

    # Initialize DataFrame with MultiIndex
    df_adex_analysis = pd.DataFrame(index=multi_index)
    epsilon = pd.read_csv('outputs/adex_orc/orc_comps_real.csv', index_col=0)['epsilon']
    epsilon['cond'] = float('nan')

    for k in components:
        df_real = pd.read_csv(f'outputs/adex_orc/orc_streams_real.csv', index_col=0)
        df_k = pd.read_csv(f'outputs/adex_orc/orc_streams_{k}.csv', index_col=0)
        df_adex_analysis.loc[(k, ''), 'ED [kW]'] = df_ed.loc[('real', k), 'ED [kW]']
        df_adex_analysis.loc[(k, ''), 'epsilon [%]'] = epsilon[k] * 100
        df_adex_analysis.loc[(k, ''), 'ED EN [kW]'] = df_ed.loc[(k, k), 'ED [kW]']
        df_adex_analysis.loc[(k, ''), 'ED EX [kW]'] = df_adex_analysis.loc[(k, ''), 'ED [kW]'] - df_adex_analysis.loc[
            (k, ''), 'ED EN [kW]']
        df_adex_analysis.loc[(k, ''), 'm EN [kg/s]'] = df_k.loc[31, 'm [kg/s]']
        df_adex_analysis.loc[(k, ''), 'm EX [kg/s]'] = df_real.loc[31, 'm [kg/s]'] - df_k.loc[31, 'm [kg/s]']
        sum_ed_ex_l = 0
        for l in components:
            if k != l:
                k_l = f'{k}_{l}'
                if (k_l, l) not in df_ed.index:
                    k_l = f'{l}_{k}'
                df_k_l = pd.read_csv(f'outputs/adex_orc/orc_streams_{k_l}.csv', index_col=0)
                df_adex_analysis.loc[(k, l), 'm EN [kg/s]'] = df_k_l.loc[31, 'm [kg/s]']
                df_adex_analysis.loc[(k, l), 'm EX [kg/s]'] = df_real.loc[31, 'm [kg/s]'] - df_k_l.loc[31, 'm [kg/s]']
                df_adex_analysis.loc[(k, l), 'ED kl [kW]'] = df_ed.loc[(k_l, k), 'ED [kW]']
                df_adex_analysis.loc[(k, l), 'ED kl [kW]'] = df_ed.loc[(k_l, k), 'ED [kW]']
                df_adex_analysis.loc[(k, l), 'ED EX l [kW]'] = df_adex_analysis.loc[(k, ''), 'ED EN [kW]'] - \
                                                               df_adex_analysis.loc[(k, l), 'ED kl [kW]']
                sum_ed_ex_l += df_adex_analysis.loc[(k, l), 'ED EX l [kW]']
        df_adex_analysis.loc[(k, ''), 'ED MEXO [kW]'] = df_adex_analysis.loc[(k, ''), 'ED EX [kW]'] - sum_ed_ex_l
    df_adex_analysis.round(2).to_csv('outputs/adex_orc/orc_adex_analysis.csv')


def run_orc_adex(config, label, df_ed, adex, save, print_results, calc_epsilon, output_buffer=None):
    """
   Runs the advanced exergy analysis for an Organic Rankine Cycle (ORC) system with a given configuration and label.

   This function performs the advanced exergy analysis by calling the `perform_adex_orc` function with a starting
   pressure of 2.8 MPa (28e5 Pa) at state point 12. It updates the DataFrame with the exergy destruction results
   and returns it.

   Parameters:
   - config (dict): Configuration dictionary specifying which components are included in the simulation.
   - label (str): A label for the simulation case, used in printing and saving results.
   - df_ed (DataFrame): A DataFrame to store the exergy destruction (ED) results for each component of the system.
   - adex (bool): If True, uses advanced exergetic analysis parameters.
   - save (bool): If True, saves the simulation results to CSV files.
   - print_results (bool): If True, prints the results of the simulation.
   - calc_epsilon (bool): If True, calculates the exergy efficiency of the system components.
   - output_buffer (io.StringIO, optional): A buffer to capture and return the optimization process output. If None, output is not captured.

   Returns:
   - df_ed (DataFrame): Updated DataFrame with the exergy destruction (ED) results for each component.
   """

    df_ed = perform_adex_orc(28e5, config, label, df_ed, adex, save, print_results, calc_epsilon, output_buffer)
    return df_ed


def main_multiprocess():
    """
    The main function for running a multiprocessing execution of advanced exergy analysis (ADEX) on an Organic Rankine Cycle (ORC) system.

    This function parallelizes the execution of the ADEX analysis for different configurations of an ORC system, including the base case,
    the ideal case, and variations with individual components and pairs of components in real conditions. Exergy destruction calculations
    are performed for each configuration, and the results are stored, with the option to save and print the results.

    The multiprocessing approach significantly reduces the overall computation time, especially beneficial for extensive simulations
    involving multiple configurations or large parameter spaces.

    Parameters:
    None

    Returns:
    None

    Note:
    - This function utilizes the multiprocessing module to create a pool of worker processes, which execute tasks in parallel.
    - The configurations for the ADEX analysis are defined within the function, and the perform_adex_orc function is called for each configuration.
    - Results from each process are collected and concatenated into a final DataFrame, which is then saved to a CSV file.
    - Execution time for the entire process is measured and printed at the end of the function.
    """

    start = time.perf_counter()

    # Initialize the DataFrame for storing results
    columns = ['ED [kW]']
    multi_index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['Label', 'Key'])
    df_ed_base = pd.DataFrame(columns=columns, index=multi_index)
    df_ed_i = pd.DataFrame(columns=columns, index=multi_index)
    df_ed = pd.DataFrame(columns=columns, index=multi_index)

    # Shared data structure for output
    manager = multiprocessing.Manager()
    output_buffer = manager.list()

    # BASE CASE
    [config_base, label_base] = set_adex_orc_config()
    perform_adex_orc(22e5, config_base, label_base, df_ed_base, adex=False, save=True, 
                     print_results=True, calc_epsilon=True, output_buffer=output_buffer)

    tasks = []

    # Create a pool of workers
    with multiprocessing.Pool() as pool:

        # IDEAL CASE
        config_ideal, label_ideal = set_adex_orc_config(pump='ideal', eva='ideal', ihx='ideal', exp='ideal', cond='ideal')
        tasks.append((config_ideal, label_ideal, df_ed_i, True, True, True, True, output_buffer))

        components = ['pump', 'eva', 'ihx', 'exp', 'cond']
        component_pairs = list(itertools.combinations(components, 2))

        # EN ED -- single components
        for component in components:
            config_kwargs = {comp: 'ideal' for comp in components}  # Start with all components as 'ideal'
            config_kwargs[component] = 'real'  # Set the current component to 'real'
            config_i, label_i = set_adex_orc_config(**config_kwargs)
            tasks.append((config_i, label_i, df_ed_i, True, True, True, True, output_buffer))

        # EN ED -- pair of components
        for pair in component_pairs:
            config_kwargs = {comp: 'ideal' for comp in components}  # Start with all components as 'ideal'
            for comp in pair:
                config_kwargs[comp] = 'real'  # Set each component in the pair to 'real'
            config_i, label_i = set_adex_orc_config(**config_kwargs)
            tasks.append((config_i, label_i, df_ed_i, True, True, True, True, output_buffer))

        # UN ED -- single components
        for component in components:
            config_kwargs = {comp: 'real' for comp in components}  # Start with all components as 'real'
            config_kwargs[component] = 'unavoid'  # Set the current component to 'unavoid'
            config_i, label_i = set_adex_orc_config(**config_kwargs)
            tasks.append((config_i, label_i, df_ed_i, False, True, True, True, output_buffer))

        # Map tasks to the pool
        results = pool.starmap(run_orc_adex, tasks)

    print(round(time.perf_counter() - start, 3))

    # Print the stored outputs after all tasks are done
    for output in output_buffer:
        print(output)

    # Collect and concatenate results
    for result in results:
        df_result = pd.DataFrame(result, columns=columns)
        df_ed = pd.concat([df_ed, df_result])
    df_ed = pd.concat([df_ed_base, df_ed])

    # Create long table with all the cases and save it
    for col in df_ed.columns[:]:
        df_ed[col] = pd.to_numeric(df_ed[col], errors='coerce')
    df_ed.round(6).to_csv('outputs/adex_orc/orc_adex_ed.csv')

    # Generate combinations for EN / EX
    combinations = []
    for component in components:
        combinations.append((component, ''))  # Component considered alone
        for other_component in components:
            if component != other_component:
                combinations.append((component, other_component))

    # Create a MultiIndex
    multi_index = pd.MultiIndex.from_tuples(combinations, names=['k', 'l'])

    # Initialize DataFrame with MultiIndex
    df_adex_analysis = pd.DataFrame(index=multi_index)
    epsilon = pd.read_csv('outputs/adex_orc/orc_comps_all_real.csv', index_col=0)['epsilon']

    for k in components:
        # EN / EX / binary interaction / MX
        df_k_real_streams = pd.read_csv(f'outputs/adex_orc/orc_streams_all_real.csv', index_col=0)
        df_k_endo = pd.read_csv(f'outputs/adex_orc/orc_streams_{k}.csv', index_col=0)
        df_adex_analysis.loc[(k, ''), 'ED [kW]'] = df_ed.loc[('all_real', k), 'ED [kW]']
        df_adex_analysis.loc[(k, ''), 'epsilon [%]'] = epsilon[k] * 100
        df_adex_analysis.loc[(k, ''), 'ED EN [kW]'] = df_ed.loc[(k, k), 'ED [kW]']
        df_adex_analysis.loc[(k, ''), 'ED EX [kW]'] = df_adex_analysis.loc[(k, ''), 'ED [kW]']-df_adex_analysis.loc[(k, ''), 'ED EN [kW]']
        df_adex_analysis.loc[(k, ''), 'm EN [kg/s]'] = df_k_endo.loc[31, 'm [kg/s]']
        df_adex_analysis.loc[(k, ''), 'm EX [kg/s]'] = df_k_real_streams.loc[31, 'm [kg/s]'] - df_k_endo.loc[31, 'm [kg/s]']
        sum_ed_ex_kl = 0
        for l in components:
            if k != l:
                k_l = f'{k}_{l}'
                if (k_l, l) not in df_ed.index:
                    k_l = f'{l}_{k}'
                df_k_l = pd.read_csv(f'outputs/adex_orc/orc_streams_{k_l}.csv', index_col=0)
                df_adex_analysis.loc[(k, l), 'm EN [kg/s]'] = df_k_l.loc[31, 'm [kg/s]']
                df_adex_analysis.loc[(k, l), 'm EX [kg/s]'] = df_k_real_streams.loc[31, 'm [kg/s]'] - df_k_l.loc[31, 'm [kg/s]']
                df_adex_analysis.loc[(k, l), 'ED kl [kW]'] = df_ed.loc[(k_l, k), 'ED [kW]']
                df_adex_analysis.loc[(k, l), 'ED kl [kW]'] = df_ed.loc[(k_l, k), 'ED [kW]']
                df_adex_analysis.loc[(k, l), 'ED EX kl [kW]'] = df_adex_analysis.loc[(k, l), 'ED kl [kW]'] - df_adex_analysis.loc[(k, ''), 'ED EN [kW]']
                sum_ed_ex_kl += df_adex_analysis.loc[(k, l), 'ED EX kl [kW]']
        df_adex_analysis.loc[(k, ''), 'ED MX [kW]'] = df_adex_analysis.loc[(k, ''), 'ED EX [kW]'] - sum_ed_ex_kl
        
        # UN / AV/ binary interaction
        df_k_unavoid_comps = pd.read_csv(f'outputs/adex_orc/orc_comps_{k}_unavoid.csv', index_col=0)
        df_k_endo_comps = pd.read_csv(f'outputs/adex_orc/orc_comps_{k}.csv', index_col=0)
        df_k_endo_streams = pd.read_csv(f'outputs/adex_orc/orc_streams_{k}.csv', index_col=0)
        df_k_real_comps = pd.read_csv(f'outputs/adex_orc/orc_comps_all_real.csv', index_col=0)
        if k == 'cond':
            df_adex_analysis.loc[(k, ''), 'ED UN [kW]'] = df_k_real_streams.loc[31, 'm [kg/s]'] * df_k_unavoid_comps.loc[k, 'unavoid ratio']
        else:
            df_adex_analysis.loc[(k, ''), 'ED UN [kW]'] = df_k_real_comps.loc[k, 'EP [kW]'] * df_k_unavoid_comps.loc[k, 'unavoid ratio']
        df_adex_analysis.loc[(k, ''), 'ED AV [kW]'] = df_ed.loc[('all_real', k), 'ED [kW]'] - df_adex_analysis.loc[(k, ''), 'ED UN [kW]']
        if k == 'cond':
            df_adex_analysis.loc[(k, ''), 'ED UN EN [kW]'] = df_k_endo_streams.loc[31, 'm [kg/s]'] * df_k_unavoid_comps.loc[k, 'unavoid ratio']
        else:
            df_adex_analysis.loc[(k, ''), 'ED UN EN [kW]'] = df_k_endo_comps.loc[k, 'EP [kW]'] * df_k_unavoid_comps.loc[k, 'unavoid ratio']
        df_adex_analysis.loc[(k, ''), 'ED UN EX [kW]'] = df_adex_analysis.loc[(k, ''), 'ED UN [kW]'] - df_adex_analysis.loc[(k, ''), 'ED UN EN [kW]']
        df_adex_analysis.loc[(k, ''), 'ED AV EN [kW]'] = df_adex_analysis.loc[(k, ''), 'ED EN [kW]'] - df_adex_analysis.loc[(k, ''), 'ED UN EN [kW]']
        df_adex_analysis.loc[(k, ''), 'ED AV EX [kW]'] = df_adex_analysis.loc[(k, ''), 'ED AV [kW]'] - df_adex_analysis.loc[(k, ''), 'ED AV EN [kW]']
        for l in components:
            if k != l:
                df_adex_analysis.loc[(k, l), 'ED AV EX kl [kW]'] = (df_adex_analysis.loc[(k, ''), 'ED AV EX [kW]']
                                                                    * df_adex_analysis.loc[(k, l), 'ED EX kl [kW]'] / df_adex_analysis.loc[(k, ''), 'ED EX [kW]'])
    # AV SUM
    for k in components:
        sum_ed_ex_av_kl = 0
        for l in components:
            if k != l:
                sum_ed_ex_av_kl += df_adex_analysis.loc[(l, k), 'ED AV EX kl [kW]']
        df_adex_analysis.loc[(k, ''), 'ED AV SUM [kW]'] = df_adex_analysis.loc[(k, ''), 'ED AV EN [kW]'] + sum_ed_ex_av_kl

    df_adex_analysis.round(5).to_csv('outputs/adex_orc/orc_adex_analysis.csv')

    end = time.perf_counter()
    print(f'Elapsed time: {round(end - start, 3)} seconds.')


# ----------------------------------------------------------------------------------------------------------------------
# ADVANCED EXERGY ANALYSIS
multi = True  # true: multiprocess, false: sequential computation

if __name__ == '__main__':
    if multi:
        main_multiprocess()
    else:
        main_serial()

'''[config_test, label_test] = set_adex_orc_config('real', 'real', 'real', 'real', 'real')
[df_test, target_diff, cop_test] = solve_orc(target_p33=25e5, print_results=True, config=config_test, label=label_test, adex=False)
p12_opt = find_opt_p33(25e5, target_diff, config=config_test, label=label_test, adex=False)
[df_opt, _, _] = solve_orc(p12_opt, print_results=True, config=config_test, label=label_test, adex=False, plot=True)'''