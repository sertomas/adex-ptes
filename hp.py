import numpy as np
from CoolProp.CoolProp import PropsSI as PSI
import itertools
import multiprocessing
import time
import io
import json
import pandas as pd
from functions import (pr_func, pr_deriv, eta_s_compressor_func, eta_s_compressor_deriv, turbo_func, turbo_deriv, he_func,
                       he_deriv, ihx_func, ihx_deriv, temperature_func, temperature_deriv, valve_func, valve_deriv,
                       x_saturation_func, x_saturation_deriv, qt_diagram, eps_compressor_func, eps_compressor_deriv,
                       eps_real_he_func, eps_real_he_deriv, eps_real_ihx_func, eps_real_ihx_deriv, simple_he_func,
                       simple_he_deriv, ideal_ihx_entropy_func, ideal_ihx_entropy_deriv, ideal_he_entropy_func,
                       ideal_he_entropy_deriv, ideal_valve_entropy_func, ideal_valve_entropy_deriv, he_with_p_func,
                       he_with_p_deriv, same_temperature_func, same_temperature_deriv)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def solve_hp(target_p12, print_results, config, label, adex=False, plot=False):
    """
     Simulates the performance of a heat pump system under specified conditions and configurations.

     This function performs a simultaneous solution approach to determine the operating conditions of a heat pump system.
     It considers thermodynamic properties, pressure drops, and heat exchange efficiencies to solve for the state points in the
     cycle, including the compressor, condenser, expansion valve, and evaporator. The function can optionally account for
     advanced exergetic analysis (adex) parameters and plot the results.

     Parameters:
     - target_p12 (float): Target pressure at state point 12 in Pa.
     - print_results (bool): If True, prints the results of the simulation.
     - config (dict): Configuration dictionary specifying which components are included in the simulation.
     - label (str): A label for the simulation case, used in printing and plotting.
     - adex (bool, optional): If True, uses advanced exergetic analysis parameters. Defaults to False.
     - plot (bool, optional): If True, plots the results of the simulation. Defaults to False.

     Returns:
     - df_streams (DataFrame): A DataFrame containing the mass flow rates, temperatures, enthalpies, pressures, entropies,
       and fluids for each state point in the cycle.
     - t_pinch_cond_sh (float): The pinch temperature difference in the superheater section of the condenser.
     - cop (float): The coefficient of performance of the heat pump system.
     """

    # Load configurations from JSON file
    with open('inputs/hp_simult_unavoid.json', 'r') as file:
        hp_config_unavoid = json.load(file)
    with open('inputs/hp_simult_base.json', 'r') as file:
        hp_config = json.load(file)
    try:
        epsilon = pd.read_csv('outputs/adex_hp/hp_comps_all_real.csv', index_col=0)['epsilon']  # from base case
        # Additional code to process epsilon if needed
    except FileNotFoundError:
        print('File not found. Please run base case model first!')
        # Handle the exception (e.g., set epsilon to None or a default value)
        epsilon = None

    # Access values directly from the loaded JSON configuration
    wf = hp_config['fluid_properties']['wf']
    fluid_tes = hp_config['fluid_properties']['fluid_tes']

    # TES
    t21 = hp_config['tes']['t21']  # K
    t22 = hp_config['tes']['t22']  # K
    p21 = hp_config['tes']['p21']  # Pa
    m21 = hp_config['tes']['m21']  # kg/s

    # PRESSURE DROPS
    if config['cond'] == 'unavoid':
        pr_cond_cold = hp_config_unavoid['pressure_drops']['cond']['pr_cond_cold']
        pr_cond_hot = hp_config_unavoid['pressure_drops']['cond']['pr_cond_hot']
    elif config['cond'] == 'real':
        pr_cond_cold = hp_config['pressure_drops']['cond']['pr_cond_cold']
        pr_cond_hot = hp_config['pressure_drops']['cond']['pr_cond_hot']
    else:
        pr_cond_cold = 1
        pr_cond_hot = 1

    if config['ihx'] == 'unavoid':
        pr_ihx_hot = hp_config_unavoid['pressure_drops']['ihx']['pr_ihx_hot']
        pr_ihx_cold = hp_config_unavoid['pressure_drops']['ihx']['pr_ihx_cold']
    elif config['ihx'] == 'real':
        pr_ihx_hot = hp_config['pressure_drops']['ihx']['pr_ihx_hot']
        pr_ihx_cold = hp_config['pressure_drops']['ihx']['pr_ihx_cold']
    else:
        pr_ihx_hot = 1
        pr_ihx_cold = 1

    if config['eva'] == 'unavoid':
        pr_eva_cold = hp_config_unavoid['pressure_drops']['eva']['pr_eva_cold']
    elif config['eva'] == 'real':
        pr_eva_cold = hp_config['pressure_drops']['eva']['pr_eva_cold']
    else:
        pr_eva_cold = 1

    pr_cond_part_cold = np.cbrt(pr_cond_cold)  # cond pressure drop is split equally (geom, mean) between ECO-EVA-SH
    pr_cond_part_hot = np.cbrt(pr_cond_hot)    # cond pressure drop is split equally (geom, mean) between ECO-EVA-SH

    # AMBIENT
    t0 = hp_config['ambient']['t0']  # K
    p0 = hp_config['ambient']['p0']  # Pa

    # HEAT PUMP
    if config['cond'] == 'unavoid':
        ttd_l_cond = hp_config_unavoid['heat_pump']['ttd_l_cond']  # K
    elif config['cond'] == 'real':
        ttd_l_cond = hp_config['heat_pump']['ttd_l_cond']  # K
    else:
        ttd_l_cond = 0  # K

    if config['ihx'] == 'unavoid':
        ttd_u_ihx = hp_config_unavoid['heat_pump']['ttd_u_ihx']   # K
    elif config['ihx'] == 'real':
        ttd_u_ihx = hp_config['heat_pump']['ttd_u_ihx']  # K
    else:
        ttd_u_ihx = 0   # K

    if config['eva'] == 'unavoid':
        ttd_l_eva = hp_config_unavoid['heat_pump']['ttd_l_eva']   # K
    elif config['eva'] == 'real':
        ttd_l_eva = hp_config['heat_pump']['ttd_l_eva']  # K
    else:
        ttd_l_eva = 0   # K

    # TECHNICAL PARAMETERS
    if config['comp'] == 'unavoid':
        eta_s = hp_config_unavoid['technical_parameters']['eta_s']  # -
    else:
        eta_s = hp_config['technical_parameters']['eta_s']  # -

    # PRE-CALCULATION
    t12 = t21 + ttd_l_cond
    t14 = t0 - ttd_l_eva
    t16 = t12 - ttd_u_ihx

    # STARTING VALUES
    h_values = hp_config['starting_values']['enthalpies']
    p_values = hp_config['starting_values']['pressures']
    m11 = hp_config['starting_values']['mass_flows']['m11']
    power_comp = hp_config['starting_values']['power_comp']

    variables = np.array([h_values['h16'], h_values['h11'], p_values['p11'], h_values['h12'], m11, power_comp,
                          h_values['h21'], h_values['h22'], h_values['h13'], h_values['h15'], p_values['p16'],
                          p_values['p13'], h_values['h14'], p_values['p14'], p_values['p22'], p_values['p15'],
                          h_values['h18'], h_values['h19'], h_values['h28'], h_values['h29'], p_values['p18'],
                          p_values['p19'], p_values['p28'], p_values['p29']])

    residual = np.ones(len(variables))

    iter_step = 0

    while np.linalg.norm(residual) > 1e-4:
        #      [h16, h11, p11, h12, m11, power, h21, h22, h13, h15, p16, p13, h14, p14, p22, p15,
        #        0    1    2    3    4     5     6    7    8    9    10   11   12   13   14   15
        #       h18, h19, h28, h29, p18, p19, p28, p29])]
        #        16   17   18   19   20   21   22   23

        #   0
        if adex and config['comp'] == 'ideal':  # ideal comp: epsilon = 1
            t11_calc_comp = eps_compressor_func(1, variables[0], variables[10], variables[1], variables[2], wf)
        elif adex and config['comp'] == 'real':  # adex real comp: epsilon from base case
            t11_calc_comp = eps_compressor_func(epsilon['comp'], variables[0], variables[10], variables[1], variables[2], wf)
        else:  # real or unavoid comp: given eta_s
            t11_calc_comp = eta_s_compressor_func(eta_s, variables[0], variables[10], variables[1], variables[2], wf)
        #   1
        if adex and config['ihx'] == 'ideal':  # ideal ihx: ttd_u = 0
            t16_set = same_temperature_func(variables[3], target_p12, wf, variables[0], variables[10], wf)  # t16 = t12
        elif adex and config['ihx'] == 'real':  # adex real ihx: epsilon from base case
            t16_set = eps_real_ihx_func(epsilon['ihx'], variables[3], target_p12, variables[8], variables[11], wf,
                                        variables[9], variables[15], variables[0], variables[10], wf)
        else:  # real or unavoid ihx: given ttd_u
            t16_set = temperature_func(t16, variables[0], variables[10], wf)
        #   2
        if adex and config['cond'] == 'real':  # adex real cond: epsilon from base case
            t12_set = eps_real_he_func(epsilon['cond'], variables[1], variables[2], variables[3], target_p12, variables[4], wf,
                                       variables[6], p21, variables[7], variables[14], m21, fluid_tes)
        else:  # ideal, real, unavoid cond: given ttd_l (different from cond, because temperature is known in this case)
            t12_set = temperature_func(t12, variables[3], target_p12, wf)
        #   3
        p11_set = pr_func(pr_cond_hot, variables[2], target_p12)
        #   4
        power_calc_comp = turbo_func(variables[5], variables[4], variables[0], variables[1])
        #   5
        if adex and config['cond'] == 'ideal':  # mass flow from ideal case: entropy balance equation
            m11_calc_cond = ideal_he_entropy_func(variables[1], variables[2], variables[3], target_p12, variables[4], wf,
                                                  variables[6], p21, variables[7], variables[14], m21, fluid_tes)
        else:  # mass flow from real and unavoid case: energy balance equation
            m11_calc_cond = he_func(variables[4], variables[1], variables[3], m21, variables[6], variables[7])
        #   6
        t21_set = temperature_func(t21, variables[6], p21, fluid_tes)
        #   7
        t22_set = temperature_func(t22, variables[7], variables[14], fluid_tes)
        #   8
        if adex and config['ihx'] == 'ideal':  # ideal ihx: entropy balance equation for upper state
            t13_calc_ihx = ideal_ihx_entropy_func(variables[3], target_p12, variables[8], variables[11], wf,
                                                  variables[9], variables[15], variables[0], variables[10], wf)
        else:  # real or unavoid ihx: energy balance equation for upper state
            t13_calc_ihx = ihx_func(variables[3], variables[8], variables[9], variables[0])
        #   9
        p16_set = pr_func(pr_ihx_cold, variables[15], variables[10])
        #   10
        p13_set = pr_func(pr_ihx_hot, target_p12, variables[11])
        #   11
        if adex and config['val'] == 'ideal':  # ideal val: entropy balance equation (isentropic expansion)
            t14_calc_valve = ideal_valve_entropy_func(variables[8], variables[11], variables[12], variables[13], wf)
        else:  # real val: energy balance equation (isenthalpic expansion)
            t14_calc_valve = valve_func(variables[8], variables[12])
        #   12
        p15_calc_eva = x_saturation_func(1, variables[9], variables[15], wf)
        #   13
        t14_set = temperature_func(t14, variables[12], variables[13], wf)
        #   14
        p22_set = pr_func(pr_cond_cold, p21, variables[14])
        #   15
        p14_set = pr_func(pr_eva_cold, variables[13], variables[15])
        #   16
        cond_eco_outlet_sat = x_saturation_func(1, variables[16], variables[20], wf)
        #   17
        cond_eco_en_bal = he_func(variables[4], variables[1], variables[16], m21, variables[19], variables[7])
        #   18
        cond_sh_inlet_sat = x_saturation_func(0, variables[17], variables[21], wf)
        #   19
        cond_sh_en_bal = he_func(variables[4], variables[17], variables[3], m21, variables[6], variables[18])
        # 20
        p18_set = pr_func(pr_cond_part_hot, variables[2], variables[20])
        # 21
        p19_set = pr_func(pr_cond_part_hot, variables[20], variables[21])
        # 22
        p28_set = pr_func(pr_cond_part_cold, p21, variables[22])
        # 23
        p29_set = pr_func(pr_cond_part_cold, variables[22], variables[23])

        residual = np.array([t11_calc_comp, t16_set, t12_set, p11_set, power_calc_comp, m11_calc_cond, t21_set, t22_set,
                             t13_calc_ihx, p16_set, p13_set, t14_calc_valve, p15_calc_eva, t14_set, p22_set, p14_set,
                             cond_eco_outlet_sat, cond_eco_en_bal, cond_sh_inlet_sat, cond_sh_en_bal, p18_set,
                             p19_set, p28_set, p29_set], dtype=float)
        jacobian = np.zeros((len(variables), len(variables)))

        #   0
        if adex and config['comp'] == 'ideal':  # ideal comp: epsilon_j = 1
            t11_calc_comp_j = eps_compressor_deriv(1, variables[0], variables[10], variables[1], variables[2], wf)
        elif adex and config['comp'] == 'real':  # adex real comp: epsilon from base case
            t11_calc_comp_j = eps_compressor_deriv(epsilon['comp'], variables[0], variables[10], variables[1], variables[2], wf)
        else:  # real or unavoid comp: given eta_s
            t11_calc_comp_j = eta_s_compressor_deriv(eta_s, variables[0], variables[10], variables[1], variables[2], wf)
        #   1
        if adex and config['ihx'] == 'ideal':  # ideal ihx: ttd_u_j = 0
            t16_set_j = same_temperature_deriv(variables[3], target_p12, wf, variables[0], variables[10], wf)  # t16_j = t12
        elif adex and config['ihx'] == 'real':  # adex real ihx: epsilon from base case
            t16_set_j = eps_real_ihx_deriv(epsilon['ihx'], variables[3], target_p12, variables[8], variables[11], wf,
                                           variables[9], variables[15], variables[0], variables[10], wf)
        else:  # real or unavoid ihx: given ttd_u
            t16_set_j = temperature_deriv(t16, variables[0], variables[10], wf)
        #   2
        if adex and config['cond'] == 'real':  # adex real cond: epsilon from base case
            t12_set_j = eps_real_he_deriv(epsilon['cond'], variables[1], variables[2], variables[3], target_p12, variables[4], wf,
                                          variables[6], p21, variables[7], variables[14], m21, fluid_tes)
        else:  # ideal, real, unavoid cond: given ttd_l (different from cond, because temperature is known in this case)
            t12_set_j = temperature_deriv(t12, variables[3], target_p12, wf)
        #   3
        p11_set_j = pr_deriv(pr_cond_hot, variables[2], target_p12)
        #   4
        power_calc_comp_j = turbo_deriv(variables[5], variables[4], variables[0], variables[1])
        #   5
        if adex and config['cond'] == 'ideal':  # mass flow from ideal case: entropy balance equation
            m11_calc_cond_j = ideal_he_entropy_deriv(variables[1], variables[2], variables[3], target_p12, variables[4], wf,
                                                     variables[6], p21, variables[7], variables[14], m21, fluid_tes)
        else:  # mass flow from real and unavoid case: energy balance equation
            m11_calc_cond_j = he_deriv(variables[4], variables[1], variables[3], m21, variables[6], variables[7])
        #   6
        t21_set_j = temperature_deriv(t21, variables[6], p21, fluid_tes)
        #   7
        t22_set_j = temperature_deriv(t22, variables[7], variables[14], fluid_tes)
        #   8
        if adex and config['ihx'] == 'ideal':  # ideal ihx: entropy balance equation for upper state
            t13_calc_ihx_j = ideal_ihx_entropy_deriv(variables[3], target_p12, variables[8], variables[11], wf,
                                                     variables[9], variables[15], variables[0], variables[10], wf)
        else:  # real or unavoid ihx: energy balance equation for upper state
            t13_calc_ihx_j = ihx_deriv(variables[3], variables[8], variables[9], variables[0])
        #   9
        p16_set_j = pr_deriv(pr_ihx_cold, variables[15], variables[10])
        #   10
        p13_set_j = pr_deriv(pr_ihx_hot, target_p12, variables[11])
        #   11
        if adex and config['val'] == 'ideal':  # ideal val: entropy balance equation (isentropic expansion)
            t14_calc_valve_j = ideal_valve_entropy_deriv(variables[8], variables[11], variables[12], variables[13], wf)
        else:  # real val: energy balance equation (isenthalpic expansion)
            t14_calc_valve_j = valve_deriv(variables[8], variables[12])
        #   12
        p15_calc_eva_j = x_saturation_deriv(1, variables[9], variables[15], wf)
        #   13
        t14_set_j = temperature_deriv(t14, variables[12], variables[13], wf)
        #   14
        p22_set_j = pr_deriv(pr_cond_cold, p21, variables[14])
        #   15
        p14_set_j = pr_deriv(pr_eva_cold, variables[13], variables[15])
        #   16
        cond_eco_outlet_sat_j = x_saturation_deriv(1, variables[16], variables[20], wf)
        #   17
        cond_eco_en_bal_j = he_deriv(variables[4], variables[1], variables[16], m21, variables[19], variables[7])
        #   18
        cond_sh_inlet_sat_j = x_saturation_deriv(0, variables[17], variables[21], wf)
        #   19
        cond_sh_en_bal_j = he_deriv(variables[4], variables[17], variables[3], m21, variables[6], variables[18])
        # 20
        p18_set_j = pr_deriv(pr_cond_part_hot, variables[2], variables[20])
        # 21
        p19_set_j = pr_deriv(pr_cond_part_hot, variables[20], variables[21])
        # 22
        p28_set_j = pr_deriv(pr_cond_part_cold, p21, variables[22])
        # 23
        p29_set_j = pr_deriv(pr_cond_part_cold, variables[22], variables[23])

        #      [h16, h11, p11, h12, m11, power, h21, h22, h13, h15, p16, p13, h14, p14, p22, p15,
        #        0    1    2    3    4     5     6    7    8    9    10   11   12   13   14   15
        #       h18, h19, h28, h29, p18, p19, p28, p29])]
        #        16   17   18   19   20   21   22   23

        jacobian[0, 0] = t11_calc_comp_j['h_1']  # derivative of t11_calc_comp with respect to h16
        jacobian[0, 10] = t11_calc_comp_j['p_1']  # derivative of t11_calc_comp with respect to p16
        jacobian[0, 1] = t11_calc_comp_j['h_2']  # derivative of t11_calc_comp with respect to h11
        jacobian[0, 2] = t11_calc_comp_j['p_2']  # derivative of t11_calc_comp with respect to p11
        if adex and config['ihx'] == 'ideal':
            jacobian[1, 3] = t16_set_j['h_copy']  # derivative of t16_set with respect to h16
            jacobian[1, 0] = t16_set_j['h_paste']  # derivative of t16_set with respect to p16
            jacobian[1, 10] = t16_set_j['p_paste']  # derivative of t16_set with respect to h16
        elif adex and config['ihx'] == 'real':
            jacobian[1, 3] = t16_set_j['h_hot_in']  # derivative of t16_set with respect to h16
            jacobian[1, 8] = t16_set_j['h_hot_out']  # derivative of t16_set with respect to p16
            jacobian[1, 11] = t16_set_j['p_hot_out']  # derivative of t16_set with respect to h16
            jacobian[1, 9] = t16_set_j['h_cold_in']  # derivative of t16_set with respect to p16
            jacobian[1, 15] = t16_set_j['p_cold_in']  # derivative of t16_set with respect to h16
            jacobian[1, 0] = t16_set_j['h_cold_out']  # derivative of t16_set with respect to p16
            jacobian[1, 10] = t16_set_j['p_cold_out']  # derivative of t16_set with respect to h16
        else:
            jacobian[1, 0] = t16_set_j['h']  # derivative of t16_set with respect to h16
            jacobian[1, 10] = t16_set_j['p']  # derivative of t16_set with respect to p16
        if adex and config['cond'] == 'real':
            jacobian[2, 1] = t12_set_j['h_hot_in']  # derivative of m11_calc_cond with respect to h11
            jacobian[2, 2] = t12_set_j['p_hot_in']  # derivative of m11_calc_cond with respect to h12
            jacobian[2, 3] = t12_set_j['h_hot_out']  # derivative of m11_calc_cond with respect to m11
            jacobian[2, 4] = t12_set_j['m_hot']  # derivative of m11_calc_cond with respect to h21
            jacobian[2, 6] = t12_set_j['h_cold_in']  # derivative of m11_calc_cond with respect to h22
            jacobian[2, 7] = t12_set_j['h_cold_out']  # derivative of m11_calc_cond with respect to h22
            jacobian[2, 14] = t12_set_j['p_cold_out']  # derivative of m11_calc_cond with respect to h22
        else:
            jacobian[2, 3] = t12_set_j['h']  # derivative of t12_set with respect to h12
        jacobian[3, 2] = p11_set_j['p_1']  # derivative of p11_set with respect to p11
        jacobian[4, 5] = power_calc_comp_j['P']  # derivative of power_calc_comp with respect to power
        jacobian[4, 4] = power_calc_comp_j['m']  # derivative of power_calc_comp with respect to m11
        jacobian[4, 0] = power_calc_comp_j['h_1']  # derivative of power_calc_comp with respect to h16
        jacobian[4, 1] = power_calc_comp_j['h_2']  # derivative of power_calc_comp with respect to h11
        if adex and config['cond'] == 'ideal':
            jacobian[5, 1] = m11_calc_cond_j['h_hot_in']  # derivative of m11_calc_cond with respect to h11
            jacobian[5, 2] = m11_calc_cond_j['p_hot_in']  # derivative of m11_calc_cond with respect to h12
            jacobian[5, 3] = m11_calc_cond_j['h_hot_out']  # derivative of m11_calc_cond with respect to h12
            jacobian[5, 4] = m11_calc_cond_j['m_hot']  # derivative of m11_calc_cond with respect to m11
            jacobian[5, 6] = m11_calc_cond_j['h_cold_in']  # derivative of m11_calc_cond with respect to h21
            jacobian[5, 7] = m11_calc_cond_j['h_cold_out']  # derivative of m11_calc_cond with respect to h22
            jacobian[5, 14] = m11_calc_cond_j['p_cold_out']  # derivative of m11_calc_cond with respect to h12
        else:
            jacobian[5, 1] = m11_calc_cond_j['h_1']  # derivative of m11_calc_cond with respect to h11
            jacobian[5, 3] = m11_calc_cond_j['h_2']  # derivative of m11_calc_cond with respect to h12
            jacobian[5, 4] = m11_calc_cond_j['m_hot']  # derivative of m11_calc_cond with respect to m11
            jacobian[5, 6] = m11_calc_cond_j['h_3']  # derivative of m11_calc_cond with respect to h21
            jacobian[5, 7] = m11_calc_cond_j['h_4']  # derivative of m11_calc_cond with respect to h22
        jacobian[6, 6] = t21_set_j['h']  # derivative of t21_set with respect to h21
        jacobian[7, 7] = t22_set_j['h']  # derivative of t22_set with respect to h22
        jacobian[7, 14] = t22_set_j['p']  # derivative of t22_set with respect to p22
        if adex and config['ihx'] == 'ideal':
            jacobian[8, 3] = t13_calc_ihx_j['h_hot_in']  # derivative of t16_set with respect to h16
            jacobian[8, 8] = t13_calc_ihx_j['h_hot_out']  # derivative of t16_set with respect to p16
            jacobian[8, 11] = t13_calc_ihx_j['p_hot_out']  # derivative of t16_set with respect to h16
            jacobian[8, 9] = t13_calc_ihx_j['h_cold_in']  # derivative of t16_set with respect to p16
            jacobian[8, 15] = t13_calc_ihx_j['p_cold_in']  # derivative of t16_set with respect to h16
            jacobian[8, 0] = t13_calc_ihx_j['h_cold_out']  # derivative of t16_set with respect to p16
            jacobian[8, 10] = t13_calc_ihx_j['p_cold_out']  # derivative of t16_set with respect to h16
        else:
            jacobian[8, 3] = t13_calc_ihx_j['h_1']  # derivative of t13_calc_ihx with respect to h12
            jacobian[8, 8] = t13_calc_ihx_j['h_2']  # derivative of t13_calc_ihx with respect to h13
            jacobian[8, 9] = t13_calc_ihx_j['h_3']  # derivative of t13_calc_ihx with respect to h15
            jacobian[8, 0] = t13_calc_ihx_j['h_4']  # derivative of t13_calc_ihx with respect to h16
        jacobian[9, 10] = p16_set_j['p_2']  # derivative of p16_set with respect to p16
        jacobian[9, 15] = p16_set_j['p_1']  # derivative of p16_set with respect to p15
        jacobian[10, 11] = p13_set_j['p_2']  # derivative of p13_set with respect to p13
        if adex and config['val'] == 'ideal':
            jacobian[11, 8] = t14_calc_valve_j['h_1']  # derivative of t14_calc_valve with respect to h13
            jacobian[11, 11] = t14_calc_valve_j['p_1']  # derivative of t14_calc_valve with respect to p13
            jacobian[11, 12] = t14_calc_valve_j['h_2']  # derivative of t14_calc_valve with respect to h14
            jacobian[11, 13] = t14_calc_valve_j['p_2']  # derivative of t14_calc_valve with respect to p14
        else:
            jacobian[11, 8] = t14_calc_valve_j['h_1']  # derivative of t14_calc_valve with respect to h13
            jacobian[11, 12] = t14_calc_valve_j['h_2']  # derivative of t14_calc_valve with respect to h14
        jacobian[12, 9] = p15_calc_eva_j['h']  # derivative of p15_calc_eva with respect to h15
        jacobian[12, 15] = p15_calc_eva_j['p']  # derivative of p15_calc_eva with respect to p15
        jacobian[13, 12] = t14_set_j['h']  # derivative of t14_set with respect to h14
        jacobian[13, 13] = t14_set_j['p']  # derivative of t14_set with respect to p14
        jacobian[14, 14] = p22_set_j['p_2']  # derivative of p22_set with respect to p22
        jacobian[15, 13] = p14_set_j['p_1']  # derivative of p14_set with respect to p14
        jacobian[15, 15] = p14_set_j['p_2']  # derivative of p14_set with respect to p15
        jacobian[16, 16] = cond_eco_outlet_sat_j['h']
        jacobian[16, 20] = cond_eco_outlet_sat_j['p']
        jacobian[17, 4] = cond_eco_en_bal_j['m_hot']
        jacobian[17, 1] = cond_eco_en_bal_j['h_1']
        jacobian[17, 16] = cond_eco_en_bal_j['h_2']
        jacobian[17, 19] = cond_eco_en_bal_j['h_3']
        jacobian[17, 7] = cond_eco_en_bal_j['h_4']
        jacobian[18, 17] = cond_sh_inlet_sat_j['h']
        jacobian[18, 21] = cond_sh_inlet_sat_j['p']
        jacobian[19, 4] = cond_sh_en_bal_j['m_hot']
        jacobian[19, 17] = cond_sh_en_bal_j['h_1']
        jacobian[19, 6] = cond_sh_en_bal_j['h_2']
        jacobian[19, 3] = cond_sh_en_bal_j['h_3']
        jacobian[19, 18] = cond_sh_en_bal_j['h_4']
        jacobian[20, 2] = p18_set_j['p_1']
        jacobian[20, 20] = p18_set_j['p_2']
        jacobian[21, 20] = p19_set_j['p_1']
        jacobian[21, 21] = p19_set_j['p_2']
        jacobian[22, 22] = p28_set_j['p_2']
        jacobian[23, 22] = p29_set_j['p_1']
        jacobian[23, 23] = p29_set_j['p_2']

        # Save the DataFrame as a CSV file
        # pd.DataFrame(jacobian).round(4).to_csv('jacobian_hp.csv', index=False)

        variables -= np.linalg.inv(jacobian).dot(residual)

        iter_step += 1

        cond_number = np.linalg.cond(jacobian)
        # print('Condition number: ', cond_number, ' and residual: ', np.linalg.norm(residual))
        # print(variables)
    #      [h16, h11, p11, h12, m11, power, h21, h22, h13, h15, p16, p13, h14, p14, p22, p15,
    #        0    1    2    3    4     5     6    7    8    9    10   11   12   13   14   15
    #       h18, h19, h28, h29, p18, p19, p28, p29])]
    #        16   17   18   19   20   21   22   23

    p11 = variables[2]
    p12 = target_p12
    p13 = variables[11]
    p14 = variables[13]
    p15 = variables[15]
    p16 = variables[10]
    p22 = variables[14]
    p18 = variables[20]
    p19 = variables[21]
    p28 = variables[22]
    p29 = variables[23]

    h11 = variables[1]
    h12 = variables[3]
    h13 = variables[8]
    h14 = variables[12]
    h15 = variables[9]
    h16 = variables[0]
    h21 = variables[6]
    h22 = variables[7]
    h18 = variables[16]
    h19 = variables[17]
    h28 = variables[18]
    h29 = variables[19]

    t11 = PSI('T', 'H', h11, 'P', p11, wf)
    t12 = PSI('T', 'H', h12, 'P', p12, wf)
    t13 = PSI('T', 'H', h13, 'P', p13, wf)
    t14 = PSI('T', 'H', h14, 'P', p14, wf)
    t15 = PSI('T', 'H', h15, 'P', p15, wf)
    t16 = PSI('T', 'H', h16, 'P', p16, wf)
    t21 = PSI('T', 'H', h21, 'P', p21, fluid_tes)
    t22 = PSI('T', 'H', h22, 'P', p22, fluid_tes)
    t18 = PSI('T', 'H', h18, 'P', p18, wf)
    t19 = PSI('T', 'H', h19, 'P', p19, wf)
    t28 = PSI('T', 'H', h28, 'P', p28, fluid_tes)
    t29 = PSI('T', 'H', h29, 'P', p29, fluid_tes)

    s11 = PSI('S', 'H', h11, 'P', p11, wf)
    s12 = PSI('S', 'H', h12, 'P', p12, wf)
    s13 = PSI('S', 'H', h13, 'P', p13, wf)
    s14 = PSI('S', 'H', h14, 'P', p14, wf)
    s15 = PSI('S', 'H', h15, 'P', p15, wf)
    s16 = PSI('S', 'H', h16, 'P', p16, wf)
    s21 = PSI('S', 'H', h21, 'P', p21, fluid_tes)
    s22 = PSI('S', 'H', h22, 'P', p22, fluid_tes)
    s18 = PSI('S', 'H', h18, 'P', p18, wf)
    s19 = PSI('S', 'H', h19, 'P', p19, wf)
    s28 = PSI('S', 'H', h28, 'P', p28, fluid_tes)
    s29 = PSI('S', 'H', h29, 'P', p29, fluid_tes)

    m11 = variables[4]

    df_streams = pd.DataFrame(index=[11, 12, 13, 14, 15, 16, 21, 22, 18, 19, 28, 29],
                              columns=['m [kg/s]', 'T [°C]', 'h [kJ/kg]', 'p [bar]', 's [J/kgK]', 'fluid'])
    df_streams.loc[11] = [m11, t11, h11, p11, s11, wf]
    df_streams.loc[12] = [m11, t12, h12, p12, s12, wf]
    df_streams.loc[13] = [m11, t13, h13, p13, s13, wf]
    df_streams.loc[14] = [m11, t14, h14, p14, s14, wf]
    df_streams.loc[15] = [m11, t15, h15, p15, s15, wf]
    df_streams.loc[16] = [m11, t16, h16, p16, s16, wf]
    df_streams.loc[21] = [m21, t21, h21, p21, s21, fluid_tes]
    df_streams.loc[22] = [m21, t22, h22, p21, s22, fluid_tes]
    df_streams.loc[18] = [m11, t18, h18, p18, s18, wf]
    df_streams.loc[19] = [m11, t19, h19, p19, s19, wf]
    df_streams.loc[28] = [m21, t28, h28, p28, s28, fluid_tes]
    df_streams.loc[29] = [m21, t29, h29, p29, s29, fluid_tes]

    df_streams['T [°C]'] = df_streams['T [°C]'] - 273.15
    df_streams['h [kJ/kg]'] = df_streams['h [kJ/kg]'] * 1e-3
    df_streams['p [bar]'] = df_streams['p [bar]'] * 1e-5

    if print_results:
        print('-------------------------------------------------------------\n', 'case:', label)
        print(df_streams.iloc[:, :5])
        print('-------------------------------------------------------------')

    power_comp = df_streams.loc[11, 'm [kg/s]'] * (df_streams.loc[11, 'h [kJ/kg]'] - df_streams.loc[16, 'h [kJ/kg]'])
    heat_eva = df_streams.loc[11, 'm [kg/s]'] * (df_streams.loc[15, 'h [kJ/kg]'] - df_streams.loc[14, 'h [kJ/kg]'])
    heat_cond = df_streams.loc[21, 'm [kg/s]'] * (df_streams.loc[22, 'h [kJ/kg]'] - df_streams.loc[21, 'h [kJ/kg]'])
    power_cond = (df_streams.loc[11, 'm [kg/s]'] * (df_streams.loc[11, 'h [kJ/kg]'] - df_streams.loc[12, 'h [kJ/kg]'])
                  + df_streams.loc[21, 'm [kg/s]'] * (df_streams.loc[21, 'h [kJ/kg]'] - df_streams.loc[22, 'h [kJ/kg]']))
    power_ihx = df_streams.loc[11, 'm [kg/s]'] * (df_streams.loc[12, 'h [kJ/kg]'] - df_streams.loc[13, 'h [kJ/kg]'] +
                                                  df_streams.loc[15, 'h [kJ/kg]'] - df_streams.loc[16, 'h [kJ/kg]'])
    power_val = df_streams.loc[11, 'm [kg/s]'] * (df_streams.loc[13, 'h [kJ/kg]'] - df_streams.loc[14, 'h [kJ/kg]'])

    if abs(power_comp+heat_eva-heat_cond-power_cond-power_ihx-power_val) > 1e-4:
        print('Energy balances are not fulfilled! :(')

    if plot:
        [min_td_cond, max_td_cond] = qt_diagram(df_streams, 'COND', 11, 12, 21, 22, ttd_l_cond, 'HP',
                   plot=plot, case=f'{label}', step_number=100, path=f'outputs/diagrams/adex_hp_qt_cond_{label}.png')
        # [min_td_cond_sh, max_td_cond_sh] = qt_diagram(df_streams, 'COND-SH', 11, 18, 29, 22, ttd_l_cond, 'HP',
                   #plot=plot, case=f'{label}', step_number=100)
        # [min_td_cond_eco, max_td_cond_eco] = qt_diagram(df_streams, 'COND-ECO', 19, 12, 21, 28, ttd_l_cond, 'HP',
                   # plot=plot, case=f'{label}', step_number=100)
        [min_td_cond_eva, max_td_cond_eva] = qt_diagram(df_streams, 'COND-EVA', 18, 19, 28, 29, ttd_l_eva, 'HP',
                   plot=plot, case=f'{label}', step_number=100, path=f'outputs/diagrams/adex_hp_qt_eva_{label}.png')
        qt_diagram(df_streams, 'IHX', 12, 13, 15, 16, ttd_u_ihx, 'HP',
                   plot=plot, case=f'{label}', step_number=100, path=f'outputs/diagrams/adex_hp_qt_ihx_{label}.png')

    t_pinch_cond_sh = t18-t29

    cop = heat_cond/(power_comp-power_ihx-power_val-power_cond)

    for col in df_streams.columns[:5]:
        df_streams[col] = pd.to_numeric(df_streams[col], errors='coerce')

    return df_streams, t_pinch_cond_sh, cop


def exergy_analysis_hp(t0, p0, df):
    """
    Conducts exergy analysis on a heat pump system using the thermodynamic properties of the cycle's state points.

    This function calculates the physical exergy, exergy destruction, and exergy efficiency of each component
    within a heat pump system, based on the inlet temperatures, pressures, and the thermodynamic states of
    the working fluid and the secondary fluid (typically water) in thermal energy storage (TES).
    The analysis helps in identifying the major sources of inefficiency within the system.

    Parameters:
    - t0 (float): Ambient temperature in Kelvin.
    - p0 (float): Ambient pressure in Pa.
    - df (DataFrame): A DataFrame containing the mass flow rates (m [kg/s]), temperatures (T [°C]),
      enthalpies (h [kJ/kg]), pressures (p [bar]), entropies (s [J/kgK]), and fluids for each state point in the cycle.

    Returns:
    - df_comps (DataFrame): A DataFrame with the exergy flow (EF [kW]), exergy destruction (ED [kW]), exergy
      performance (EP [kW]), exergy efficiency (epsilon), power (P [kW]), and heat (Q [kW]) for each component of the heat pump system.
    """

    h0_wf = PSI('H', 'T', t0, 'P', p0, df.loc[11, 'fluid']) * 1e-3
    s0_wf = PSI('S', 'T', t0, 'P', p0, df.loc[11, 'fluid'])
    h0_tes = PSI('H', 'T', t0, 'P', p0, df.loc[21, 'fluid']) * 1e-3
    s0_tes = PSI('S', 'T', t0, 'P', p0, df.loc[21, 'fluid'])

    for i in [11, 12, 13, 14, 15, 16, 18, 19]:
        df.loc[i, 'e^PH [kJ/kg]'] = df.loc[i, 'h [kJ/kg]'] - h0_wf - t0 * (df.loc[i, 's [J/kgK]'] - s0_wf) * 1e-3
    for i in [21, 22, 28, 29]:
        df.loc[i, 'e^PH [kJ/kg]'] = df.loc[i, 'h [kJ/kg]'] - h0_tes - t0 * (df.loc[i, 's [J/kgK]'] - s0_tes) * 1e-3

    # ED_COMP = T0 * (s11 - s16)
    ed_comp = t0 * (df.loc[11, 'm [kg/s]'] * (df.loc[11, 's [J/kgK]'] - df.loc[16, 's [J/kgK]'])) * 1e-3
    # ED_COND = T0 * (s12 - s11 + S22 - S11)
    ed_cond = t0 * (df.loc[11, 'm [kg/s]'] * (df.loc[12, 's [J/kgK]'] - df.loc[11, 's [J/kgK]']) +
                    df.loc[21, 'm [kg/s]'] * (df.loc[22, 's [J/kgK]'] - df.loc[21, 's [J/kgK]'])) * 1e-3
    # ED_IHX = T0 * (s13 - s12 + s16 - s15)
    ed_ihx = t0 * (df.loc[11, 'm [kg/s]'] * (df.loc[13, 's [J/kgK]'] - df.loc[12, 's [J/kgK]']) +
                   df.loc[11, 'm [kg/s]'] * (df.loc[16, 's [J/kgK]'] - df.loc[15, 's [J/kgK]'])) * 1e-3
    # ED_VAL = T0 * (s14 - s13)
    ed_val = t0 * (df.loc[11, 'm [kg/s]'] * (df.loc[14, 's [J/kgK]'] - df.loc[13, 's [J/kgK]'])) * 1e-3
    # ED_EVA = E14 - E15
    ed_eva = (df.loc[11, 'm [kg/s]'] * (df.loc[14, 'e^PH [kJ/kg]'] - df.loc[15, 'e^PH [kJ/kg]']))

    ed = {
        'comp': ed_comp,
        'cond': ed_cond,
        'ihx': ed_ihx,
        'val': ed_val,
        'eva': ed_eva,
        'tot': ed_comp+ed_cond+ed_ihx+ed_val+ed_eva
    }

    ef_ihx = df.loc[11, 'm [kg/s]'] * (df.loc[12, 'e^PH [kJ/kg]'] - df.loc[13, 'e^PH [kJ/kg]'])
    ef_cond = df.loc[11, 'm [kg/s]'] * (df.loc[11, 'e^PH [kJ/kg]'] - df.loc[12, 'e^PH [kJ/kg]'])
    ef_comp = df.loc[11, 'm [kg/s]'] * (df.loc[11, 'h [kJ/kg]'] - df.loc[16, 'h [kJ/kg]'])

    epsilon_comp = 1 - ed_comp / ef_comp
    epsilon_cond = 1 - ed_cond / ef_cond
    epsilon_ihx = 1 - ed_ihx / ef_ihx
    if abs(df.loc[13, 's [J/kgK]'] - df.loc[14, 's [J/kgK]']) < 1e-3:  # --> ideal expander instead of valve
        ef_val = df.loc[11, 'm [kg/s]'] * (df.loc[13, 'e^PH [kJ/kg]'] - df.loc[14, 'e^PH [kJ/kg]'])
        epsilon_val = 1 - ed_val / ef_val
    else:  # --> real valve
        if df.loc[14, 'T [°C]'] > (t0-273.15):  # --> dissipative valve
            epsilon_val = np.nan
            ef_val = np.nan
        else:
            h13a = PSI('H', 'T', t0, 'P', df.loc[13, 'p [bar]'] * 1e5, df.loc[13, 'fluid']) * 1e-3
            s13a = PSI('S', 'T', t0, 'P', df.loc[13, 'p [bar]'] * 1e5, df.loc[13, 'fluid'])
            h14a = PSI('H', 'T', t0, 'P', df.loc[14, 'p [bar]'] * 1e5, df.loc[14, 'fluid']) * 1e-3
            s14a = PSI('S', 'T', t0, 'P', df.loc[14, 'p [bar]'] * 1e5, df.loc[14, 'fluid'])
            e13t = df.loc[13, 'h [kJ/kg]'] - h13a - t0 * (df.loc[13, 's [J/kgK]'] - s13a) * 1e-3
            e14t = df.loc[14, 'h [kJ/kg]'] - h14a - t0 * (df.loc[14, 's [J/kgK]'] - s14a) * 1e-3
            e13m = h13a - h0_wf - t0 * (s13a - s0_wf) * 1e-3
            e14m = h14a - h0_wf - t0 * (s14a - s0_wf) * 1e-3
            ef_val = (e13t + e13m - e14m) * df.loc[11, 'm [kg/s]']
            epsilon_val = e14t / (e13t + e13m - e14m)

    heat_eva = df.loc[11, 'm [kg/s]'] * (df.loc[15, 'h [kJ/kg]'] - df.loc[14, 'h [kJ/kg]'])
    heat_cond = df.loc[21, 'm [kg/s]'] * (df.loc[22, 'h [kJ/kg]'] - df.loc[21, 'h [kJ/kg]'])
    heat_ihx = df.loc[11, 'm [kg/s]'] * (df.loc[16, 'h [kJ/kg]'] - df.loc[15, 'h [kJ/kg]'])
    power_comp = df.loc[11, 'm [kg/s]'] * (df.loc[11, 'h [kJ/kg]'] - df.loc[16, 'h [kJ/kg]'])
    power_cond = (df.loc[11, 'm [kg/s]'] * (df.loc[11, 'h [kJ/kg]'] - df.loc[12, 'h [kJ/kg]'])
                  + df.loc[21, 'm [kg/s]'] * (df.loc[21, 'h [kJ/kg]'] - df.loc[22, 'h [kJ/kg]']))
    power_ihx = df.loc[11, 'm [kg/s]'] * (df.loc[12, 'h [kJ/kg]'] - df.loc[13, 'h [kJ/kg]'] +
                                            df.loc[15, 'h [kJ/kg]'] - df.loc[16, 'h [kJ/kg]'])
    power_val = df.loc[11, 'm [kg/s]'] * (df.loc[13, 'h [kJ/kg]'] - df.loc[14, 'h [kJ/kg]'])

    exergy_efficiency = ((df.loc[21, 'm [kg/s]'] * (df.loc[22, 'e^PH [kJ/kg]'] - df.loc[21, 'e^PH [kJ/kg]']))
                         / (power_comp - power_ihx - power_val - power_cond))

    ef = {
        'comp': ef_comp,
        'cond': ef_cond,
        'ihx': ef_ihx,
        'val': ef_val,
        'eva': np.nan,
        'tot': power_comp - power_ihx - power_val - power_cond
    }

    epsilon = {
        'comp': epsilon_comp,
        'cond': epsilon_cond,
        'ihx': epsilon_ihx,
        'val': epsilon_val,
        'eva': np.nan,
        'tot': exergy_efficiency
    }

    df_comps = pd.DataFrame(index=['comp', 'cond', 'ihx', 'val', 'eva', 'tot'],
                            columns=['EF [kW]', 'EP [kW]', 'ED [kW]', 'epsilon', 'P [kW]', 'Q [kW]', 'unavoid ratio'])

    for k in ['comp', 'cond', 'ihx', 'val', 'eva', 'tot']:
        df_comps.loc[k, 'epsilon'] = epsilon[k]
        df_comps.loc[k, 'EF [kW]'] = ef[k]
        if df_comps.loc[k, 'EF [kW]'] is not None or df_comps.loc[k, 'epsilon'] is not None:
            df_comps.loc[k, 'EP [kW]'] = df_comps.loc[k, 'EF [kW]'] * df_comps.loc[k, 'epsilon']
        df_comps.loc[k, 'ED [kW]'] = ed[k]
        if df_comps.loc[k, 'EP [kW]'] is not None:
            df_comps.loc[k, 'unavoid ratio'] = df_comps.loc[k, 'ED [kW]'] / df_comps.loc[k, 'EP [kW]']

    df_comps.loc['val', 'unavoid ratio'] = df_comps.loc['val', 'ED [kW]'] / df.loc[11, 'm [kg/s]']
    df_comps.loc['eva', 'unavoid ratio'] = df_comps.loc['eva', 'ED [kW]'] / df.loc[11, 'm [kg/s]']

    df_comps.loc['comp', 'P [kW]'] = power_comp
    df_comps.loc['cond', 'P [kW]'] = power_cond
    df_comps.loc['ihx', 'P [kW]'] = power_ihx
    df_comps.loc['cond', 'Q [kW]'] = heat_cond
    df_comps.loc['ihx', 'Q [kW]'] = heat_ihx
    df_comps.loc['eva', 'Q [kW]'] = heat_eva
    df_comps.loc['val', 'P [kW]'] = power_val
    df_comps.loc['tot', 'P [kW]'] = power_comp-power_cond-power_val-power_ihx

    return df_comps


def set_adex_hp_config(comp='real', cond='real', ihx='real', val='real', eva='real'):
    """
    Generates a configuration dictionary and label for an advanced exergy analysis of a heat pump system based on the state of its components.

    This version of the function constructs the label with the following rules:
    1. If a component is 'real', it's mentioned by name in the label unless another component is 'unavoid'.
    2. If a component is 'ideal', it's not mentioned in the label.
    3. If a component is 'unavoid', it's mentioned by name with '_unavoid' appended.
    4. If there's any 'unavoid' component, 'real' components are not included in the label.
    5. If all components share the same state, the label is 'all_real', 'all_ideal', or 'all_unavoid' accordingly.

    Parameters:
    - comp, cond, ihx, val, eva (str): States of the compressor, condenser, internal heat exchanger, valve, and evaporator respectively.

    Returns:
    - config (dict): States of heat pump components.
    - label (str): Identifier for the configuration.
    """

    config = {
        'comp': comp,
        'cond': cond,
        'ihx': ihx,
        'val': val,
        'eva': eva
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


def find_opt_p12(p12_opt_start, min_t_diff_cond_start, config, label, adex=False, output_buffer=None):
    """
    Finds the optimal pressure at state point 12 to achieve a target minimum temperature difference in the condenser
    of a heat pump system through an iterative optimization process.

    This function iteratively adjusts the pressure at state point 12 based on the difference between the current
    minimum temperature difference in the condenser and the target minimum temperature difference. The goal is to
    optimize the system's performance, particularly focusing on the coefficient of performance (COP) and ensuring
    the condenser operates within specified temperature difference constraints.

    Parameters:
    - p12_opt_start (float): The starting value of pressure at state point 12 in Pa.
    - min_t_diff_cond_start (float): The starting value of the minimum temperature difference in the condenser.
    - config (dict): Configuration dictionary specifying components included in the simulation (e.g., condenser, IHX).
    - label (str): A label for the optimization case, used in logging.
    - adex (bool, optional): If True, uses advanced exergetic analysis parameters in the simulation. Defaults to False.
    - output_buffer (list, optional): A list to which the optimization process logs will be appended. If None, logs are not saved.

    Returns:
    - p12_opt (float): The optimized pressure at state point 12 in Pa, aiming to achieve the target minimum temperature
      difference in the condenser while maintaining or improving the COP.
    """

    buffer = io.StringIO()  # Create a new buffer

    with open('inputs/hp_simult_unavoid.json', 'r') as file:
        hp_config_unavoid = json.load(file)
    with open('inputs/hp_simult_base.json', 'r') as file:
        hp_config = json.load(file)

    if config['cond'] == 'unavoid':
        target_min_td_cond = hp_config_unavoid['heat_pump']['ttd_l_cond']  # K
    elif config['cond'] == 'real':
        target_min_td_cond = hp_config['heat_pump']['ttd_l_cond']  # K
    else:
        target_min_td_cond = 0  # K

    min_t_diff_cond = target_min_td_cond
    p12_opt = p12_opt_start
    tolerance = 1e-3  # relative to min temperature difference
    learning_rate = 1e4  # relative to p12
    diff = min_t_diff_cond_start - target_min_td_cond
    step = 0

    while abs(diff) > tolerance:
        # Adjust p12 based on the difference
        adjustment = (target_min_td_cond - min_t_diff_cond) * learning_rate
        # adjustment is the smaller, the smaller the difference target_min_td_cond - min_td_cond
        p12_opt += adjustment

        [_, min_t_diff_cond, cop] = solve_hp(p12_opt, print_results=False, label=label, config=config, adex=adex)
        diff = abs(min_t_diff_cond - target_min_td_cond)

        step += 1
        buffer.write(f'Optimization in progress for {label}: step = {step}, diff = {round(diff,6)}, p12 = {round(p12_opt*1e-5, 4)} bar, COP = {round(cop, 4)}.\n')

    buffer.write(f'Optimization completed successfully in {step} steps!\n')
    buffer.write(f'Optimal p12: {round(p12_opt*1e-5, 4)}, bar.\n')

    if output_buffer is not None:
        output_buffer.append(buffer.getvalue())

    return p12_opt


def perform_adex_hp(p12_start, config, label, df_ed, adex=False, save=False, print_results=False, calc_epsilon=False, output_buffer=None):
    """
    Executes an advanced exergy analysis for a heat pump system, optimizing a specific target condition, and evaluating system performance.

    This function automates the process of optimizing the pressure at state point 12 based on a target condition, performing
    a heat pump system simulation under the optimized conditions, conducting exergy analysis to identify inefficiencies, and
    optionally calculating and saving exergy efficiency results.

    Parameters:
    - p12_start (float): Starting pressure at state point 12 in Pa, used as the initial guess for optimization.
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
    - Error messages are printed if there are negative temperature differences in the condenser or internal heat exchanger (IHX),
      indicating an issue with the optimization or simulation setup.
    """

    # Run first simulation and get pinch point in condenser (target_diff)
    [_, target_diff, _] = solve_hp(p12_start, print_results=print_results, config=config, label=label, adex=adex)
    # Search for optimal p12 according to minimum pinch point in cond
    p12_opt = find_opt_p12(p12_start, target_diff, config=config, label=label, adex=adex, output_buffer=output_buffer)
    # Run simulation with optimal p12 and minimum pinch point in cond
    [df_opt, _, _] = solve_hp(p12_opt, print_results=print_results, config=config, label=f'optimal {label}', adex=adex)

    # Run conventional exergy analysis to get exergetic efficiencies and exergy destruction rates
    with open('inputs/hp_simult_base.json', 'r') as file:
        hp_config = json.load(file)
    t0 = hp_config['ambient']['t0']  # K
    p0 = hp_config['ambient']['p0']  # Pa
    df_components = exergy_analysis_hp(t0, p0, df_opt)

    for component in df_components.index:
        df_ed.loc[(label, component), 'ED [kW]'] = df_components.loc[component, 'ED [kW]']
    if calc_epsilon:
        for col in df_components.columns[:5]:
            df_components[col] = pd.to_numeric(df_components[col], errors='coerce')
        if label == "all_real":
            df_components.to_csv(f'outputs/adex_hp/hp_comps_{label}.csv')
        else:
            df_components.round(4).to_csv(f'outputs/adex_hp/hp_comps_{label}.csv')
    if df_opt.loc[12, 'T [°C]'] < df_opt.loc[21, 'T [°C]'] - 1e-3:
        print('Error: Lower temperature difference in COND is negative! Check for errors!')
    if df_opt.loc[11, 'T [°C]'] < df_opt.loc[22, 'T [°C]'] - 1e-3:
        print('Error: Upper temperature difference in COND is negative! Check for errors!')
    if df_opt.loc[12, 'T [°C]'] < df_opt.loc[16, 'T [°C]'] - 1e-3:
        print('Error: Upper temperature difference in IHX is negative! Check for errors!')
    if df_opt.loc[13, 'T [°C]'] < df_opt.loc[15, 'T [°C]'] - 1e-3:
        print('Error: Lower temperature difference in IHX is negative! Check for errors!')
    if save:
        round(df_opt, 5).to_csv(f'outputs/adex_hp/hp_streams_{label}.csv')

    return df_ed


def main_serial():
    """
    The main function for running a serial (sequential) execution of advanced exergy analysis (ADEX) on a heat pump (HP) system.

    This function sequentially executes the ADEX analysis for different configurations of a HP system, starting with the base case,
    followed by the ideal case, single components in real conditions, and pairs of components in real conditions. Each configuration's
    exergy destruction is calculated, and the results are saved and optionally printed. Avoidable/Unavoidable is not
    considered. In this case, please run the multiprocess function. This function should be used only for debugging.

    The function:
    - Sets up configurations for the base case, ideal case, individual components, and pairs of components.
    - Executes the perform_adex_hp function for each configuration to perform the ADEX analysis.
    - Saves the results in a DataFrame which includes exergy destruction values for each component and configuration.
    - Optionally prints the results and calculates exergy efficiency for further analysis.

    Parameters:
    None

    Returns:
    None
    """

    columns = ['ED']
    multi_index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['Label', 'Component'])
    df_ed = pd.DataFrame(columns=columns, index=multi_index)

    # BASE CASE
    [config_base, label_base] = set_adex_hp_config('comp', 'cond', 'ihx', 'val', 'eva')
    perform_adex_hp(13e5, config_base, label_base, df_ed, adex=False, save=True, print_results=True, calc_epsilon=True)

    # IDEAL CASE
    [config_ideal, label_ideal] = set_adex_hp_config()
    perform_adex_hp(13e5, config_ideal, label_ideal, df_ed, adex=False, save=True, print_results=True, calc_epsilon=True)

    # ADVANCED EXERGY ANALYSIS -- single components
    components = ['comp', 'cond', 'ihx', 'val', 'eva']
    for component in components:
        config_i, label_i = set_adex_hp_config(component)
        perform_adex_hp(13e5, config_i, label_i, df_ed, adex=True, save=True, print_results=True, calc_epsilon=True)

    # ADVANCED EXERGY ANALYSIS -- pair of components
    component_pairs = list(itertools.combinations(components, 2))
    for pair in component_pairs:
        config_i, label_i = set_adex_hp_config(*pair)
        perform_adex_hp(13e5, config_i, label_i, df_ed, adex=True, save=True, print_results=True, calc_epsilon=True)


def run_hp_adex(config, label, df_ed, adex, save, print_results, calc_epsilon, output_buffer=None):
    """
    Runs the advanced exergy analysis for a heat pump system with a given configuration and label.

    This function performs the advanced exergy analysis by calling the `perform_adex_hp` function with a starting
    pressure of 1.3 MPa (13e5 Pa) at state point 12. It updates the DataFrame with the exergy destruction results
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

    df_ed = perform_adex_hp(13e5, config, label, df_ed, adex, save, print_results, calc_epsilon, output_buffer)
    return df_ed


def main_multiprocess():
    """
    The main function for running a multiprocessed execution of advanced exergy analysis (ADEX) on a heat pump (HP) system.

    This function initializes the data structures for storing results, sets up multiprocessing tasks for different configurations
    of a HP system (including the base case, ideal case, individual components, and pairs of components), and executes these tasks
    in parallel using multiprocessing.Pool. It collects and concatenates the results from each process, prints the output stored
    during execution, and saves the final analysis results to a CSV file.

    It demonstrates an efficient approach to performing computationally intensive simulations and analyses by leveraging
    Python's multiprocessing capabilities to parallelize the work, significantly reducing the total computation time.

    Parameters:
    None

    Returns:
    None

    Note:
    - This function utilizes global variables for configuration and task setup.
    - It directly performs file I/O operations to save the analysis results.
    - Multiprocessing is used to parallelize the execution of simulation tasks, improving performance on multicore systems.
    """


    start = time.perf_counter()

    # Initialize the DataFrame for storing results
    columns = ['ED [kW]']
    multi_index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['Label', 'Component'])
    df_ed_base = pd.DataFrame(columns=columns, index=multi_index)
    df_ed_i = pd.DataFrame(columns=columns, index=multi_index)
    df_ed = pd.DataFrame(columns=columns, index=multi_index)

    # Shared data structure for output
    manager = multiprocessing.Manager()
    output_buffer = manager.list()

    # BASE CASE
    [config_base, label_base] = set_adex_hp_config()
    perform_adex_hp(13e5, config_base, label_base, df_ed_base, adex=False, save=True,
                    print_results=True, calc_epsilon=True, output_buffer=output_buffer)

    tasks = []

    # Create a pool of workers
    with multiprocessing.Pool() as pool:

        # IDEAL CASE
        config_ideal, label_ideal = set_adex_hp_config(comp='ideal', cond='ideal', ihx='ideal', val='ideal', eva='ideal')
        tasks.append((config_ideal, label_ideal, df_ed_i, True, True, True, True, output_buffer))

        components = ['comp', 'cond', 'ihx', 'val', 'eva']
        component_pairs = list(itertools.combinations(components, 2))

        # EN ED -- single components
        for component in components:
            config_kwargs = {comp: 'ideal' for comp in components}  # Start with all components as 'ideal'
            config_kwargs[component] = 'real'  # Set the current component to 'real'
            config_i, label_i = set_adex_hp_config(**config_kwargs)
            tasks.append((config_i, label_i, df_ed_i, True, True, True, True, output_buffer))

        # EN ED -- pair of components
        for pair in component_pairs:
            config_kwargs = {comp: 'ideal' for comp in components}  # Start with all components as 'ideal'
            for comp in pair:
                config_kwargs[comp] = 'real'  # Set each component in the pair to 'real'
            config_i, label_i = set_adex_hp_config(**config_kwargs)
            tasks.append((config_i, label_i, df_ed_i, True, True, True, True, output_buffer))

        # UN ED -- single components
        for component in components:
            config_kwargs = {comp: 'real' for comp in components}  # Start with all components as 'real'
            config_kwargs[component] = 'unavoid'  # Set the current component to 'unavoid'
            config_i, label_i = set_adex_hp_config(**config_kwargs)
            tasks.append((config_i, label_i, df_ed_i, False, True, True, True, output_buffer))

        # Map tasks to the pool
        results = pool.starmap(run_hp_adex, tasks)

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
    df_ed.round(6).to_csv('outputs/adex_hp/hp_adex_ed.csv')

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
    epsilon = pd.read_csv('outputs/adex_hp/hp_comps_all_real.csv', index_col=0)['epsilon']

    epsilon['eva'] = float('nan')

    for k in components:
        # EN / EX / binary interaction / MX
        df_k_real_streams = pd.read_csv(f'outputs/adex_hp/hp_streams_all_real.csv', index_col=0)
        df_k_endo = pd.read_csv(f'outputs/adex_hp/hp_streams_{k}.csv', index_col=0)
        df_adex_analysis.loc[(k, ''), 'ED [kW]'] = df_ed.loc[('all_real', k), 'ED [kW]']
        df_adex_analysis.loc[(k, ''), 'epsilon [%]'] = epsilon[k] * 100
        df_adex_analysis.loc[(k, ''), 'ED EN [kW]'] = df_ed.loc[(k, k), 'ED [kW]']
        df_adex_analysis.loc[(k, ''), 'ED EX [kW]'] = df_adex_analysis.loc[(k, ''), 'ED [kW]']-df_adex_analysis.loc[(k, ''), 'ED EN [kW]']
        df_adex_analysis.loc[(k, ''), 'm EN [kg/s]'] = df_k_endo.loc[11, 'm [kg/s]']
        df_adex_analysis.loc[(k, ''), 'm EX [kg/s]'] = df_k_real_streams.loc[11, 'm [kg/s]'] - df_k_endo.loc[11, 'm [kg/s]']
        sum_ed_ex_kl = 0
        for l in components:
            if k != l:
                k_l = f'{k}_{l}'
                if (k_l, l) not in df_ed.index:
                    k_l = f'{l}_{k}'
                df_k_l = pd.read_csv(f'outputs/adex_hp/hp_streams_{k_l}.csv', index_col=0)
                df_adex_analysis.loc[(k, l), 'm EN [kg/s]'] = df_k_l.loc[11, 'm [kg/s]']
                df_adex_analysis.loc[(k, l), 'm EX [kg/s]'] = df_k_real_streams.loc[11, 'm [kg/s]'] - df_k_l.loc[11, 'm [kg/s]']
                df_adex_analysis.loc[(k, l), 'ED kl [kW]'] = df_ed.loc[(k_l, k), 'ED [kW]']
                df_adex_analysis.loc[(k, l), 'ED kl [kW]'] = df_ed.loc[(k_l, k), 'ED [kW]']
                df_adex_analysis.loc[(k, l), 'ED EX kl [kW]'] = df_adex_analysis.loc[(k, l), 'ED kl [kW]'] - df_adex_analysis.loc[(k, ''), 'ED EN [kW]']
                sum_ed_ex_kl += df_adex_analysis.loc[(k, l), 'ED EX kl [kW]']
        df_adex_analysis.loc[(k, ''), 'ED MX [kW]'] = df_adex_analysis.loc[(k, ''), 'ED EX [kW]'] - sum_ed_ex_kl

        # UN / AV/ binary interaction
        df_k_unavoid_comps = pd.read_csv(f'outputs/adex_hp/hp_comps_{k}_unavoid.csv', index_col=0)
        df_k_endo_comps = pd.read_csv(f'outputs/adex_hp/hp_comps_{k}.csv', index_col=0)
        df_k_endo_streams = pd.read_csv(f'outputs/adex_hp/hp_streams_{k}.csv', index_col=0)
        df_k_real_comps = pd.read_csv(f'outputs/adex_hp/hp_comps_all_real.csv', index_col=0)
        if k == 'val' or k == 'eva':
            df_adex_analysis.loc[(k, ''), 'ED UN [kW]'] = df_k_real_streams.loc[11, 'm [kg/s]'] * df_k_unavoid_comps.loc[k, 'unavoid ratio']
        else:
            df_adex_analysis.loc[(k, ''), 'ED UN [kW]'] = df_k_real_comps.loc[k, 'EP [kW]'] * df_k_unavoid_comps.loc[k, 'unavoid ratio']
        df_adex_analysis.loc[(k, ''), 'ED AV [kW]'] = df_ed.loc[('all_real', k), 'ED [kW]'] - df_adex_analysis.loc[(k, ''), 'ED UN [kW]']
        if k == 'val' or k == 'eva':
            df_adex_analysis.loc[(k, ''), 'ED UN EN [kW]'] = df_k_endo_streams.loc[11, 'm [kg/s]'] * df_k_unavoid_comps.loc[k, 'unavoid ratio']
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

    df_adex_analysis.round(5).to_csv('outputs/adex_hp/hp_adex_analysis.csv')

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

# TEST OF ONE SINGLE SIMULATION
'''[config_test, label_test] = set_adex_hp_config('real', 'real', 'real', 'real', 'real')
[df_test, target_diff, cop_test] = solve_hp(target_p12=13e5, print_results=True, config=config_test, label=label_test, adex=False)
p12_opt = find_opt_p12(13e5, target_diff, config=config_test, label=label_test, adex=False)
[df_opt, _, _] = solve_hp(p12_opt, print_results=True, config=config_test, label=label_test, adex=False, plot=True)'''

'''[config_test, label_test] = set_adex_hp_config('real', 'real', 'real', 'real', 'real')
[df_test, target_diff, cop_test] = solve_hp(target_p12=13e5, print_results=True, config=config_test, label=label_test, adex=False)
p12_opt = find_opt_p12(13e5, target_diff, config=config_test, label=label_test, adex=False)
[df_opt, _, _] = solve_hp(p12_opt, print_results=True, config=config_test, label=label_test, adex=False, plot=True)
'''