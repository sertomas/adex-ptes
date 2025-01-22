import io
import os
import json
import itertools
import multiprocessing
import time

import numpy as np
import pandas as pd
from tabulate import tabulate as tabulate_func
from CoolProp.CoolProp import PropsSI as PSI

from functions import (pr_func, pr_deriv, eta_s_compressor_func, eta_s_compressor_deriv, turbo_func, turbo_deriv, he_func,
                       he_deriv, ihx_energy_bal, ihx_energy_bal_deriv, temperature_func, temperature_deriv, valve_func, valve_deriv,
                       x_saturation_func, x_saturation_deriv, qt_diagram, eps_compressor_func, eps_compressor_deriv,
                       eps_real_he_func, eps_real_he_deriv, eps_real_ihx_energy_bal, eps_real_ihx_energy_bal_deriv, simple_he_func,
                       simple_he_deriv, ideal_ihx_entropy_func, ideal_ihx_entropy_deriv, ideal_he_entropy_func,
                       ideal_he_entropy_deriv, ideal_valve_entropy_func, ideal_valve_entropy_deriv, he_with_p_func,
                       he_with_p_deriv, same_temperature_func, same_temperature_deriv, ttd_temperature_func,
                       ttd_temperature_deriv, store_cop_in_csv)


def check_minimum_temperature_differences_hp(df_streams, min_temps, case):
    """
    Checks if the temperature differences in the heat exchangers meet the minimum values.

    Parameters:
    - df_streams (DataFrame): Stream data containing temperatures and other thermodynamic properties.
    - min_temps (dict): Minimum temperature differences for each heat exchanger (ideal, real, or unavoid).
    - case (str): Identifier for the current simulation case
    """
    issues = []
    tol = 0.1
    # Check IHX
    ihx_diff = df_streams.loc[12, 'T [°C]'] - df_streams.loc[16, 'T [°C]']
    if ihx_diff < min_temps['ihx'] - tol:
        print(f"Warning: Upper terminal temperature difference in IHX (for the case {case}) too low: {ihx_diff:.2f} K (min: {min_temps['ihx']} K)")

    # Check COND-ECO
    cond_eco_diff = df_streams.loc[12, 'T [°C]'] - df_streams.loc[21, 'T [°C]']
    if cond_eco_diff < min_temps['cond_eco'] - tol:
        print(f"Warning: Lower terminal temperature difference in COND (for the case {case}) too low: {cond_eco_diff:.2f} K (min: {min_temps['cond_eco']} K)")

    # Check COND-SH
    cond_sh_diff = df_streams.loc[18, 'T [°C]'] - df_streams.loc[29, 'T [°C]']
    if cond_sh_diff < min_temps['cond_sh'] - tol:
        print(f"Warning: Temperature difference at boiling pinch in COND (for the case {case}) is too low: {cond_sh_diff:.2f} K (min: {min_temps['cond_sh']} K)")

    if issues:
        print("\nWarning: Temperature difference validation found issues:")
        for issue in issues:
            print(issue)


def simulate_hp(target_p12, config, label, config_paths, adex=False, print_results=False, qt_diagrams=False):
    """
    Simulate the performance of a heat pump system under specified conditions.

    Parameters
    ----------
    target_p12 : float
        Target pressure at state point 12 in Pa.
    config : dict
        Configuration dictionary specifying component states ('real', 'ideal', or 'unavoid').
        Must include keys: 'comp', 'cond', 'ihx', 'val', 'eva'.
    label : str
        Identifier for the simulation case, used in outputs and plots.
    config_paths : dict
        Dictionary containing file paths for configuration and outputs.
        Must include keys: 'base', 'unavoid', 'outputs'.
    adex : bool, optional
        If True, uses advanced exergetic analysis parameters, by default False.
    print_results : bool, optional
        If True, prints simulation results, by default False.
    qt_diagrams : bool, optional
        If True, generates Q-T diagrams for heat exchangers, by default False.

    Returns
    -------
    df_streams : pandas.DataFrame
        DataFrame containing thermodynamic properties for each state point.
        Includes mass flow rates, temperatures, enthalpies, pressures, and entropies.
    t_pinch_cond_sh : float
        Pinch temperature difference in the superheater section of the condenser.
    cop : float
        Coefficient of performance of the heat pump system.
    allowed_min_temps : dict
        Dictionary of minimum allowed temperature differences for each heat exchanger.

    Notes
    -----
    The function solves a system of nonlinear equations to determine the operating
    conditions of the heat pump cycle. It handles different component configurations
    (real, ideal, unavoidable) and includes pressure drops and heat exchanger
    effectiveness calculations.
    """

    # Load configurations from JSON file
    with open(config_paths['unavoid'], 'r') as file:
        hp_config_unavoid = json.load(file)
    with open(config_paths['base'], 'r') as file:
        hp_config = json.load(file)
    try:
        # Use epsilon from base case
        epsilon_path = os.path.join(config_paths['outputs'], 'comps', 'hp_comps_all_real.csv')
        epsilon = pd.read_csv(epsilon_path, index_col=0)['epsilon']
    except FileNotFoundError:
        print('File not found. Please run base case model first!')
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
        # elif adex and config['ihx'] == 'real':  # adex real ihx: epsilon from base case
            # t16_set = eps_real_ihx_energy_bal(epsilon['ihx'], variables[3], target_p12, variables[8], variables[11], wf,
                                        # variables[9], variables[15], variables[0], variables[10], wf)
        else:  # real or unavoid ihx: given ttd_u
            t16_set = ttd_temperature_func(-ttd_u_ihx, variables[3], target_p12, wf, variables[0], variables[10], wf)
        #   2
        if adex and config['cond'] == 'real':  # adex real cond: epsilon from base case
            t12_set = eps_real_he_func(epsilon['cond'], variables[1], variables[2], variables[3], target_p12, variables[4], wf,
                                       variables[6], p21, variables[7], variables[14], m21, fluid_tes)
        else:  # ideal, real, unavoid cond: given ttd_l (different from cond, because temperature is known in this case)
            t12_set = ttd_temperature_func(-ttd_l_cond, variables[3], target_p12, wf, variables[6], p21, fluid_tes)
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
            t13_calc_ihx = ihx_energy_bal(variables[3], variables[8], variables[9], variables[0])
        #   9
        p16_set = pr_func(pr_ihx_cold, variables[15], variables[10])
        #   10
        p13_set = pr_func(pr_ihx_hot, target_p12, variables[11])
        #   11
        if adex and config['val'] == 'ideal':  # ideal val: entropy balance equation (isentropic expansion)
            t14_calc_valve = ideal_valve_entropy_func(variables[8], variables[11], variables[12], variables[13], wf)
        elif config['val'] == 'unavoid':
            t14_calc_valve = eta_s_compressor_func(0.95, variables[8], variables[11], variables[12], variables[13], wf)
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
        # elif adex and config['ihx'] == 'real':  # adex real ihx: epsilon from base case
            # t16_set_j = eps_real_ihx_energy_bal_deriv(epsilon['ihx'], variables[3], target_p12, variables[8], variables[11], wf,
                                           # variables[9], variables[15], variables[0], variables[10], wf)
        else:  # real or unavoid ihx: given ttd_u
            t16_set_j = ttd_temperature_deriv(-ttd_u_ihx, variables[3], target_p12, wf, variables[0], variables[10], wf)
        #   2
        if adex and config['cond'] == 'real':  # adex real cond: epsilon from base case
            t12_set_j = eps_real_he_deriv(epsilon['cond'], variables[1], variables[2], variables[3], target_p12, variables[4], wf,
                                          variables[6], p21, variables[7], variables[14], m21, fluid_tes)
        else:  # ideal, real, unavoid cond: given ttd_l (different from cond, because temperature is known in this case)
            t12_set_j = ttd_temperature_deriv(-ttd_l_cond, variables[3], target_p12, wf, variables[6], p21, fluid_tes)
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
            t13_calc_ihx_j = ihx_energy_bal_deriv(variables[3], variables[8], variables[9], variables[0])
        #   9
        p16_set_j = pr_deriv(pr_ihx_cold, variables[15], variables[10])
        #   10
        p13_set_j = pr_deriv(pr_ihx_hot, target_p12, variables[11])
        #   11
        if adex and config['val'] == 'ideal':  # ideal val: entropy balance equation (isentropic expansion)
            t14_calc_valve_j = ideal_valve_entropy_deriv(variables[8], variables[11], variables[12], variables[13], wf)
        elif config['val'] == 'unavoid':
            t14_calc_valve_j = eta_s_compressor_deriv(0.95, variables[8], variables[11], variables[12], variables[13], wf)
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
        #elif adex and config['ihx'] == 'real':
            #jacobian[1, 3] = t16_set_j['h_hot_in']  # derivative of t16_set with respect to h16
            #jacobian[1, 8] = t16_set_j['h_hot_out']  # derivative of t16_set with respect to p16
            #jacobian[1, 11] = t16_set_j['p_hot_out']  # derivative of t16_set with respect to h16
            #jacobian[1, 9] = t16_set_j['h_cold_in']  # derivative of t16_set with respect to p16
            #jacobian[1, 15] = t16_set_j['p_cold_in']  # derivative of t16_set with respect to h16
            #jacobian[1, 0] = t16_set_j['h_cold_out']  # derivative of t16_set with respect to p16
            #jacobian[1, 10] = t16_set_j['p_cold_out']  # derivative of t16_set with respect to h16
        else:
            jacobian[1, 3] = t16_set_j['h_copy']  # derivative of t16_set with respect to h16
            jacobian[1, 0] = t16_set_j['h_paste']  # derivative of t16_set with respect to h16
            jacobian[1, 10] = t16_set_j['p_paste']  # derivative of t16_set with respect to p16
        if adex and config['cond'] == 'real':
            jacobian[2, 1] = t12_set_j['h_hot_in']  # derivative of m11_calc_cond with respect to h11
            jacobian[2, 2] = t12_set_j['p_hot_in']  # derivative of m11_calc_cond with respect to h12
            jacobian[2, 3] = t12_set_j['h_hot_out']  # derivative of m11_calc_cond with respect to m11
            jacobian[2, 4] = t12_set_j['m_hot']  # derivative of m11_calc_cond with respect to h21
            jacobian[2, 6] = t12_set_j['h_cold_in']  # derivative of m11_calc_cond with respect to h22
            jacobian[2, 7] = t12_set_j['h_cold_out']  # derivative of m11_calc_cond with respect to h22
            jacobian[2, 14] = t12_set_j['p_cold_out']  # derivative of m11_calc_cond with respect to h22
        else:
            jacobian[2, 3] = t12_set_j['h_copy']  # derivative of t12_set with respect to h12 (hot stream)
            jacobian[2, 6] = t12_set_j['h_paste']  # derivative of t12_set with respect to h21 (cold stream)
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

    # Define minimum temperature differences based on configuration for later checks
    allowed_min_temps = {
        'ihx': ttd_u_ihx,
        'cond_eco': ttd_l_cond,
        'cond_eva': ttd_l_cond,
        'cond_sh': ttd_l_cond
    }

    # check_minimum_temperature_differences_hp(df_streams, allowed_min_temps, case=f'_{label}')

    # Create diagram paths
    cond_path = os.path.join(config_paths['outputs'], 'diagrams', f'hp_qt_COND_{label}.png')
    cond_sh_path = os.path.join(config_paths['outputs'], 'diagrams', f'hp_qt_COND_SH_{label}.png')
    cond_eva_path = os.path.join(config_paths['outputs'], 'diagrams', f'hp_qt_COND_EVA_{label}.png')
    cond_eco_path = os.path.join(config_paths['outputs'], 'diagrams', f'hp_qt_COND_ECO_{label}.png')
    eva_path = os.path.join(config_paths['outputs'], 'diagrams', f'hp_qt_EVA_{label}.png')
    ihx_path = os.path.join(config_paths['outputs'], 'diagrams', f'hp_qt_IHX_{label}.png')

    if qt_diagrams:
        qt_diagram(df_streams, 'COND', 11, 12, 21, 22, ttd_l_cond, 'HP', case=f'{label}', show=False, path=cond_path, step_number=50, tol=1)
        # qt_diagram(df_streams, 'COND-EVA', 18, 19, 28, 29, ttd_l_eva, 'HP', case=f'{label}', show=False, path=cond_eva_path, step_number=50, tol=1)
        # qt_diagram(df_streams, 'COND-SH', 11, 18, 29, 22, ttd_l_cond, 'HP', case=f'{label}', show=False, path=cond_sh_path, step_number=50, tol=1)
        # qt_diagram(df_streams, 'COND-ECO', 19, 12, 21, 28, ttd_l_cond, 'HP', case=f'{label}', show=False, path=cond_eco_path, step_number=50, tol=1)
        qt_diagram(df_streams, 'IHX', 12, 13, 15, 16, ttd_u_ihx, 'HP', case=f'{label}', show=False, path=ihx_path, step_number=50, tol=1)

    t_pinch_cond_sh = t18-t29

    cop = heat_cond/(power_comp-power_ihx-power_val-power_cond)

    for col in df_streams.columns[:5]:
        df_streams[col] = pd.to_numeric(df_streams[col], errors='coerce')

    return df_streams, t_pinch_cond_sh, cop, allowed_min_temps


def exergy_analysis_hp(t0, p0, df):
    """
    Conduct exergy analysis on a heat pump system.

    Parameters
    ----------
    t0 : float
        Ambient temperature in Kelvin.
    p0 : float
        Ambient pressure in Pa.
    df : pandas.DataFrame
        Dataframe containing thermodynamic properties for each state point.
        Must include columns: 'm [kg/s]', 'T [°C]', 'h [kJ/kg]', 'p [bar]',
        's [J/kgK]', 'fluid'.

    Returns
    -------
    df_comps : pandas.DataFrame
        DataFrame containing exergy analysis results for each component.
        Includes columns: 'EF [kW]' (exergy flow), 'EP [kW]' (exergy product),
        'ED [kW]' (exergy destruction), 'epsilon' (exergy efficiency),
        'P [kW]' (power), 'Q [kW]' (heat), 'unavoid ratio'.

    Notes
    -----
    Calculates physical exergy, exergy destruction, and exergy efficiency
    for each component (compressor, condenser, IHX, valve, evaporator).
    Special handling for valve exergy calculations based on operating conditions.
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


def config_hp(comp='real', cond='real', ihx='real', val='real', eva='real'):
    """
    Generate configuration dictionary and label for heat pump system.

    Parameters
    ----------
    comp : str, optional
        Compressor state, by default 'real'.
    cond : str, optional
        Condenser state, by default 'real'.
    ihx : str, optional
        Internal heat exchanger state, by default 'real'.
    val : str, optional
        Valve state, by default 'real'.
    eva : str, optional
        Evaporator state, by default 'real'.

    Returns
    -------
    config : dict
        Dictionary specifying state ('real', 'ideal', or 'unavoid') for each component.
    label : str
        Identifier string for the configuration.

    Notes
    -----
    Label construction rules:
    1. 'all_real' if all components are real
    2. 'all_ideal' if all components are ideal
    3. 'all_unavoid' if all components are unavoidable
    4. Component names with states for mixed configurations
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


def find_opt_p12(p12_opt_start, min_t_diff_cond_start, config, label, config_paths, print_results, adex=False, output_buffer=None):
    """
    Find optimal pressure at state point 12 to achieve target temperature difference.

    Parameters
    ----------
    p12_opt_start : float
        Initial guess for pressure at state point 12 in Pa.
    min_t_diff_cond_start : float
        Initial minimum temperature difference in condenser.
    config : dict
        Component configuration dictionary.
    label : str
        Simulation case identifier.
    config_paths : dict
        Dictionary of configuration file paths.
    print_results : bool
        If True, print optimization progress.
    adex : bool, optional
        If True, use advanced exergetic analysis, by default False.
    output_buffer : list, optional
        Buffer for storing optimization output messages.

    Returns
    -------
    float
        Optimized pressure at state point 12 in Pa.

    Notes
    -----
    Uses iterative optimization to minimize condenser temperature difference
    while maintaining minimum IHX temperature difference constraint.
    """

    buffer = io.StringIO()

    with open(config_paths['unavoid'], 'r') as file:
        hp_config_unavoid = json.load(file)
    with open(config_paths['base'], 'r') as file:
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
    learning_rate = 2e4  # relative to p12
    diff = min_t_diff_cond_start - target_min_td_cond
    step = 0

    while abs(diff) > tolerance:
        # Adjust p12 based on the difference
        adjustment = (target_min_td_cond - min_t_diff_cond) * learning_rate
        # adjustment is the smaller, the smaller the difference target_min_td_cond - min_td_cond
        p12_opt += adjustment

        [df, min_t_diff_cond, cop, allowed_min_temps] = simulate_hp(p12_opt, label=label, config=config, config_paths=config_paths,
                                                                    adex=adex, print_results=False, qt_diagrams=False)

        # check_minimum_temperature_differences_hp(df, allowed_min_temps, case=f'_{label}_{step}')

        diff = abs(min_t_diff_cond - target_min_td_cond)

        step += 1
        if print_results:  # Only add to buffer if print_results is True
            buffer.write(
                f'Optimization in progress for {label}: step = {step}, diff = {round(diff, 6)}, p12 = {round(p12_opt * 1e-5, 4)} bar, COP = {round(cop, 4)}.\n')

    if print_results:  # Only add to buffer if print_results is True
        buffer.write(f'Optimization completed successfully in {step} steps!\n')
        buffer.write(f'Optimal p12: {round(p12_opt*1e-5, 4)}, bar.\n')

    if output_buffer is not None and print_results:  # Only append to output_buffer if print_results is True
        output_buffer.append(buffer.getvalue())

    return p12_opt


def perform_adex_hp(p12_start, config, label, df_ed, config_paths, adex=False, save=False, print_results=False, calc_epsilon=False, output_buffer=None):
    """
    Perform advanced exergy analysis for heat pump system.

    Parameters
    ----------
    p12_start : float
        Initial pressure at state point 12 in Pa.
    config : dict
        Component configuration dictionary.
    label : str
        Simulation case identifier.
    df_ed : pandas.DataFrame
        DataFrame to store exergy destruction results.
    config_paths : dict
        Dictionary of configuration file paths.
    adex : bool, optional
        If True, use advanced exergetic analysis, by default False.
    save : bool, optional
        If True, save results to files, by default False.
    print_results : bool, optional
        If True, print analysis results, by default False.
    calc_epsilon : bool, optional
        If True, calculate exergy efficiencies, by default False.
    output_buffer : list, optional
        Buffer for storing analysis output messages.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with exergy destruction results.
    """

    # Run first simulation and get pinch point in condenser (target_diff)
    [df_unopt, target_diff, _, _] = simulate_hp(p12_start, config=config, label=label, config_paths=config_paths, adex=adex,
                                      print_results=print_results, qt_diagrams=False)

    # Search for optimal p12 according to minimum pinch point in cond
    p12_opt = find_opt_p12(p12_start, target_diff, config=config, label=label,  config_paths=config_paths,
                           print_results=print_results, adex=adex, output_buffer=output_buffer)

    # Run simulation with optimal p12 and minimum pinch point in cond
    [df_opt, _, cop, allowed_min_temps] = simulate_hp(p12_opt, config=config, label=f'optimal_{label}', config_paths=config_paths,
                                                    adex=adex, print_results=print_results, qt_diagrams=True)

    # Store the optimized COP
    store_cop_in_csv(cop, f"optimal_{label}", config_paths)

    # Check for minimum temperature differences after optimal pressure simulation
    try:
        check_minimum_temperature_differences_hp(df_opt, allowed_min_temps, case=f'optimal_{label}')
    except ValueError as e:
        print(f"Validation Error after optimization: {e}")

    # Run conventional exergy analysis to get exergetic efficiencies and exergy destruction rates
    with open(config_paths['base'], 'r') as file:
        hp_config = json.load(file)

    t0 = hp_config['ambient']['t0']  # K
    p0 = hp_config['ambient']['p0']  # Pa
    df_components = exergy_analysis_hp(t0, p0, df_opt)

    for component in df_components.index:
        df_ed.loc[(label, component), 'ED [kW]'] = df_components.loc[component, 'ED [kW]']
    if calc_epsilon:
        for col in df_components.columns[:5]:
            df_components[col] = pd.to_numeric(df_components[col], errors='coerce')

        output_path = os.path.join(config_paths['outputs'], 'comps', f'hp_comps_{label}.csv')
        if label == "all_real":
            df_components.to_csv(output_path)
        else:
            df_components.round(4).to_csv(output_path)

    if save:
        output_path = os.path.join(config_paths['outputs'], 'streams', f'hp_streams_{label}.csv')
        round(df_opt, 5).to_csv(output_path)

    return df_ed


def run_adex_hp(high_pressure_level, config, label, df_ed, config_paths, adex, save, print_results, calc_epsilon, output_buffer=None):
    """
    Run advanced exergy analysis with specified configuration.

    Parameters
    ----------
    high_pressure_level : float
        Initial high-side pressure level in Pa.
    config : dict
        Component configuration dictionary.
    label : str
        Simulation case identifier.
    df_ed : pandas.DataFrame
        DataFrame to store exergy destruction results.
    config_paths : dict
        Dictionary of configuration file paths.
    adex : bool
        If True, use advanced exergetic analysis.
    save : bool
        If True, save results to files.
    print_results : bool
        If True, print analysis results.
    calc_epsilon : bool
        If True, calculate exergy efficiencies.
    output_buffer : list, optional
        Buffer for storing analysis output messages.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with exergy destruction results.
    """

    df_ed = perform_adex_hp(high_pressure_level, config, label, df_ed, config_paths, adex, save, print_results, calc_epsilon, output_buffer)
    return df_ed


def serial_hp(config_paths, high_pressure_level, print_results=False):
    """
    Run serial execution of advanced exergy analysis.

    Parameters
    ----------
    config_paths : dict
        Dictionary of configuration file paths.
    high_pressure_level : float
        Initial high-side pressure level in Pa.
    print_results : bool, optional
        If True, print analysis results, by default False.

    Notes
    -----
    Performs sequential analysis for:
    1. Base case (all real components)
    2. Ideal case (all ideal components)
    3. Single real component cases
    4. Component pair cases

    Should be used mainly for debugging purposes.
    For production use, prefer multiprocess_hp().
    """

    columns = ['ED']
    multi_index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['Label', 'Component'])
    df_ed = pd.DataFrame(columns=columns, index=multi_index)

    # BASE CASE
    [config_base, label_base] = config_hp('comp', 'cond', 'ihx', 'val', 'eva')
    perform_adex_hp(high_pressure_level, config_base, label_base, df_ed, config_paths, adex=False, save=True, print_results=print_results, calc_epsilon=True)

    # IDEAL CASE
    [config_ideal, label_ideal] = config_hp()
    perform_adex_hp(high_pressure_level, config_ideal, label_ideal, df_ed, config_paths, adex=False, save=True, print_results=print_results, calc_epsilon=True)

    # ADVANCED EXERGY ANALYSIS -- single components
    components = ['comp', 'cond', 'ihx', 'val', 'eva']
    for component in components:
        config_i, label_i = config_hp(component)
        perform_adex_hp(high_pressure_level, config_i, label_i, df_ed, config_paths, adex=True, save=True, print_results=print_results, calc_epsilon=True)

    # ADVANCED EXERGY ANALYSIS -- pair of components
    component_pairs = list(itertools.combinations(components, 2))
    for pair in component_pairs:
        config_i, label_i = config_hp(*pair)
        perform_adex_hp(high_pressure_level, config_i, label_i, df_ed, config_paths, adex=True, save=True, print_results=print_results, calc_epsilon=True)


def multiprocess_hp(config_paths, high_pressure_level, print_results=False):
    """
    Run parallel execution of advanced exergy analysis.

    Parameters
    ----------
    config_paths : dict
        Dictionary of configuration file paths.
    high_pressure_level : float
        Initial high-side pressure level in Pa.
    print_results : bool, optional
        If True, print analysis results, by default False.

    Notes
    -----
    Performs parallel analysis using multiprocessing for:
    1. Base case
    2. Ideal case
    3. Single real component cases
    4. Component pair cases
    5. Unavoidable cases

    Results are saved to CSV files and include:
    - Stream properties
    - Component performance
    - Exergy destruction analysis
    - Advanced exergy analysis metrics
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
    [config_base, label_base] = config_hp()
    perform_adex_hp(high_pressure_level, config_base, label_base, df_ed_base, config_paths, adex=False, save=True,
                    print_results=print_results, calc_epsilon=True, output_buffer=output_buffer)

    tasks = []

    # Create a pool of workers
    with multiprocessing.Pool() as pool:

        # IDEAL CASE
        config_ideal, label_ideal = config_hp(comp='ideal', cond='ideal', ihx='ideal', val='ideal', eva='ideal')
        tasks.append((high_pressure_level, config_ideal, label_ideal, df_ed_i, config_paths, True, True, print_results, True, output_buffer))

        components = ['comp', 'cond', 'ihx', 'val', 'eva']
        component_pairs = list(itertools.combinations(components, 2))

        # EN ED -- single components
        for component in components:
            config_kwargs = {comp: 'ideal' for comp in components}  # Start with all components as 'ideal'
            config_kwargs[component] = 'real'  # Set the current component to 'real'
            config_i, label_i = config_hp(**config_kwargs)
            tasks.append((high_pressure_level, config_i, label_i, df_ed_i, config_paths, True, True, print_results, True, output_buffer))

        # EN ED -- pair of components
        for pair in component_pairs:
            config_kwargs = {comp: 'ideal' for comp in components}  # Start with all components as 'ideal'
            for comp in pair:
                config_kwargs[comp] = 'real'  # Set each component in the pair to 'real'
            config_i, label_i = config_hp(**config_kwargs)
            tasks.append((high_pressure_level, config_i, label_i, df_ed_i, config_paths, True, True, print_results, True, output_buffer))

        # UN ED -- single components
        for component in components:
            config_kwargs = {comp: 'real' for comp in components}  # Start with all components as 'real'
            config_kwargs[component] = 'unavoid'  # Set the current component to 'unavoid'
            config_i, label_i = config_hp(**config_kwargs)
            tasks.append((high_pressure_level, config_i, label_i, df_ed_i, config_paths, False, True, print_results, True, output_buffer))

        # Map tasks to the pool
        results = pool.starmap(run_adex_hp, tasks)

    # Print the stored outputs after all tasks are done
    if print_results:  # Only print buffer contents if print_results is True
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
    df_ed.round(6).to_csv(os.path.join(config_paths['outputs'], 'hp_adex_ed.csv'))

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
    epsilon = pd.read_csv(os.path.join(config_paths['outputs'], 'comps/hp_comps_all_real.csv'), index_col=0)['epsilon']

    epsilon['eva'] = float('nan')

    for k in components:
        # EN / EX / binary interaction / MX
        df_k_real_streams = pd.read_csv(os.path.join(config_paths['outputs'], 'streams/hp_streams_all_real.csv'), index_col=0)
        df_k_endo = pd.read_csv(os.path.join(config_paths['outputs'], f'streams/hp_streams_{k}.csv'), index_col=0)
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
                df_k_l = pd.read_csv(os.path.join(config_paths['outputs'], f'streams/hp_streams_{k_l}.csv'), index_col=0)
                df_adex_analysis.loc[(k, l), 'm EN [kg/s]'] = df_k_l.loc[11, 'm [kg/s]']
                df_adex_analysis.loc[(k, l), 'm EX [kg/s]'] = df_k_real_streams.loc[11, 'm [kg/s]'] - df_k_l.loc[11, 'm [kg/s]']
                df_adex_analysis.loc[(k, l), 'ED kl [kW]'] = df_ed.loc[(k_l, k), 'ED [kW]']
                df_adex_analysis.loc[(k, l), 'ED kl [kW]'] = df_ed.loc[(k_l, k), 'ED [kW]']
                df_adex_analysis.loc[(k, l), 'ED EX kl [kW]'] = df_adex_analysis.loc[(k, l), 'ED kl [kW]'] - df_adex_analysis.loc[(k, ''), 'ED EN [kW]']
                sum_ed_ex_kl += df_adex_analysis.loc[(k, l), 'ED EX kl [kW]']
        df_adex_analysis.loc[(k, ''), 'ED MX [kW]'] = df_adex_analysis.loc[(k, ''), 'ED EX [kW]'] - sum_ed_ex_kl

        # UN / AV/ binary interaction
        df_k_unavoid_comps = pd.read_csv(os.path.join(config_paths['outputs'], f'comps/hp_comps_{k}_unavoid.csv'), index_col=0)
        df_k_endo_comps = pd.read_csv(os.path.join(config_paths['outputs'], f'comps/hp_comps_{k}.csv'), index_col=0)
        df_k_endo_streams = pd.read_csv(os.path.join(config_paths['outputs'], f'streams/hp_streams_{k}.csv'), index_col=0)
        df_k_real_comps = pd.read_csv(os.path.join(config_paths['outputs'], 'comps/hp_comps_all_real.csv'), index_col=0)
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
                                                                    * (df_adex_analysis.loc[(k, l), 'ED EX kl [kW]'] /
                                                                       df_adex_analysis.loc[(k, ''), 'ED EX [kW]']))

    # AV SUM
    for k in components:
        sum_ed_ex_av_kl = 0
        for l in components:
            if k != l:
                sum_ed_ex_av_kl += df_adex_analysis.loc[(k, l), 'ED AV EX kl [kW]']
        df_adex_analysis.loc[(k, ''), 'ED AV SUM [kW]'] = (
                df_adex_analysis.loc[(k, ''), 'ED AV EN [kW]']
                + sum_ed_ex_av_kl
        )

        df_adex_analysis.loc[(k, ''), 'ED AV SUM [kW]'] = df_adex_analysis.loc[(k, ''), 'ED AV EN [kW]'] + sum_ed_ex_av_kl

    df_adex_analysis.round(5).to_csv(os.path.join(config_paths['outputs'], 'hp_adex_analysis.csv'))

    print("\nBase Case Results:")
    df_base = pd.read_csv(os.path.join(config_paths['outputs'], 'streams/hp_streams_all_real.csv'), index_col=0)
    print(tabulate_func(df_base, headers='keys', tablefmt='psql', floatfmt=".2f"))

    df_exergy = pd.read_csv(os.path.join(config_paths['outputs'], 'comps/hp_comps_all_real.csv'), index_col=0)
    columns_to_print = ['EF [kW]', 'EP [kW]', 'ED [kW]', 'epsilon', 'P [kW]', 'Q [kW]']
    print(tabulate_func(df_exergy[columns_to_print], headers='keys', tablefmt='psql', floatfmt=".2f"))

    print("\nIdeal Case Results:")
    df_ideal = pd.read_csv(os.path.join(config_paths['outputs'], 'streams/hp_streams_all_ideal.csv'), index_col=0)
    print(tabulate_func(df_ideal, headers='keys', tablefmt='psql', floatfmt=".2f"))

    df_exergy = pd.read_csv(os.path.join(config_paths['outputs'], 'comps/hp_comps_all_ideal.csv'), index_col=0)
    columns_to_print = ['EF [kW]', 'EP [kW]', 'ED [kW]', 'epsilon', 'P [kW]', 'Q [kW]']
    print(tabulate_func(df_exergy[columns_to_print], headers='keys', tablefmt='psql', floatfmt=".2f"))

    print("\nAdvanced Exergy Analysis Summary:")
    total_results = df_adex_analysis.xs('', level='l', drop_level=False)
    columns_to_print = ['ED [kW]', 'ED EN [kW]', 'ED EX [kW]', 'ED MX [kW]',
                        'ED UN [kW]', 'ED AV [kW]', 'ED AV SUM [kW]']
    # Remove the multi-index tuple formatting
    total_results.index = [idx[0] for idx in total_results.index]
    print(tabulate_func(total_results[columns_to_print], headers='keys', tablefmt='psql', floatfmt=".2f"))

    end = time.perf_counter()
    print(f'Elapsed time: {round(end - start, 3)} seconds.')