"""
Author: Sergio Tomasinelli
Date: December 2023
Description: Modeling, simulation and advanced exergy analysis of a Carnot battery
"""

import json

import pandas as pd
from func_var import run_hp, run_orc, perf_adex_hp, perf_adex_orc, sens_an_p32_TESPy, sens_an_ttd_u_eva_ORC_TESPy, sens_an_ttd_l_cond_ORC_TESPy
from tespy.networks import Network
from tespy.tools import logger
import numpy as np
import logging
logger.define_logging(screen_level=logging.WARNING, file_level=logging.INFO)


with open('inputs/config.json') as file:
    config = json.load(file)
with open('inputs/hp_pars.json') as file:
    hp_pars_base = json.load(file)
with open('inputs/orc_pars.json') as file:
    orc_pars_base = json.load(file)


# --- CONTROLLERS ------------------------------------------------------------------------------------------------------
controller = {
    'run_HP': False,  # Control flag for running the Heat Pump (HP) model.
    'run_ORC': True,  # Control flag for running the Organic Rankine Cycle (ORC) model.

    'sens_HP': False,  # Control flag for performing sensitivity analysis on the HP model.
    'sens_ORC': False,  # Control flag for performing sensitivity analysis on the ORC model.

    'adex_HP': False,  # Control flag for performing advanced exergy analysis on the HP model.
    'adex_ORC': True,  # Control flag for performing advanced exergy analysis on the ORC model.

    'check_temp_diff': True  # Control flag for checking temperature differences in the heat exchangers during advanced exergy analysis.
}


# --- RUN MODELS -------------------------------------------------------------------------------------------------------
if controller['run_HP']:
    hp_base = run_hp(config, hp_pars_base, 'base', Tdeltah=True, logph=False, show=False)
if controller['run_ORC']:
    orc_base = run_orc(config, orc_pars_base, 'base', Tdeltah=True, logph=False, show=False)


# --- PERFORM SENSITIVITY ANALYSIS -------------------------------------------------------------------------------------
if controller['run_HP'] and controller['sens_HP']:
    p_32_opt = sens_an_p32_TESPy(hp_base, 11, 14, 0.25, 5)
    hp_base.conns['c32'].set_attr(p=p_32_opt)
    hp_base.network.solve('design')
    hp_base.network.results['Connection'].round(5).to_csv(f'outputs/hp_base_results.csv')  # rewrite on previous
    print(f'Optimal value for p32 of HP = {p_32_opt} bar.')

if controller['run_ORC'] and controller['sens_ORC']:
    ttd_l_cond_opt = sens_an_ttd_l_cond_ORC_TESPy(orc_base, 0, 10, 0.5, 5)
    orc_base.comps['cond'].set_attr(ttd_l=ttd_l_cond_opt)
    ttd_u_eva_opt = sens_an_ttd_u_eva_ORC_TESPy(orc_base, 0, 15, 0.5, 5)
    orc_base.comps['eva'].set_attr(ttd_u=ttd_u_eva_opt)
    orc_base.network.solve('design')
    orc_base.network.results['Connection'].round(5).to_csv(f'outputs/orc_base_results.csv')  # rewrite on previous
    print(f'Optimal value for ttd_l_cond of ORC = {ttd_l_cond_opt} K.')
    print(f'Optimal value for ttd_u_eva of ORC = {ttd_u_eva_opt} K.')


# --- PERFORM CONVENTIONAL AND ADVANCED EXERGY ANALYSIS ----------------------------------------------------------------
if controller['run_HP'] and controller['adex_HP']:
    hp_base.perform_exergy_analysis(path='outputs/exan/hp_base_exan_')
    print("Conventional exergy analysis of the HP base case: epsilon_real =", round(hp_base.epsilon, 3))
    perf_adex_hp(hp_base, p32=13, components=['COMP', 'COND', 'IHX', 'VAL', 'EVA'], check_temp_diff=controller['check_temp_diff'])

if controller['run_ORC'] and controller['adex_ORC']:
    orc_base.perform_exergy_analysis(path='outputs/exan/orc_base_exan_')
    print("Conventional exergy analysis of the ORC base case: epsilon_real =", round(orc_base.epsilon, 3))
    perf_adex_orc(orc_base, ttd_u_eva=8.5, ttd_l_cond=6.5, components=['EVA', 'EXP', 'IHX', 'COND', 'PUMP'], check_temp_diff=controller['check_temp_diff'])



