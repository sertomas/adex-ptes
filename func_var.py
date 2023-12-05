import pandas as pd
import numpy as np
import json
from itertools import product
from itertools import combinations
from CoolProp.CoolProp import PropsSI as PSI
from models.hp.HeatPumpIhx import HeatPumpIhx
from models.orc.OrcIhx import OrcIhx
from func_fix import qt_diagram, calc_vapor_content, pump, turbine, valve, heat_exchanger, turbo_mach_bal, valve_bal, he_bal

# --- 1. OPEN CONFIG AND PARS SETTINGS ---------------------------------------------------------------------------------

with open('inputs/config.json') as file:
    config = json.load(file)
with open('inputs/hp_pars.json') as file:
    hp_pars_base = json.load(file)
with open('inputs/orc_pars.json') as file:
    orc_pars_base = json.load(file)


# --- 2. AMBIENT CONDITIONS AND FLUIDS ---------------------------------------------------------------------------------

T_0 = config['ambient']['T'] + 273.15  # [K]
p_0 = config['ambient']['p']  # [bar]

wf_hp = hp_pars_base['fluids']['working_fluid']  # name of working fluid of the HP
wf_orc = orc_pars_base['fluids']['working_fluid']  # name of working fluid of the ORC
fluid_TES = hp_pars_base['fluids']['fluid_TES']  # name of working fluid of the TES
fluid_ambient = hp_pars_base['fluids']['fluid_ambient']  # name of fluid of the ambient stream

h_0_fluids = {
    wf_hp: PSI('H', 'T', T_0, 'P', p_0 * 1e5, wf_hp) * 1e-3,
    wf_orc: PSI('H', 'T', T_0, 'P', p_0 * 1e5, wf_orc) * 1e-3,
    fluid_TES: PSI('H', 'T', T_0, 'P', p_0 * 1e5, fluid_TES) * 1e-3,
    fluid_ambient: PSI('H', 'T', T_0, 'P', p_0 * 1e5, fluid_ambient) * 1e-3,
}  # enthalpy at ambient conditions

s_0_fluids = {
    wf_hp: PSI('S', 'T', T_0, 'P', p_0 * 1e5, wf_hp),
    wf_orc: PSI('S', 'T', T_0, 'P', p_0 * 1e5, wf_orc),
    fluid_TES: PSI('S', 'T', T_0, 'P', p_0 * 1e5, fluid_TES),
    fluid_ambient: PSI('S', 'T', T_0, 'P', p_0 * 1e5, fluid_ambient),
}  # entropy at ambient conditions


# --- 3. FUNCTIONS FOR THE MODELS --------------------------------------------------------------------------------------

def run_hp(config, hp_pars, case, Tdeltah=False, logph=False, show=False):
    """
    Run a base case simulation of a heat pump with an internal heat exchanger and optionally generate T-delta-h and log(p)-h diagrams.

    Args:
        config (dict): Configuration parameters for the HeatPumpIhx instance.
        hp_pars (dict): Heat pump parameters for the HeatPumpIhx instance.
        case (str): Identifier for the simulation case.
        Tdeltah (bool, optional): Whether to generate T-delta-h diagrams. Default is False.
        logph (bool, optional): Whether to generate log(p)-h diagrams. Default is False.
        show (bool, optional): Whether to display generated diagrams. Default is False.

    Returns:
        HeatPumpIhx: An instance of the HeatPumpIhx class after running the simulation.
    """

    hp = HeatPumpIhx(config, hp_pars)
    hp.run_model()
    hp.network.results['Connection'].round(5).to_csv(f'outputs/hp_{case}_results.csv')
    print('Q_out = ', round(hp.network.get_comp('Condenser').get_attr('Q').val * 1e-3, 1), 'kW')
    print('Q_in = ', round(hp.network.get_comp('Evaporator').get_attr('Q').val * 1e-3, 1), 'kW')
    print('P_in = ', round(hp.network.get_comp('Compressor').get_attr('P').val * 1e-3, 1), 'kW')
    if Tdeltah:
        hp.qt_diagram(11, 12, 34, 35, 5, case, plot=show, path=f'outputs/diagrams/qt_hp_{case}_eva.png')  # EVA
        hp.qt_diagram(31, 32, 21, 22, 5, case, plot=show, path=f'outputs/diagrams/qt_hp_{case}_cond.png')  # COND
        hp.qt_diagram(32, 33, 35, 36, 5, case, plot=show, path=f'outputs/diagrams/qt_hp_{case}_ihx.png')  # IHX
    if logph:
        hp.plot_logph(path=f'outputs/diagrams/{case}.png', return_diagram=show)
    return hp


def run_orc(config, orc_pars, case, Tdeltah=False, logph=False, show=False):
    """
    Run a base case simulation of an ORC (Organic Rankine Cycle) with an internal heat exchanger and optionally
    generate T-delta-h and log(p)-h diagrams.

    Args:
        config (dict): Configuration parameters for the OrcIhx instance.
        orc_pars (dict): ORC parameters for the OrcIhx instance.
        case (str): Identifier for the simulation case.
        Tdeltah (bool, optional): Whether to generate T-delta-h diagrams. Default is False.
        logph (bool, optional): Whether to generate log(p)-h diagrams. Default is False.
        show (bool, optional): Whether to display generated diagrams. Default is False.

    Returns:
        OrcIhx: An instance of the OrcIhx class after running the simulation.
    """

    orc = OrcIhx(config, orc_pars)
    orc.run_model()
    orc.network.results['Connection'].round(5).to_csv(f'outputs/orc_{case}_results.csv')
    if Tdeltah:
        orc.qt_diagram(51, 52, 63, 64, 5, case, plot=show, path=f'outputs/diagrams/qt_orc_{case}_eva.png')  # EVA
        orc.qt_diagram(66, 61, 41, 42, 5, case, plot=show, path=f'outputs/diagrams/qt_orc_{case}_cond.png')  # COND
        orc.qt_diagram(65, 66, 62, 63, 5, case, plot=show, path=f'outputs/diagrams/qt_orc_{case}_ihx.png')  # IHX
    if logph:
        orc.plot_logph(path=f'outputs/diagrams/logph_orc_{case}.png', return_diagram=show)
    return orc


# --- 4. FUNCTIONS FOR SIMULATION AND ANALYSIS -------------------------------------------------------------------------

def init_hp(hp_base):
    index = [31, 32, 33, 34, 35, 36, 21, 22, 11, 12]
    columns = ["m [kg/s]", "T [°C]", "p [bar]", "h [kJ/kg]", "s [J/kgK]", "e^PH [kJ/kg]", "x [-]", "fluid"]
    df_conns = pd.DataFrame(index=index, columns=columns)

    fluid_mapping = {
        31: hp_base.params['fluids']['working_fluid'],
        32: hp_base.params['fluids']['working_fluid'],
        33: hp_base.params['fluids']['working_fluid'],
        34: hp_base.params['fluids']['working_fluid'],
        35: hp_base.params['fluids']['working_fluid'],
        36: hp_base.params['fluids']['working_fluid'],
        21: hp_base.params['fluids']['fluid_TES'],
        22: hp_base.params['fluids']['fluid_TES'],
        11: hp_base.params['fluids']['fluid_ambient'],
        12: hp_base.params['fluids']['fluid_ambient']
    }
    df_conns['fluid'] = df_conns.index.map(fluid_mapping)

    # The following states are fixed because either they are part of the product or they are ambient streams
    fixed_state_numbers = [21, 22, 11]

    # Iterate through the state numbers
    for state_number in fixed_state_numbers:
        # Assign values to the DataFrame using the fixed state number
        df_conns.loc[state_number, 'T [°C]'] = hp_base.conns[f'c{state_number}'].T.val
        df_conns.loc[state_number, 'p [bar]'] = hp_base.conns[f'c{state_number}'].p.val
        df_conns.loc[state_number, 'h [kJ/kg]'] = hp_base.conns[f'c{state_number}'].h.val
        df_conns.loc[state_number, 's [J/kgK]'] = hp_base.conns[f'c{state_number}'].s.val
        df_conns.loc[state_number, 'm [kg/s]'] = hp_base.conns[f'c{state_number}'].m.val

    df_conns.loc[11, 'm [kg/s]'] = np.nan  # because it depends on the working conditions of EVA
    
    return df_conns


def init_orc(orc_base):
    index = [61, 62, 63, 64, 65, 66, 51, 52, 41, 42]
    columns = ["m [kg/s]", "T [°C]", "p [bar]", "h [kJ/kg]", "s [J/kgK]", "e^PH [kJ/kg]", "x [-]", "fluid"]
    df_conns = pd.DataFrame(index=index, columns=columns)

    fluid_mapping = {
        61: orc_base.params['fluids']['working_fluid'],
        62: orc_base.params['fluids']['working_fluid'],
        63: orc_base.params['fluids']['working_fluid'],
        64: orc_base.params['fluids']['working_fluid'],
        65: orc_base.params['fluids']['working_fluid'],
        66: orc_base.params['fluids']['working_fluid'],
        51: orc_base.params['fluids']['fluid_TES'],
        52: orc_base.params['fluids']['fluid_TES'],
        41: orc_base.params['fluids']['fluid_ambient'],
        42: orc_base.params['fluids']['fluid_ambient']
    }
    df_conns['fluid'] = df_conns.index.map(fluid_mapping)

    # The following states are fixed because either they are part of the product or they are ambient streams
    fixed_state_numbers = [51, 52, 41]

    # Iterate through the state numbers
    for state_number in fixed_state_numbers:
        # Assign values to the DataFrame using the fixed state number
        df_conns.loc[state_number, 'T [°C]'] = orc_base.conns[f'c{state_number}'].T.val
        df_conns.loc[state_number, 'p [bar]'] = orc_base.conns[f'c{state_number}'].p.val
        df_conns.loc[state_number, 'h [kJ/kg]'] = orc_base.conns[f'c{state_number}'].h.val
        df_conns.loc[state_number, 's [J/kgK]'] = orc_base.conns[f'c{state_number}'].s.val
        df_conns.loc[state_number, 'm [kg/s]'] = orc_base.conns[f'c{state_number}'].m.val

    df_conns.loc[41, 'm [kg/s]'] = np.nan  # because it depends on the working conditions of COND
    df_conns.loc[62, 'p [bar]'] = orc_base.conns['c62'].p.val  # pressure 61 is fixed

    return df_conns


def set_hp(df, p32, COMP=False, COND=False, VAL=False, EVA=False, IHX=False):

    # states 12 and 35 are defined according to the operating conditions of EVA
    if EVA:
        df.loc[12, 'T [°C]'] = config['ambient']['T_out_hp']  # --> very large (almost infinite mass) flow, 10 causes problems with division
        df.loc[12, 'p [bar]'] = df.loc[11, 'p [bar]']  # no pressure drops
        df.loc[34, 'T [°C]'] = df.loc[12, 'T [°C]']  # no temperature difference
        df.loc[34, 'p [bar]'] = PSI('P', 'Q', 1, 'T', df.loc[34, 'T [°C]'] + 273.15, df.loc[34, 'fluid']) * 1e-5
        df.loc[35, 'p [bar]'] = df.loc[34, 'p [bar]']

    else:
        df.loc[12, 'T [°C]'] = hp_pars_base['ambient']['T_out']
        df.loc[12, 'p [bar]'] = df.loc[11, 'p [bar]'] * hp_pars_base['eva']['pr1']
        df.loc[34, 'T [°C]'] = df.loc[12, 'T [°C]'] - hp_pars_base['eva']['ttd_l']
        df.loc[34, 'p [bar]'] = PSI('P', 'Q', 1, 'T', df.loc[34, 'T [°C]'] + 273.15, df.loc[34, 'fluid']) * 1e-5
        df.loc[35, 'p [bar]'] = df.loc[34, 'p [bar]'] * hp_pars_base['eva']['pr2']

    df.loc[12, 'h [kJ/kg]'] = PSI('H', 'P', df.loc[12, 'p [bar]'] * 1e5,
                                  'T', df.loc[12, 'T [°C]'] + 273.15, df.loc[12, 'fluid']) * 1e-3
    df.loc[12, 's [J/kgK]'] = PSI('S', 'P', df.loc[12, 'p [bar]'] * 1e5,
                                  'T', df.loc[12, 'T [°C]'] + 273.15, df.loc[12, 'fluid'])
    df.loc[35, 'T [°C]'] = PSI('T', 'Q', 1, 'P', df.loc[35, 'p [bar]'] * 1e5, df.loc[35, 'fluid']) - 273.15
    df.loc[35, 'x [-]'] = 1
    df.loc[35, 's [J/kgK]'] = PSI('S', 'Q', 1, 'T', df.loc[35, 'T [°C]'] + 273.15, df.loc[35, 'fluid'])
    df.loc[35, 'h [kJ/kg]'] = PSI('H', 'Q', 1, 'T', df.loc[35, 'T [°C]'] + 273.15, df.loc[35, 'fluid']) * 1e-3

    # state 32 is defined according to the operating conditions of COND
    if COND:
        df.loc[32, 'T [°C]'] = df.loc[21, 'T [°C]']
        df.loc[32, 'p [bar]'] = p32
        p31 = df.loc[32, 'p [bar]']  # required by COMP
    else:
        df.loc[32, 'T [°C]'] = df.loc[21, 'T [°C]'] + hp_pars_base['cond']['ttd_l']
        df.loc[32, 'p [bar]'] = p32
        p31 = df.loc[32, 'p [bar]'] / hp_pars_base['cond']['pr1']  # required by COMP

    df.loc[32, 'p [bar]'] = p32
    df.loc[32, 'h [kJ/kg]'] = PSI('H', 'P', df.loc[32, 'p [bar]'] * 1e5,
                                  'T', df.loc[32, 'T [°C]'] + 273.15, df.loc[32, 'fluid']) * 1e-3
    df.loc[32, 's [J/kgK]'] = PSI('S', 'P', df.loc[32, 'p [bar]'] * 1e5,
                                  'T', df.loc[32, 'T [°C]'] + 273.15, df.loc[32, 'fluid'])

    # the other states are calculated using the functions of the components
    heat_exchanger(df, inlet_h=32, outlet_h=33, inlet_c=35, outlet_c=36,
                   pr_hot=hp_pars_base['ihx']['pr1'], pr_cold=hp_pars_base['ihx']['pr2'],
                   ttd_u=hp_pars_base['ihx']['ttd_u'], ideal=IHX)  # calc 36

    pump(df, inlet=36, outlet=31, p_out=p31, eta_s=hp_pars_base['comp']['eta_s'], ideal=COMP)  # calc 31

    heat_exchanger(df, inlet_h=31, outlet_h=32, inlet_c=21, outlet_c=22,
                   pr_hot=hp_pars_base['cond']['pr1'], pr_cold=hp_pars_base['cond']['pr2'], ideal=COND)  # calc m31

    df.loc[33, 'm [kg/s]'] = df.loc[32, 'm [kg/s]']
    df.loc[34, 'm [kg/s]'] = df.loc[32, 'm [kg/s]']
    df.loc[35, 'm [kg/s]'] = df.loc[32, 'm [kg/s]']

    heat_exchanger(df, inlet_h=32, outlet_h=33, inlet_c=35, outlet_c=36,
                   pr_hot=hp_pars_base['ihx']['pr1'], pr_cold=hp_pars_base['ihx']['pr2'], ideal=IHX)  # calc 33

    valve(df, inlet=33, outlet=34, p_out=df.loc[34, 'p [bar]'], ideal=VAL)  # calc 34

    heat_exchanger(df, inlet_h=11, outlet_h=12, inlet_c=34, outlet_c=35,
                   pr_hot=hp_pars_base['eva']['pr1'], pr_cold=hp_pars_base['eva']['pr2'], ideal=EVA)  # calc m11

    df.loc[34, 'x [-]'] = calc_vapor_content(df.loc[34, 'p [bar]'], df.loc[34, 'h [kJ/kg]'], hp_pars_base['fluids']['working_fluid'])

    df.fillna(0, inplace=True)  # relevant in order to can round the values later on

    return df


def set_orc(df, ttd_u_eva, ttd_l_cond, PUMP=False, COND=False, EXP=False, EVA=False, IHX=False):

    # states 42 and 61 are defined according to the operating conditions of COND
    if COND:
        df.loc[42, 'T [°C]'] = config['ambient']['T_out_orc']  # --> very large (almost infinite mass) flow, 10 causes problems with division
        df.loc[42, 'p [bar]'] = df.loc[41, 'p [bar]']  # no pressure drops
        df.loc[61, 'T [°C]'] = df.loc[41, 'T [°C]']  # no temperature difference
        df.loc[61, 'p [bar]'] = PSI('P', 'Q', 1, 'T', df.loc[61, 'T [°C]'] + 273.15, df.loc[61, 'fluid']) * 1e-5
        df.loc[66, 'p [bar]'] = df.loc[61, 'p [bar]']

    else:
        df.loc[42, 'T [°C]'] = orc_pars_base['ambient']['T_out']
        df.loc[42, 'p [bar]'] = df.loc[41, 'p [bar]'] * orc_pars_base['cond']['pr2']
        df.loc[61, 'T [°C]'] = df.loc[41, 'T [°C]'] + ttd_l_cond
        df.loc[61, 'p [bar]'] = PSI('P', 'Q', 1, 'T', df.loc[61, 'T [°C]'] + 273.15, df.loc[61, 'fluid']) * 1e-5
        df.loc[66, 'p [bar]'] = df.loc[61, 'p [bar]'] / orc_pars_base['cond']['pr1']

    df.loc[42, 'h [kJ/kg]'] = PSI('H', 'P', df.loc[42, 'p [bar]'] * 1e5,
                                  'T', df.loc[42, 'T [°C]'] + 273.15, df.loc[42, 'fluid']) * 1e-3
    df.loc[42, 's [J/kgK]'] = PSI('S', 'P', df.loc[42, 'p [bar]'] * 1e5,
                                  'T', df.loc[42, 'T [°C]'] + 273.15, df.loc[42, 'fluid'])
    df.loc[61, 'x [-]'] = 0
    df.loc[61, 's [J/kgK]'] = PSI('S', 'Q', 0, 'T', df.loc[61, 'T [°C]'] + 273.15, df.loc[61, 'fluid'])
    df.loc[61, 'h [kJ/kg]'] = PSI('H', 'Q', 0, 'T', df.loc[61, 'T [°C]'] + 273.15, df.loc[61, 'fluid']) * 1e-3

    # pressure and temperature of 64 are calculated according to the pressure drops of IHX and COND and the ttd_u of EVA
    if IHX:
        df.loc[65, 'p [bar]'] = df.loc[66, 'p [bar]']
        if COND:
            df.loc[64, 'p [bar]'] = df.loc[62, 'p [bar]']
        else:
            df.loc[64, 'p [bar]'] = df.loc[62, 'p [bar]'] * orc_pars_base['cond']['pr1']
    else:
        df.loc[65, 'p [bar]'] = df.loc[66, 'p [bar]'] / orc_pars_base['ihx']['pr1']
        if COND:
            df.loc[64, 'p [bar]'] = df.loc[62, 'p [bar]'] * orc_pars_base['ihx']['pr2']
        else:
            df.loc[64, 'p [bar]'] = df.loc[62, 'p [bar]'] * orc_pars_base['cond']['pr1'] * orc_pars_base['ihx']['pr2']

    df.loc[64, 'T [°C]'] = df.loc[51, 'T [°C]'] - ttd_u_eva  # value of ttd_u_eva from sens analysis (as low as possible)
    df.loc[64, 'h [kJ/kg]'] = PSI('H', 'P', df.loc[64, 'p [bar]'] * 1e5,
                                  'T', df.loc[64, 'T [°C]'] + 273.15, df.loc[64, 'fluid']) * 1e-3
    df.loc[64, 's [J/kgK]'] = PSI('S', 'P', df.loc[64, 'p [bar]'] * 1e5,
                                  'T', df.loc[64, 'T [°C]'] + 273.15, df.loc[64, 'fluid'])

    # the other states are calculated using the functions of the components
    pump(df, inlet=61, outlet=62, p_out=df.loc[62, 'p [bar]'], eta_s=orc_pars_base['pump']['eta_s'], ideal=PUMP)  # calc 62

    turbine(df, inlet=64, outlet=65, p_out=df.loc[65, 'p [bar]'], eta_s=orc_pars_base['exp']['eta_s'], ideal=EXP)  # calc 65

    heat_exchanger(df, inlet_h=65, outlet_h=66, inlet_c=62, outlet_c=63,
                   pr_hot=orc_pars_base['ihx']['pr1'], pr_cold=orc_pars_base['ihx']['pr2'],
                   ttd_l=orc_pars_base['ihx']['ttd_l'], ideal=IHX)  # calc 66

    df.loc[62, 'm [kg/s]'] = 1
    df.loc[65, 'm [kg/s]'] = 1

    heat_exchanger(df, inlet_h=65, outlet_h=66, inlet_c=62, outlet_c=63,
                   pr_hot=orc_pars_base['ihx']['pr1'], pr_cold=orc_pars_base['ihx']['pr2'], ideal=IHX)  # calc 63

    df.loc[62, 'm [kg/s]'] = np.nan
    df.loc[63, 'm [kg/s]'] = np.nan
    df.loc[65, 'm [kg/s]'] = np.nan
    df.loc[66, 'm [kg/s]'] = np.nan

    heat_exchanger(df, inlet_h=51, outlet_h=52, inlet_c=63, outlet_c=64,
                   pr_hot=orc_pars_base['eva']['pr1'], pr_cold=orc_pars_base['eva']['pr2'], ideal=EVA)  # calc m61

    df.loc[61, 'm [kg/s]'] = df.loc[63, 'm [kg/s]']
    df.loc[62, 'm [kg/s]'] = df.loc[63, 'm [kg/s]']
    df.loc[65, 'm [kg/s]'] = df.loc[63, 'm [kg/s]']
    df.loc[66, 'm [kg/s]'] = df.loc[63, 'm [kg/s]']

    heat_exchanger(df, inlet_h=66, outlet_h=61, inlet_c=41, outlet_c=42,
                   pr_hot=orc_pars_base['cond']['pr1'], pr_cold=orc_pars_base['cond']['pr2'], ideal=COND)  # calc m41
    
    df.fillna(0, inplace=True)  # relevant in order to can round the values later on

    return df


def check_balance_hp(df_conns, case):
    index = ['COMP', 'COND', 'IHX', 'VAL', 'EVA', 'SUM']
    columns = ['P [kW]', 'Q [kW]', 'S_gen [kW/K]', 'E_D [kW]']
    df_comps = pd.DataFrame(index=index, columns=columns, data=0)

    turbo_mach_bal(df_comps, df_conns, 36, 31, 'COMP')
    he_bal(df_comps, df_conns, 31, 32, 21, 22, 'COND')
    he_bal(df_comps, df_conns, 32, 33, 35, 36, 'IHX')
    valve_bal(df_comps, df_conns, 33, 34, 'VAL')
    he_bal(df_comps, df_conns, 11, 12, 34, 35, 'EVA')

    df_comps.loc['SUM', 'P [kW]'] = df_comps.iloc[0:5, 0].sum()
    df_comps.loc['SUM', 'Q [kW]'] = df_comps.iloc[0:5, 1].sum()
    df_comps.loc['SUM', 'S_gen [kW/K]'] = df_comps.iloc[0:5, 2].sum()

    P_in = df_comps.loc['SUM', 'P [kW]']
    Q_in = df_conns.loc[11, 'm [kg/s]'] * (df_conns.loc[11, 'h [kJ/kg]'] - df_conns.loc[12, 'h [kJ/kg]'])
    Q_out = df_conns.loc[21, 'm [kg/s]'] * (df_conns.loc[22, 'h [kJ/kg]'] - df_conns.loc[21, 'h [kJ/kg]'])

    tot_bal = P_in + Q_in - Q_out

    if abs(tot_bal) > 1e-5:
        print(f'Attention! Energy balance equation of the total HP for the case {case} is violated by', str(round(tot_bal, 3)), 'kW.')

    df_comps['E_D [kW]'] = df_comps['S_gen [kW/K]'] * T_0

    return df_comps


def check_balance_orc(df_conns, case):
    index = ['EVA', 'EXP', 'IHX', 'COND', 'PUMP', 'SUM']
    columns = ['P [kW]', 'Q [kW]', 'S_gen [kW/K]', 'E_D [kW]']
    df_comps = pd.DataFrame(index=index, columns=columns, data=0)

    turbo_mach_bal(df_comps, df_conns, 61, 62, 'PUMP')
    he_bal(df_comps=df_comps, df_conns=df_conns, inlet_h=66, outlet_h=61, inlet_c=41, outlet_c=42, label='COND')
    he_bal(df_comps=df_comps, df_conns=df_conns, inlet_h=51, outlet_h=52, inlet_c=63, outlet_c=64, label='EVA')
    he_bal(df_comps=df_comps, df_conns=df_conns, inlet_h=65, outlet_h=66, inlet_c=62, outlet_c=63, label='IHX')
    turbo_mach_bal(df_comps, df_conns, 64, 65, 'EXP')

    df_comps.loc['SUM', 'P [kW]'] = df_comps.iloc[0:5, 0].sum()
    df_comps.loc['SUM', 'Q [kW]'] = df_comps.iloc[0:5, 1].sum()
    df_comps.loc['SUM', 'S_gen [kW/K]'] = df_comps.iloc[0:5, 2].sum()

    P_out = df_comps.loc['SUM', 'P [kW]']
    Q_in = df_conns.loc[51, 'm [kg/s]'] * (df_conns.loc[51, 'h [kJ/kg]'] - df_conns.loc[52, 'h [kJ/kg]'])
    Q_out = df_conns.loc[41, 'm [kg/s]'] * (df_conns.loc[42, 'h [kJ/kg]'] - df_conns.loc[41, 'h [kJ/kg]'])

    tot_bal = Q_in - Q_out + P_out

    if abs(tot_bal) > 1e-5:
        print(f'Attention! Energy balance equation of the total HP for the case {case} is violated by', str(round(tot_bal, 3)), 'kW.')

    df_comps['E_D [kW]'] = df_comps['S_gen [kW/K]'] * T_0

    return df_comps


def exan_hp(df):

    h_0_wf = h_0_fluids[df.loc[31, 'fluid']]
    s_0_wf = s_0_fluids[df.loc[31, 'fluid']]
    h_0_TES = h_0_fluids[df.loc[21, 'fluid']]
    s_0_TES = s_0_fluids[df.loc[21, 'fluid']]
    h_0_amb = h_0_fluids[df.loc[11, 'fluid']]
    s_0_amb = s_0_fluids[df.loc[11, 'fluid']]

    for i in [31, 32, 33, 34, 35, 36]:
        df.loc[i, 'e^PH [kJ/kg]'] = df.loc[i, 'h [kJ/kg]'] - h_0_wf - T_0 * (df.loc[i, 's [J/kgK]'] - s_0_wf) * 1e-3
    for i in [21, 22]:
        df.loc[i, 'e^PH [kJ/kg]'] = df.loc[i, 'h [kJ/kg]'] - h_0_TES - T_0 * (df.loc[i, 's [J/kgK]'] - s_0_TES) * 1e-3
    for i in [11, 12]:
        df.loc[i, 'e^PH [kJ/kg]'] = df.loc[i, 'h [kJ/kg]'] - h_0_amb - T_0 * (df.loc[i, 's [J/kgK]'] - s_0_amb) * 1e-3

    return df


def exan_orc(df):

    h_0_wf = h_0_fluids[df.loc[61, 'fluid']]
    s_0_wf = s_0_fluids[df.loc[61, 'fluid']]
    h_0_TES = h_0_fluids[df.loc[51, 'fluid']]
    s_0_TES = s_0_fluids[df.loc[51, 'fluid']]
    h_0_amb = h_0_fluids[df.loc[41, 'fluid']]
    s_0_amb = s_0_fluids[df.loc[41, 'fluid']]

    for i in [61, 62, 63, 64, 65, 66]:
        df.loc[i, 'e^PH [kJ/kg]'] = df.loc[i, 'h [kJ/kg]'] - h_0_wf - T_0 * (df.loc[i, 's [J/kgK]'] - s_0_wf) * 1e-3
    for i in [51, 52]:
        df.loc[i, 'e^PH [kJ/kg]'] = df.loc[i, 'h [kJ/kg]'] - h_0_TES - T_0 * (df.loc[i, 's [J/kgK]'] - s_0_TES) * 1e-3
    for i in [41, 42]:
        df.loc[i, 'e^PH [kJ/kg]'] = df.loc[i, 'h [kJ/kg]'] - h_0_amb - T_0 * (df.loc[i, 's [J/kgK]'] - s_0_amb) * 1e-3

    return df


def perf_adex_hp(hp_base, p32, components, check_temp_diff=False):

    # Generate all possible combinations of component IDs
    combos = product([False, True], repeat=len(components))

    for combo in combos:
        # Map the combination of component IDs to the component names
        comp_ids = dict(zip(components, combo))

        # Initialize hp_base with the current combination of component IDs
        df_conns_set = init_hp(hp_base)

        # Set the hp_base connections with the specified parameters
        df_conns = set_hp(df_conns_set, p32, **comp_ids)

        # get the case as a string
        case = '_'.join([comp for comp, comp_id in comp_ids.items() if not comp_id])

        # Check the mass and energy balance
        df_comps = check_balance_hp(df_conns, case)

        # Save the results to CSV files
        df_comps.round(3).to_csv(f'outputs/adex_hp/adex_hp_{case}_comps.csv')
        df_conns = exan_hp(df_conns)
        df_conns.round(3).to_csv(f'outputs/adex_hp/adex_hp_{case}_conns.csv')

        if check_temp_diff:
            qt_diagram(df_conns, 'COND', 31, 32, 21, 22, 0, 'HP', case, path=f'outputs/diagrams/adex_hp_qt_condenser_{case}.png', step_number=100)
            qt_diagram(df_conns, 'EVA', 11, 12, 34, 35, 0, 'HP', case, path=f'outputs/diagrams/adex_hp_qt_evapator_{case}.png', step_number=100)
            qt_diagram(df_conns, 'IHX', 32, 33, 35, 36, 0, 'HP', case, path=f'outputs/diagrams/adex_hp_qt_ihx_{case}.png', step_number=100)

    # List to store DataFrames for each combination
    ED_list = []

    # Generate all possible combinations of 1 or 2 components
    for r in range(1, 3):
        component_combinations = [comb for comb in combinations(components, r)]
        for combo in component_combinations:
            # Form the file prefix based on the selected components
            file_prefix = '_'.join(combo)

            # Load the corresponding DataFrame (replace with your actual file paths)
            df_comps = pd.read_csv(f'outputs/adex_hp/adex_hp_{file_prefix}_comps.csv')

            # Extract the relevant data and add it to the list
            df_subset = df_comps['E_D [kW]'].copy()

            df_subset.name = '_'.join([comp for comp in components if comp in combo])
            ED_list.append(df_subset)

    # Load the DataFrame where all components are real
    df_all_real = pd.read_csv('outputs/adex_hp/adex_hp_COMP_COND_IHX_VAL_EVA_comps.csv')

    # Extract the relevant column and add it to the list
    df_all_real_subset = df_all_real['E_D [kW]'].copy()
    df_all_real_subset.name = 'base'
    ED_list.insert(0, df_all_real_subset)

    # Concatenate the DataFrames along the columns (axis=1)
    result_df = pd.concat(ED_list, axis=1)
    result_df.index = components + ['TOT']

    for comp in components:
        result_df.loc[comp, 'endo'] = result_df.loc[comp, comp]
    result_df['exo'] = result_df['base'] - result_df['endo']

    for comp in components:
        other_components = [c for c in components if c != comp]
        for other_comp in other_components:
            if comp+'_'+other_comp in result_df.columns:
                result_df.loc[comp, f'exo kl {other_comp}'] = result_df.loc[comp, comp+'_'+other_comp] - result_df.loc[comp, 'endo']
            if other_comp+'_'+comp in result_df.columns:
                result_df.loc[comp, f'exo kl {other_comp}'] = result_df.loc[comp, other_comp+'_'+comp] - result_df.loc[comp, 'endo']

    for comp in components:
        other_components = [c for c in components if c != comp]
        result_df.loc[comp, 'mexo'] = result_df.loc[comp, 'exo']
        for other_comp in other_components:
            result_df.loc[comp, 'mexo'] = result_df.loc[comp, 'mexo'] - result_df.loc[comp, f'exo kl {other_comp}']

    for comp in components:
        result_df.loc[comp, f'm31 {comp}'] = pd.read_csv(f'outputs/adex_hp/adex_hp_{comp}_conns.csv').iloc[1, 1]
        other_components = [c for c in components if c != comp]
        for other_comp in other_components:
            if (comp, other_comp) in combinations(components, 2):
                result_df.loc[comp, f'm31 {other_comp}'] = pd.read_csv(f'outputs/adex_hp/adex_hp_{comp}_{other_comp}_conns.csv').iloc[1, 1]
            if (other_comp, comp) in combinations(components, 2):
                result_df.loc[comp, f'm31 {other_comp}'] = pd.read_csv(f'outputs/adex_hp/adex_hp_{other_comp}_{comp}_conns.csv').iloc[1, 1]

    # Save the result to a new CSV file
    result_df.round(3).to_csv('outputs/adex_hp/E_D_values_hp.csv')


def perf_adex_orc(orc_base, ttd_u_eva, ttd_l_cond, components, check_temp_diff=False):

    # Generate all possible combinations of component IDs
    combos = product([False, True], repeat=len(components))

    for combo in combos:
        # Map the combination of component IDs to the component names
        comp_ids = dict(zip(components, combo))

        # Initialize hp_base with the current combination of component IDs
        df_comp_cond_conns_set = init_orc(orc_base)

        # Set the hp_base connections with the specified parameters
        df_comp_cond_conns = set_orc(df_comp_cond_conns_set, ttd_u_eva, ttd_l_cond, **comp_ids)

        # get the case as a string
        case = '_'.join([comp for comp, comp_id in comp_ids.items() if not comp_id])

        # Check the mass and energy balance
        df_comp_cond_comps = check_balance_orc(df_comp_cond_conns, case)

        # Save the results to CSV files
        df_comp_cond_comps.round(3).to_csv(f'outputs/adex_orc/adex_orc_{case}_comps.csv')
        df_comp_cond_conns = exan_orc(df_comp_cond_conns)
        df_comp_cond_conns.round(3).to_csv(f'outputs/adex_orc/adex_orc_{case}_conns.csv')

        if check_temp_diff:
            qt_diagram(df_comp_cond_conns, 'EVA', 51, 52, 63, 64, 0, 'ORC',
                       case, path=f'outputs/diagrams/adex_orc_qt_evaporator_{case}.png', step_number=100)
            qt_diagram(df_comp_cond_conns, 'COND', 66, 61, 41, 42, 0, 'ORC',
                       case, path=f'outputs/diagrams/adex_orc_qt_condenser_{case}.png', step_number=100)
            qt_diagram(df_comp_cond_conns, 'IHX', 65, 66, 62, 63, 0, 'ORC',
                       case, path=f'outputs/diagrams/adex_orc_qt_ihx_{case}.png', step_number=100)


    # List to store DataFrames for each combination
    ED_list = []

    # Generate all possible combinations of 1 or 2 components
    for r in range(1, 3):
        component_combinations = [comb for comb in combinations(components, r)]
        for combo in component_combinations:
            # Form the file prefix based on the selected components
            file_prefix = '_'.join(combo)

            # Load the corresponding DataFrame (replace with your actual file paths)
            df_comps = pd.read_csv(f'outputs/adex_orc/adex_orc_{file_prefix}_comps.csv')

            # Extract the relevant data and add it to the list
            df_subset = df_comps['E_D [kW]'].copy()

            df_subset.name = '_'.join([comp for comp in components if comp in combo])
            ED_list.append(df_subset)

    # Load the DataFrame where all components are real
    df_all_real = pd.read_csv('outputs/adex_orc/adex_orc_EVA_EXP_IHX_COND_PUMP_comps.csv')

    # Extract the relevant column and add it to the list
    df_all_real_subset = df_all_real['E_D [kW]'].copy()
    df_all_real_subset.name = 'base'
    ED_list.insert(0, df_all_real_subset)

    # Concatenate the DataFrames along the columns (axis=1)
    result_df = pd.concat(ED_list, axis=1)
    result_df.index = components + ['TOT']

    for comp in components:
        result_df.loc[comp, 'endo'] = result_df.loc[comp, comp]
    result_df['exo'] = result_df['base'] - result_df['endo']

    for comp in components:
        other_components = [c for c in components if c != comp]
        for other_comp in other_components:
            if comp+'_'+other_comp in result_df.columns:
                result_df.loc[comp, f'exo kl {other_comp}'] = result_df.loc[comp, comp+'_'+other_comp] - result_df.loc[comp, 'endo']
            if other_comp+'_'+comp in result_df.columns:
                result_df.loc[comp, f'exo kl {other_comp}'] = result_df.loc[comp, other_comp+'_'+comp] - result_df.loc[comp, 'endo']

    for comp in components:
        other_components = [c for c in components if c != comp]
        result_df.loc[comp, 'mexo'] = result_df.loc[comp, 'exo']
        for other_comp in other_components:
            result_df.loc[comp, 'mexo'] = result_df.loc[comp, 'mexo'] - result_df.loc[comp, f'exo kl {other_comp}']

    for comp in components:
        result_df.loc[comp, f'm61 {comp}'] = pd.read_csv(f'outputs/adex_orc/adex_orc_{comp}_conns.csv').iloc[1, 1]
        other_components = [c for c in components if c != comp]
        for other_comp in other_components:
            if (comp, other_comp) in combinations(components, 2):
                result_df.loc[comp, f'm61 {other_comp}'] = pd.read_csv(f'outputs/adex_orc/adex_orc_{comp}_{other_comp}_conns.csv').iloc[1, 1]
            if (other_comp, comp) in combinations(components, 2):
                result_df.loc[comp, f'm61 {other_comp}'] = pd.read_csv(f'outputs/adex_orc/adex_orc_{other_comp}_{comp}_conns.csv').iloc[1, 1]

    # Save the result to a new CSV file
    result_df.round(3).to_csv('outputs/adex_orc/E_D_values_orc.csv')