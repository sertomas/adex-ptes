from CoolProp.CoolProp import PropsSI as PSI
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# --- functions --------------------------------------------------------------------------------------------------------
def calc_cycle(T_0, T_12, p_0, min_temp_diff, pr_drop_ratio, T_21, T_22, p_TES, m_31, p_31, eta_s, working_fluid,ambient_fluid, TES_fluid, ihx=False, val=False, path=None):

    hp_streams = pd.DataFrame(columns=['m [kg/s]', 'T [°C]', 'p [bar]', 'h [kJ/kg]', 's [J/kgK]', 'x [-]'])

    x_35 = 1  # saturated vapor
    T_35 = T_12 - min_temp_diff
    p_35 = PSI('P', 'T', T_35 + 273.15, 'Q', x_35, working_fluid) * 1e-5
    h_35 = PSI('H', 'T', T_35 + 273.15, 'Q', x_35, working_fluid) * 1e-3
    s_35 = PSI('S', 'T', T_35 + 273.15, 'Q', x_35, working_fluid)

    p_36 = p_35 * pr_drop_ratio
    T_36 = T_21
    h_36 = PSI('H', 'T', T_36 + 273.15, 'P', p_36 * 1e5, working_fluid) * 1e-3
    s_36 = PSI('S', 'T', T_36 + 273.15, 'P', p_36 * 1e5, working_fluid)

    h_31s = PSI('H', 'S', s_36, 'P', p_31 * 1e5, working_fluid) * 1e-3
    h_31 = (h_31s - h_36) / eta_s + h_36
    T_31 = PSI('T', 'H', h_31 * 1e3, 'P', p_31 * 1e5, working_fluid) - 273.15
    s_31 = PSI('S', 'T', T_31 + 273.15, 'P', p_31 * 1e5, working_fluid)

    p_32 = p_31 * pr_drop_ratio
    T_32 = T_21 - min_temp_diff
    h_32 = PSI('H', 'T', T_32 + 273.15, 'P', p_32 * 1e5, working_fluid) * 1e-3
    s_32 = PSI('S', 'T', T_32 + 273.15, 'P', p_32 * 1e5, working_fluid)

    if not ihx:
        p_33 = p_32 * pr_drop_ratio
        h_33 = h_32 + h_35 - h_36
        T_33 = PSI('T', 'H', h_33 * 1e3, 'P', p_33 * 1e5, working_fluid) - 273.15
        s_33 = PSI('S', 'T', T_33 + 273.15, 'P', p_33 * 1e5, working_fluid)
    else:
        p_33 = p_32 * pr_drop_ratio
        s_33 = s_32 + s_35 - s_36
        T_33 = PSI('T', 'S', s_33, 'P', p_33 * 1e5, working_fluid) - 273.15
        h_33 = PSI('S', 'T', T_33 + 273.15, 'P', p_33 * 1e5, working_fluid) * 1e-3

    if not val:
        h_34 = h_33
        p_34 = p_35 / pr_drop_ratio
        T_34 = PSI('T', 'H', h_34 * 1e3, 'P', p_34 * 1e5, working_fluid) - 273.15
        s_34 = PSI('S', 'H', h_34 * 1e3, 'P', p_34 * 1e5, working_fluid)
        h_34i = PSI('H', 'P', p_34 * 1e5, 'Q', 0, working_fluid) * 1e-3
        h_34ii = PSI('H', 'P', p_34 * 1e5, 'Q', 1, working_fluid) * 1e-3
        x_34 = (h_34 - h_34i) / (h_34ii - h_34i)
    else:
        s_34 = s_33
        p_34 = p_35 / pr_drop_ratio
        T_34 = PSI('T', 'S', s_34, 'P', p_34 * 1e5, working_fluid) - 273.15
        h_34 = PSI('S', 'S', s_34, 'P', p_34 * 1e5, working_fluid) * 1e-3
        h_34i = PSI('H', 'P', p_34 * 1e5, 'Q', 0, working_fluid) * 1e-3
        h_34ii = PSI('H', 'P', p_34 * 1e5, 'Q', 1, working_fluid) * 1e-3
        x_34 = (h_34 - h_34i) / (h_34ii - h_34i)

    Q_cond = m_31 * (h_31 - h_32)
    Q_ihx = m_31 * (h_32 - h_33)
    Q_eva = m_31 * (h_35 - h_34)
    P_comp = m_31 * (h_31 - h_36)

    p_21 = p_TES
    p_22 = p_21
    h_21 = PSI('H', 'T', T_21 + 273.15, 'P', p_TES * 1e5, TES_fluid) * 1e-3
    s_21 = PSI('S', 'T', T_21 + 273.15, 'P', p_TES * 1e5, TES_fluid)
    h_22 = PSI('H', 'T', T_22 + 273.15, 'P', p_TES * 1e5, TES_fluid) * 1e-3
    s_22 = PSI('S', 'T', T_22 + 273.15, 'P', p_TES * 1e5, TES_fluid)

    m_21 = Q_cond / (h_22 - h_21)

    T_11 = T_0
    p_11 = p_0
    p_12 = p_11
    h_11 = PSI('H', 'T', T_11 + 273.15, 'P', p_11 * 1e5, ambient_fluid) * 1e-3
    s_11 = PSI('S', 'T', T_11 + 273.15, 'P', p_11 * 1e5, ambient_fluid)
    h_12 = PSI('H', 'T', T_12 + 273.15, 'P', p_12 * 1e5, ambient_fluid) * 1e-3
    s_12 = PSI('S', 'T', T_12 + 273.15, 'P', p_12 * 1e5, ambient_fluid)
    m_11 = Q_eva / (h_11 - h_12)

    values = [
        [m_31, T_31, p_31, h_31, s_31, np.nan],
        [m_31, T_32, p_32, h_32, s_32, np.nan],
        [m_31, T_33, p_33, h_33, s_33, np.nan],
        [m_31, T_34, p_34, h_34, s_34, x_34],
        [m_31, T_35, p_35, h_35, s_35, x_35],
        [m_31, T_36, p_36, h_36, s_36, np.nan],
        [m_11, T_11, p_11, h_11, s_11, np.nan],
        [m_11, T_12, p_12, h_12, s_12, np.nan],
        [m_21, T_21, p_21, h_21, s_21, np.nan],
        [m_21, T_22, p_22, h_22, s_22, np.nan]
    ]

    for i, point in enumerate(values):
        hp_streams.loc[i] = point

    hp_streams.index = ['31', '32', '33', '34', '35', '36', '11', '12', '21', '22']

    hp_streams.round(5).to_csv(path)

    COP = (h_31 - h_32) / (h_31 - h_36)
    print('P_in =', round(P_comp, 1), 'kW')
    print('Q_out =', round(Q_cond, 1), 'kW')
    print('COP =', round(COP, 3))
    print('m_21 =', round(m_21, 3), 'kg/s')
    print('m_11 =', round(m_11, 3), 'kg/s')


def qt_diagram(T_hot_out, T_cold_in, h_hot_in, h_hot_out, h_cold_in, h_cold_out, p_hot_in, p_hot_out, p_cold_in, p_cold_out, fluid_hot, fluid_cold, name_he, delta_t_min, mass_flow_cold, plot=False, path=None):

    step_number = 100
    T_cold = [T_cold_in]
    T_hot = [T_hot_out]
    H_plot = [0]

    for i in np.linspace(1, step_number, step_number):
        h_hot = (h_hot_in - h_hot_out) / step_number * i + h_hot_out
        p_hot = (p_hot_in - p_hot_out) / step_number * i + p_hot_in
        T_hot.append(PSI('T', 'H', h_hot * 1e3, 'P', p_hot * 1e5, fluid_hot) - 273.15)
        h_cold = (h_cold_out - h_cold_in) / step_number * i + h_cold_in
        p_cold = (p_cold_in - p_cold_out) / step_number * i + p_cold_in
        T_cold.append(PSI('T', 'H', h_cold * 1e3, 'P', p_cold * 1e5, fluid_cold) - 273.15)
        H_plot.append((h_cold - h_cold_in) * mass_flow_cold)

    difference = [x - y for x, y in zip(T_hot, T_cold)]

    # plot the results
    mpl.rcParams['font.size'] = 16
    plt.figure(figsize=(14, 7))
    plt.plot(H_plot, T_hot, color='red')
    plt.plot(H_plot, T_cold, color='blue')
    plt.legend(["Hot side", "Cold side"])
    plt.xlabel('Q [kW]')
    plt.grid(True)
    plt.ylabel('T [°C]')
    plt.xlim(0, max(H_plot))
    plt.ylim(min(T_cold) - 10, max(T_hot) + 10)
    plt.title("QT diagram of the " + name_he + " of the HP")

    if path is not None:
        plt.savefig(path)
    if plot:
        plt.show()

    if min(difference) > delta_t_min - 1e-5:
        print("The min. temperature difference of the " + name_he + " of the HP is " + str(
            round(min(difference), 2)) + "K and is equal or higher than the allowed min. (" + str(
            delta_t_min) + "K).")
    else:
        print("The min. temperature difference of the " + name_he + " of the HP is " + str(
            round(min(difference), 2)) + "K and is lower than the allowed min. (" + str(
            delta_t_min) + "K).")


def exergy_analysis(cycle, T_0, p_0, working_fluid, ambient_fluid, TES_fluid):
    hp_streams = pd.read_csv(cycle, index_col=0)

    h_0_wf = PSI('H', 'T', T_0 + 273.15, 'P', p_0 * 1e5, working_fluid) * 1e-3
    s_0_wf = PSI('S', 'T', T_0 + 273.15, 'P', p_0 * 1e5, working_fluid)
    h_0_amb = PSI('H', 'T', T_0 + 273.15, 'P', p_0 * 1e5, ambient_fluid) * 1e-3
    s_0_amb = PSI('S', 'T', T_0 + 273.15, 'P', p_0 * 1e5, ambient_fluid)
    h_0_TES = PSI('H', 'T', T_0 + 273.15, 'P', p_0 * 1e5, TES_fluid) * 1e-3
    s_0_TES = PSI('S', 'T', T_0 + 273.15, 'P', p_0 * 1e5, TES_fluid)

    exergy_streams = []
    for stream in [31, 32, 33, 34, 35, 36]:
        m_i = hp_streams.at[stream, 'm [kg/s]']
        T_i = hp_streams.at[stream, 'T [°C]']
        p_i = hp_streams.at[stream, 'p [bar]']
        h_i = hp_streams.at[stream, 'h [kJ/kg]']
        s_i = hp_streams.at[stream, 's [J/kgK]']
        x_i = hp_streams.at[stream, 'x [-]']

        e_PH_i = h_i - h_0_wf - (T_0+273.15) * (s_i - s_0_wf) * 1e-3

        h_A_i = PSI('H', 'T', T_0 + 273.15, 'P', p_i * 1e5, working_fluid) * 1e-3
        s_A_i = PSI('S', 'T', T_0 + 273.15, 'P', p_i * 1e5, working_fluid)
        e_T_i = h_i - h_A_i - (T_0+273.15) * (s_i - s_A_i) * 1e-3
        e_M_i = e_PH_i - e_T_i

        exergy_streams.append((stream, m_i, T_i, p_i, e_PH_i, e_T_i, e_M_i, h_i, s_i, x_i))

    for stream in [11, 12]:
        m_i = hp_streams.at[stream, 'm [kg/s]']
        T_i = hp_streams.at[stream, 'T [°C]']
        p_i = hp_streams.at[stream, 'p [bar]']
        h_i = hp_streams.at[stream, 'h [kJ/kg]']
        s_i = hp_streams.at[stream, 's [J/kgK]']
        x_i = hp_streams.at[stream, 'x [-]']

        e_PH_i = h_i - h_0_amb - (T_0+273.15) * (s_i - s_0_amb) * 1e-3

        h_A_i = PSI('H', 'T', T_0 + 273.15, 'P', p_i * 1e5, ambient_fluid) * 1e-3
        s_A_i = PSI('S', 'T', T_0 + 273.15, 'P', p_i * 1e5, ambient_fluid)
        e_T_i = h_i - h_A_i - (T_0+273.15) * (s_i - s_A_i) * 1e-3
        e_M_i = e_PH_i - e_T_i

        exergy_streams.append((stream, m_i, T_i, p_i, e_PH_i, e_T_i, e_M_i, h_i, s_i, x_i))

    for stream in [21, 22]:
        m_i = hp_streams.at[stream, 'm [kg/s]']
        T_i = hp_streams.at[stream, 'T [°C]']
        p_i = hp_streams.at[stream, 'p [bar]']
        h_i = hp_streams.at[stream, 'h [kJ/kg]']
        s_i = hp_streams.at[stream, 's [J/kgK]']
        x_i = hp_streams.at[stream, 'x [-]']

        e_PH_i = h_i - h_0_TES - (T_0+273.15) * (s_i - s_0_TES) * 1e-3

        h_A_i = PSI('H', 'T', T_0 + 273.15, 'P', p_i * 1e5, TES_fluid) * 1e-3
        s_A_i = PSI('S', 'T', T_0 + 273.15, 'P', p_i * 1e5, TES_fluid)
        e_T_i = h_i - h_A_i - (T_0+273.15) * (s_i - s_A_i) * 1e-3
        e_M_i = e_PH_i - e_T_i

        exergy_streams.append((stream, m_i, T_i, p_i, e_PH_i, e_T_i, e_M_i, h_i, s_i, x_i))

    exergy_streams_df = pd.DataFrame(exergy_streams, columns=['Stream', 'm [kg/s]', 'T [°C]', 'p [bar]', 'e^PH [kJ/kg]', 'e^T [kJ/kg]', 'e^M [kJ/kg]', 'h [kJ/kg]', 's [J/kgK]', 'x [-]'])
    exergy_streams_df.set_index('Stream', inplace=True)

    exergy_streams_df.round(5).to_csv('simple_code_exergy.csv')

    components = {
        'COMP': {
            'E_F': hp_streams.loc[31, 'm [kg/s]'] * (hp_streams.loc[31, 'h [kJ/kg]'] - hp_streams.loc[36, 'h [kJ/kg]']),
            'E_P': hp_streams.loc[31, 'm [kg/s]'] * (exergy_streams_df.loc[31, 'e^PH [kJ/kg]'] - exergy_streams_df.loc[36, 'e^PH [kJ/kg]']),
        },
        'COND': {
            'E_F': hp_streams.loc[31, 'm [kg/s]'] * (exergy_streams_df.loc[31, 'e^PH [kJ/kg]'] - exergy_streams_df.loc[32, 'e^PH [kJ/kg]']),
            'E_P': hp_streams.loc[21, 'm [kg/s]'] * (exergy_streams_df.loc[22, 'e^PH [kJ/kg]'] - exergy_streams_df.loc[21, 'e^PH [kJ/kg]']),
        },
        'IHX': {
            'E_F': hp_streams.loc[31, 'm [kg/s]'] * (exergy_streams_df.loc[32, 'e^PH [kJ/kg]'] - exergy_streams_df.loc[33, 'e^PH [kJ/kg]']),
            'E_P': hp_streams.loc[31, 'm [kg/s]'] * (exergy_streams_df.loc[36, 'e^PH [kJ/kg]'] - exergy_streams_df.loc[35, 'e^PH [kJ/kg]']),
        },
        'VAL': {
            'E_F': hp_streams.loc[31, 'm [kg/s]'] * (exergy_streams_df.loc[33, 'e^PH [kJ/kg]'] - exergy_streams_df.loc[34, 'e^M [kJ/kg]']),
            'E_P': hp_streams.loc[31, 'm [kg/s]'] * exergy_streams_df.loc[34, 'e^T [kJ/kg]'],
        },
        'EVA': {
            'E_F': hp_streams.loc[31, 'm [kg/s]'] * (exergy_streams_df.loc[34, 'e^PH [kJ/kg]'] - exergy_streams_df.loc[35, 'e^PH [kJ/kg]']) + hp_streams.loc[11, 'm [kg/s]'] * (exergy_streams_df.loc[11, 'e^M [kJ/kg]'] - exergy_streams_df.loc[12, 'e^M [kJ/kg]']),
            'E_P': hp_streams.loc[11, 'm [kg/s]'] * (exergy_streams_df.loc[12, 'e^T [kJ/kg]'] - exergy_streams_df.loc[11, 'e^T [kJ/kg]']),
        }
    }

    # Create a DataFrame
    exergy_components_df = pd.DataFrame(components).T  # Transpose to have 'COMP', 'COND', 'IHX', 'VAL', 'EVA' as rows

    # Calculate the 'E_D' column
    exergy_components_df['E_D'] = exergy_components_df['E_F'] - exergy_components_df['E_P']

    # Total system
    exergy_components_df.loc['TOT', 'E_F'] = exergy_components_df.loc['COMP', 'E_F']
    exergy_components_df.loc['TOT', 'E_P'] = exergy_components_df.loc['COND', 'E_P']
    exergy_components_df.loc['TOT', 'E_D'] = exergy_components_df.loc['COMP', 'E_D'] + exergy_components_df.loc['COND', 'E_D'] + exergy_components_df.loc['IHX', 'E_D'] + exergy_components_df.loc['VAL', 'E_D'] + exergy_components_df.loc['EVA', 'E_D']
    exergy_components_df.loc['TOT', 'E_L'] = exergy_components_df.loc['TOT', 'E_F'] - exergy_components_df.loc['TOT', 'E_P'] - exergy_components_df.loc['TOT', 'E_D']

    # Calculate the 'epsilon' column
    exergy_components_df['epsilon'] = exergy_components_df['E_P'] / exergy_components_df['E_F']

    # Print the DataFrame
    print(exergy_components_df[['E_F', 'E_P', 'E_D', 'E_L', 'epsilon']])

# --- parameters and variables -----------------------------------------------------------------------------------------
T_0 = 10  # ambient temperature [°C]
p_0 = 1.013  # ambient pressure [bar]

T_12 = 5  # outlet temperature of heat source [°C]
T_21 = 70  # LT of TES[°C]
T_22 = 140  # HT of TES [°C]
p_TES = 5  # pressure of TES [bar]
min_temp_diff = 5  # minimum temperature difference of all heat exchangers [K]

eta_s = 0.85
pr_drop_ratio = 0.99
working_fluid = 'REFPROP::R245fa'
TES_fluid = 'REFPROP::water'
ambient_fluid = 'REFPROP::water'

p_31 = 21  # free parameter --> should be optimized (pinch point in condenser)
m_31 = 10  # [kg/s] for the dimensioning

# calc_cycle(T_0, T_12, p_0, min_temp_diff, pr_drop_ratio, T_21, T_22, p_TES, m_31, p_31, eta_s, working_fluid, ambient_fluid, TES_fluid, ihx=False, val=False, path='simple_code_results.csv')

# qt_diagram(T_33, T_35, h_32, h_33, h_35, h_36, p_32, p_33, p_35, p_36, working_fluid, working_fluid, 'Internal Heat Exchanger', min_temp_diff, m_31, plot=True, path=None)

# qt_diagram(T_32, T_21, h_31, h_32, h_21, h_22, p_31, p_32, p_21, p_21, working_fluid, TES_fluid, 'Condenser', min_temp_diff, m_21, plot=True, path=None)

# qt_diagram(T_12, T_34, h_11, h_12, h_34, h_35, p_11, p_12, p_34, p_35, ambient_fluid, working_fluid, 'Evaporator', min_temp_diff, m_31, plot=True, path=None)


# --- exergy analysis --------------------------------------------------------------------------------------------------
# exergy_analysis('simple_code_results.csv', T_0, p_0, working_fluid, ambient_fluid, TES_fluid)


# --- Idealized cycle -----------------------------------------------------------------------------------------

calc_cycle(T_0, 9.999999, p_0, 0, 1, T_21, T_22, p_TES, m_31, p_31, 1, working_fluid, ambient_fluid, TES_fluid, ihx=True, val=True, path='simple_code_results_ideal.csv')

exergy_analysis('simple_code_results_ideal.csv', T_0, p_0, working_fluid, ambient_fluid, TES_fluid)


# TODO:     1) correct exergy function with ideal / real
# TODO:     2) check every calculation because values changed a bit
