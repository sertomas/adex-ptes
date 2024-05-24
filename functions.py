import json
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI as PSI

with open('inputs/hp_simult_base.json', 'r') as file:
    hp_config = json.load(file)
t0 = hp_config['ambient']['t0']  # K


def qt_diagram(df, component_name, hot_in, hot_out, cold_in, cold_out, delta_t_min, system, case, plot=False, path=None,
               step_number=200):
    """
    Generate a QT diagram for a specified component in a thermal system.

    Args:
        df (pd.DataFrame): DataFrame containing relevant thermodynamic properties.
        component_name (str): Name of the component for labeling the diagram.
        hot_in (int): Index of the hot inlet in the DataFrame.
        hot_out (int): Index of the hot outlet in the DataFrame.
        cold_in (int): Index of the cold inlet in the DataFrame.
        cold_out (int): Index of the cold outlet in the DataFrame.
        delta_t_min (float): Minimum allowable temperature difference.
        system (str): Name of the thermal system for labeling the diagram.
        case (str): Identifier for the specific simulation case.
        plot (bool, optional): Whether to display the generated diagram. Default is False.
        path (str, optional): Filepath to save the diagram. Default is None.
        step_number (int, optional): Number of steps for interpolation. Default is 200.

    Returns:
        list: A list containing the minimum and maximum temperature differences in the diagram.

    Raises:
        None

    Note:
        The function generates and optionally plots a QT diagram for a specified component in a thermal system.
        If the minimum temperature difference is below the allowed minimum, a warning is printed.
    """

    fluid_hot = df.loc[hot_in, 'fluid']
    fluid_cold = df.loc[cold_in, 'fluid']
    T_hot_out = df.loc[hot_out, 'T [°C]']
    T_cold_in = df.loc[cold_in, 'T [°C]']
    h_hot_in = df.loc[hot_in, 'h [kJ/kg]']
    h_hot_out = df.loc[hot_out, 'h [kJ/kg]']
    h_cold_in = df.loc[cold_in, 'h [kJ/kg]']
    h_cold_out = df.loc[cold_out, 'h [kJ/kg]']
    p_hot_in = df.loc[hot_in, 'p [bar]']
    p_hot_out = df.loc[hot_out, 'p [bar]']
    p_cold_in = df.loc[cold_in, 'p [bar]']
    p_cold_out = df.loc[cold_out, 'p [bar]']
    m = df.loc[cold_in, 'm [kg/s]']

    T_cold = [T_cold_in]
    T_hot = [T_hot_out]
    H_plot = [0]

    for i in np.linspace(1, step_number, step_number):
        h_hot = h_hot_out + (h_hot_in - h_hot_out) / step_number * i
        p_hot = p_hot_out + (p_hot_in - p_hot_out) / step_number * i
        T_hot.append(PSI('T', 'H', h_hot * 1e3, 'P', p_hot * 1e5, fluid_hot) - 273.15)
        h_cold = h_cold_in + (h_cold_out - h_cold_in) / step_number * i
        p_cold = p_cold_in - (p_cold_in - p_cold_out) / step_number * i
        T_cold.append(PSI('T', 'H', h_cold * 1e3, 'P', p_cold * 1e5, fluid_cold) - 273.15)
        H_plot.append((h_cold - h_cold_in) * m)

    difference = [x - y for x, y in zip(T_hot, T_cold)]

    if path is not None or plot:  # plot the results
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
        plt.title("QT diagram of the " + component_name + f" of the {system} \nfor the case: {case}")

    if path is not None:
        plt.savefig(path)
    if plot:
        plt.show()

    if min(difference) < delta_t_min - 1e-2:
        print(
            "The min. temperature difference of the " + component_name + f" of the {system} for the case {case} is " + str(
                round(min(difference), 2)) + "K and is lower than the allowed minimum (" + str(
                delta_t_min) + "K).")

    return [min(difference), max(difference)]


def pr_func(pr, p_1, p_2):
    return pr * p_1 - p_2


def pr_deriv(pr, p_1, p_2):
    return {
        "p_1": pr,
        "p_2": -1
    }


def eta_s_compressor_func(eta_s, h_1, p_1, h_2, p_2, fluid):
    h_2s = PSI("H", "P", p_2, "S", PSI("S", "H", h_1, "P", p_1, fluid), fluid)
    return (h_2 - h_1) * eta_s - (h_2s - h_1)


def eta_s_compressor_deriv(eta_s, h_1, p_1, h_2, p_2, fluid):
    d = 1e-2
    return {
        "h_1": (eta_s_compressor_func(eta_s, h_1 + d, p_1, h_2, p_2, fluid) - eta_s_compressor_func(eta_s, h_1 - d, p_1, h_2, p_2, fluid)) / (2 * d),
        "h_2": eta_s,
        "p_1": (eta_s_compressor_func(eta_s, h_1, p_1 + d, h_2, p_2, fluid) - eta_s_compressor_func(eta_s, h_1, p_1 - d, h_2, p_2, fluid)) / (2 * d),
        "p_2": (eta_s_compressor_func(eta_s, h_1, p_1, h_2, p_2 + d, fluid) - eta_s_compressor_func(eta_s, h_1, p_1, h_2, p_2 - d, fluid)) / (2 * d),
    }


def eta_s_expander_func(eta_s, h_1, p_1, h_2, p_2, fluid):
    h_2s = PSI("H", "P", p_2, "S", PSI("S", "H", h_1, "P", p_1, fluid), fluid)
    return (h_2s - h_1) * eta_s - (h_2 - h_1)


def eta_s_expander_deriv(eta_s, h_1, p_1, h_2, p_2, fluid):
    d = 1e-2
    return {
        "h_1": - eta_s,
        "h_2": (eta_s_expander_func(eta_s, h_1, p_1, h_2 + d, p_2, fluid) - eta_s_expander_func(eta_s, h_1, p_1, h_2 - d, p_2, fluid)) / (2 * d),
        "p_1": (eta_s_expander_func(eta_s, h_1, p_1 + d, h_2, p_2, fluid) - eta_s_expander_func(eta_s, h_1, p_1 - d, h_2, p_2, fluid)) / (2 * d),
        "p_2": (eta_s_expander_func(eta_s, h_1, p_1, h_2, p_2 + d, fluid) - eta_s_expander_func(eta_s, h_1, p_1, h_2, p_2 - d, fluid)) / (2 * d),
    }


def turbo_func(P, m, h_1, h_2):
    return P + m * (h_1 - h_2)


def turbo_deriv(P, m, h_1, h_2):
    return {
        "P": 1,
        "m": (h_1 - h_2),
        "h_1": m,
        "h_2": -m
    }


def simple_he_func(Q, m, h_1, h_2):
    return Q + m * (h_1 - h_2)


def simple_he_deriv(Q, m, h_1, h_2):
    return {
        "Q": 1,
        "m": (h_1 - h_2),
        "h_1": m,
        "h_2": -m
    }


def he_func(m_hot, h_1, h_2, m_cold, h_3, h_4):
    return m_hot * (h_1 - h_2) + m_cold * (h_3 - h_4)


def he_deriv(m_hot, h_1, h_2, m_cold, h_3, h_4):
    return {
        "m_hot": (h_1 - h_2),
        "m_cold": (h_3 - h_4),
        "h_1": m_hot,
        "h_2": -m_hot,
        "h_3": m_cold,
        "h_4": -m_cold
    }


def he_with_p_func(power, m_hot, h_1, h_2, m_cold, h_3, h_4):
    return m_hot * (h_1 - h_2) + m_cold * (h_3 - h_4) + power


def he_with_p_deriv(power, m_hot, h_1, h_2, m_cold, h_3, h_4):
    return {
        "power": 1,
        "m_hot": (h_1 - h_2),
        "m_cold": (h_3 - h_4),
        "h_1": m_hot,
        "h_2": -m_hot,
        "h_3": m_cold,
        "h_4": -m_cold
    }


def ihx_func(h_1, h_2, h_3, h_4):
    return (h_1 - h_2) + (h_3 - h_4)


def ihx_deriv(h_1, h_2, h_3, h_4):
    return {
        "h_1": 1,
        "h_2": -1,
        "h_3": 1,
        "h_4": -1
    }


def temperature_func(T, h, p, fluid):
    return PSI("T", "H", h, "P", p, fluid) - T


def temperature_deriv(T, h, p, fluid):
    d = 1e-2
    return {
        "h": (PSI("T", "H", h + d, "P", p, fluid) - PSI("T", "H", h - d, "P", p, fluid)) / (2 * d),
        "p": (PSI("T", "H", h, "P", p + d, fluid) - PSI("T", "H", h, "P", p - d, fluid)) / (2 * d)
    }


def same_temperature_func(h_copy, p_copy, fluid_copy, h_paste, p_paste, fluid_paste):
    t_copy = PSI("T", "H", h_copy, "P", p_copy, fluid_copy)
    h_paste_calc = PSI("H", "T", t_copy, "P", p_paste, fluid_paste)
    return h_paste_calc - h_paste


def same_temperature_deriv(h_copy, p_copy, fluid_copy, h_paste, p_paste, fluid_paste):
    d = 1e-2
    return {
        "h_copy": (same_temperature_func(h_copy + d, p_copy, fluid_copy, h_paste, p_paste, fluid_paste) -
                   same_temperature_func(h_copy - d, p_copy, fluid_copy, h_paste, p_paste, fluid_paste)) / (2 * d),
        "p_copy": (same_temperature_func(h_copy, p_copy + d, fluid_copy, h_paste, p_paste, fluid_paste) -
                   same_temperature_func(h_copy, p_copy - d, fluid_copy, h_paste, p_paste, fluid_paste)) / (2 * d),
        "h_paste": (same_temperature_func(h_copy, p_copy, fluid_copy, h_paste + d, p_paste, fluid_paste) -
                   same_temperature_func(h_copy, p_copy, fluid_copy, h_paste - d, p_paste, fluid_paste)) / (2 * d),
        "p_paste": (same_temperature_func(h_copy, p_copy, fluid_copy, h_paste, p_paste + d, fluid_paste) -
                   same_temperature_func(h_copy, p_copy, fluid_copy, h_paste, p_paste - d, fluid_paste)) / (2 * d)

    }


def ttd_temperature_func(ttd, h_copy, p_copy, fluid_copy, h_paste, p_paste, fluid_paste):
    t_copy = PSI("T", "H", h_copy, "P", p_copy, fluid_copy)
    h_paste_calc = PSI("H", "T", t_copy + ttd, "P", p_paste, fluid_paste)
    return h_paste_calc - h_paste


def ttd_temperature_deriv(ttd, h_copy, p_copy, fluid_copy, h_paste, p_paste, fluid_paste):
    d = 1e-2
    return {
        "h_copy": (ttd_temperature_func(ttd, h_copy + d, p_copy, fluid_copy, h_paste, p_paste, fluid_paste) -
                   ttd_temperature_func(ttd, h_copy - d, p_copy, fluid_copy, h_paste, p_paste, fluid_paste)) / (2 * d),
        "p_copy": (ttd_temperature_func(ttd, h_copy, p_copy + d, fluid_copy, h_paste, p_paste, fluid_paste) -
                   ttd_temperature_func(ttd, h_copy, p_copy - d, fluid_copy, h_paste, p_paste, fluid_paste)) / (2 * d),
        "h_paste": (ttd_temperature_func(ttd, h_copy, p_copy, fluid_copy, h_paste + d, p_paste, fluid_paste) -
                   ttd_temperature_func(ttd, h_copy, p_copy, fluid_copy, h_paste - d, p_paste, fluid_paste)) / (2 * d),
        "p_paste": (ttd_temperature_func(ttd, h_copy, p_copy, fluid_copy, h_paste, p_paste + d, fluid_paste) -
                   ttd_temperature_func(ttd, h_copy, p_copy, fluid_copy, h_paste, p_paste - d, fluid_paste)) / (2 * d)

    }


def ttd_func(h, p, ttd, fluid):
    T = PSI("T", "H", h, "P", p, fluid)
    return T + ttd  # use ttd_func(h, p, -ttd, fluid) to subtract the value or


def ttd_deriv(h, p, ttd, fluid):
    d = 1e-2
    return {
        "h": (PSI("T", "H", h + d, "P", p, fluid) - PSI("T", "H", h - d, "P", p, fluid)) / (2 * d),
        "p": (PSI("T", "H", h, "P", p + d, fluid) - PSI("T", "H", h, "P", p - d, fluid)) / (2 * d)
    }


def valve_func(h_1, h_2):
    return h_1 - h_2


def valve_deriv(h_1, h_2):
    return {
        "h_1": 1,
        "h_2": -1
    }


def x_saturation_func(Q, h, p, fluid):
    return PSI("Q", "H", h, "P", p, fluid) - Q


def x_saturation_deriv(Q, h, p, fluid):
    d = 1e-2
    return {
        "h": (PSI("Q", "H", h + d, "P", p, fluid) - PSI("Q", "H", h - d, "P", p, fluid)) / (2 * d),
        "p": (PSI("Q", "H", h, "P", p + d, fluid) - PSI("Q", "H", h, "P", p - d, fluid)) / (2 * d)
    }


def eps_compressor_func(epsilon, h_1, p_1, h_2, p_2, fluid):
    # epsilon = (E2 - E1) / W
    #         = (e2 - e1) / (h2 - h1)
    #         = (h2 - h1 - T0 - (s2 - s1)) / (h2 - h1)
    s_2 = PSI("S", "H", h_2, "P", p_2, fluid)
    s_1 = PSI("S", "H", h_1, "P", p_1, fluid)
    return (h_2 - h_1) * epsilon - (h_2 - h_1 - t0 * (s_2 - s_1))


def eps_compressor_deriv(epsilon, h_1, p_1, h_2, p_2, fluid):
    d = 1e-2
    return {
        "h_1": (eps_compressor_func(epsilon, h_1 + d, p_1, h_2, p_2, fluid) - eps_compressor_func(epsilon, h_1 - d, p_1, h_2, p_2, fluid)) / (2 * d),
        "h_2": (eps_compressor_func(epsilon, h_1, p_1, h_2 + d, p_2, fluid) - eps_compressor_func(epsilon, h_1, p_1, h_2 - d, p_2, fluid)) / (2 * d),
        "p_1": (eps_compressor_func(epsilon, h_1, p_1 + d, h_2, p_2, fluid) - eps_compressor_func(epsilon, h_1, p_1 - d, h_2, p_2, fluid)) / (2 * d),
        "p_2": (eps_compressor_func(epsilon, h_1, p_1, h_2, p_2 + d, fluid) - eps_compressor_func(epsilon, h_1, p_1, h_2, p_2 - d, fluid)) / (2 * d),
    }


def eps_expander_func(epsilon, h_1, p_1, h_2, p_2, fluid):
    # epsilon = W / (E1 - E2)
    #         = (h1 - h2) / (e1 - e2)
    #         = (h1 - h2) / (h1 - h2 - T0 - (s1 - s2))
    s_2 = PSI("S", "H", h_2, "P", p_2, fluid)
    s_1 = PSI("S", "H", h_1, "P", p_1, fluid)
    return (h_1 - h_2 - t0 * (s_1 - s_2)) * epsilon - (h_1 - h_2)


def eps_expander_deriv(epsilon, h_1, p_1, h_2, p_2, fluid):
    d = 1e-2
    return {
        "h_1": (eps_expander_func(epsilon, h_1 + d, p_1, h_2, p_2, fluid) - eps_expander_func(epsilon, h_1 - d, p_1, h_2, p_2, fluid)) / (2 * d),
        "h_2": (eps_expander_func(epsilon, h_1, p_1, h_2 + d, p_2, fluid) - eps_expander_func(epsilon, h_1, p_1, h_2 - d, p_2, fluid)) / (2 * d),
        "p_1": (eps_expander_func(epsilon, h_1, p_1 + d, h_2, p_2, fluid) - eps_expander_func(epsilon, h_1, p_1 - d, h_2, p_2, fluid)) / (2 * d),
        "p_2": (eps_expander_func(epsilon, h_1, p_1, h_2, p_2 + d, fluid) - eps_expander_func(epsilon, h_1, p_1, h_2, p_2 - d, fluid)) / (2 * d),
    }


def eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot,
                     h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold):
    # epsilon = (E_cold_out - E_cold_in) / (E_hot_in - E_hot_out)
    #         = m_cold * (e_cold_out - e_cold_in) / (m_hot * (e_hot_in - e_hot_out))
    #         = m_cold * (h_cold_out - h_cold_in - T0 * (s_cold_out - s_cold_in)) / 
    #           (m_hot * (h_hot_in - h_hot_out - T0 * (s_hot_in - s_hot_out)))
    s_hot_in = PSI("S", "H", h_hot_in, "P", p_hot_in, fluid_hot)
    s_hot_out = PSI("S", "H", h_hot_out, "P", p_hot_out, fluid_hot)
    s_cold_in = PSI("S", "H", h_cold_in, "P", p_cold_in, fluid_cold)
    s_cold_out = PSI("S", "H", h_cold_out, "P", p_cold_out, fluid_cold)
    return (m_hot * (h_hot_in - h_hot_out - t0 * (s_hot_in - s_hot_out)) * epsilon
            - m_cold * (h_cold_out - h_cold_in - t0 * (s_cold_out - s_cold_in)))


def eps_real_he_deriv(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot,
                      h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold):
    d_high = 1e-2
    d_low = 1e-3
    return {
        "h_hot_in": (eps_real_he_func(epsilon, h_hot_in + d_high, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                     - eps_real_he_func(epsilon, h_hot_in - d_high, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "h_hot_out": (eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out + d_high, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                      - eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out - d_high, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "p_hot_in": (eps_real_he_func(epsilon, h_hot_in, p_hot_in + d_high, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                     - eps_real_he_func(epsilon, h_hot_in, p_hot_in - d_high, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "p_hot_out": (eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out + d_high, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                      - eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out - d_high, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "h_cold_in": (eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in + d_high, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                      - eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in - d_high, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "h_cold_out": (eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out + d_high, p_cold_out, m_cold, fluid_cold)
                       - eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out - d_high, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "p_cold_in": (eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in + d_high, h_cold_out, p_cold_out, m_cold, fluid_cold)
                      - eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in - d_high, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "p_cold_out": (eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out + d_high, m_cold, fluid_cold)
                       - eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out - d_high, m_cold, fluid_cold)) / (2 * d_high),
        "m_hot": (eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot + d_low, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                  - eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot - d_low, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_low),
        "m_cold": (eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold + d_low, fluid_cold)
                   - eps_real_he_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold - d_low, fluid_cold)) / (2 * d_low)
    }


def eps_ideal_he_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot,
                      h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold):
    # 1 = (E_cold_out - E_cold_in) / (E_hot_in - E_hot_out)
    #   = m_cold * (e_cold_out - e_cold_in) / (m_hot * (e_hot_in - e_hot_out))
    #   = m_cold * (h_cold_out - h_cold_in - T0 * (s_cold_out - s_cold_in)) /
    #     (m_hot * (h_hot_in - h_hot_out - T0 * (s_hot_in - s_hot_out)))
    s_hot_in = PSI("S", "H", h_hot_in, "P", p_hot_in, fluid_hot)
    s_hot_out = PSI("S", "H", h_hot_out, "P", p_hot_out, fluid_hot)
    s_cold_in = PSI("S", "H", h_cold_in, "P", p_cold_in, fluid_cold)
    s_cold_out = PSI("S", "H", h_cold_out, "P", p_cold_out, fluid_cold)
    return (m_hot * (h_hot_in - h_hot_out - t0 * (s_hot_in - s_hot_out))
            - m_cold * (h_cold_out - h_cold_in - t0 * (s_cold_out - s_cold_in)))


def eps_ideal_he_deriv(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot,
                      h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold):
    d_high = 1e-2
    d_low = 1e-3
    return {
        "h_hot_in": (eps_real_he_func(power, h_hot_in + d_high, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                     - eps_real_he_func(power, h_hot_in - d_high, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "h_hot_out": (eps_real_he_func(power, h_hot_in, p_hot_in, h_hot_out + d_high, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                      - eps_real_he_func(power, h_hot_in, p_hot_in, h_hot_out - d_high, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "p_hot_in": (eps_real_he_func(power, h_hot_in, p_hot_in + d_high, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                     - eps_real_he_func(power, h_hot_in, p_hot_in - d_high, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "p_hot_out": (eps_real_he_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out + d_high, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                      - eps_real_he_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out - d_high, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "h_cold_in": (eps_real_he_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in + d_high, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                      - eps_real_he_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in - d_high, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "h_cold_out": (eps_real_he_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out + d_high, p_cold_out, m_cold, fluid_cold)
                       - eps_real_he_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out - d_high, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "p_cold_in": (eps_real_he_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in + d_high, h_cold_out, p_cold_out, m_cold, fluid_cold)
                      - eps_real_he_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in - d_high, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "p_cold_out": (eps_real_he_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out + d_high, m_cold, fluid_cold)
                       - eps_real_he_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out - d_high, m_cold, fluid_cold)) / (2 * d_high),
        "m_hot": (eps_real_he_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot + d_low, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                  - eps_real_he_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot - d_low, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_low),
        "m_cold": (eps_real_he_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold + d_low, fluid_cold)
                   - eps_real_he_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold - d_low, fluid_cold)) / (2 * d_low)
    }


def eps_real_ihx_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot,
                      h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold):
    # epsilon = (E_cold_out - E_cold_in) / (E_hot_in - E_hot_out)
    #         = (e_cold_out - e_cold_in) / (e_hot_in - e_hot_out)
    #         = (h_cold_out - h_cold_in - T0 * (s_cold_out - s_cold_in)) / 
    #           (h_hot_in - h_hot_out - T0 * (s_hot_in - s_hot_out))
    s_hot_in = PSI("S", "H", h_hot_in, "P", p_hot_in, fluid_hot)
    s_hot_out = PSI("S", "H", h_hot_out, "P", p_hot_out, fluid_hot)
    s_cold_in = PSI("S", "H", h_cold_in, "P", p_cold_in, fluid_cold)
    s_cold_out = PSI("S", "H", h_cold_out, "P", p_cold_out, fluid_cold)
    return ((h_hot_in - h_hot_out - t0 * (s_hot_in - s_hot_out)) * epsilon
            - (h_cold_out - h_cold_in - t0 * (s_cold_out - s_cold_in)))


def eps_real_ihx_deriv(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot,
                       h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold):
    d = 1e-2
    return {
        "h_hot_in": (eps_real_ihx_func(epsilon, h_hot_in + d, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold)
                     - eps_real_ihx_func(epsilon, h_hot_in - d, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold)) / (2 * d),
        "h_hot_out": (eps_real_ihx_func(epsilon, h_hot_in, p_hot_in, h_hot_out + d, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold)
                      - eps_real_ihx_func(epsilon, h_hot_in, p_hot_in, h_hot_out - d, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold)) / (2 * d),
        "p_hot_in": (eps_real_ihx_func(epsilon, h_hot_in, p_hot_in + d, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold)
                     - eps_real_ihx_func(epsilon, h_hot_in, p_hot_in - d, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold)) / (2 * d),
        "p_hot_out": (eps_real_ihx_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out + d, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold)
                      - eps_real_ihx_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out - d, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold)) / (2 * d),
        "h_cold_in": (eps_real_ihx_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in + d, p_cold_in, h_cold_out, p_cold_out, fluid_cold)
                      - eps_real_ihx_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in - d, p_cold_in, h_cold_out, p_cold_out, fluid_cold)) / (2 * d),
        "h_cold_out": (eps_real_ihx_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out + d, p_cold_out, fluid_cold)
                       - eps_real_ihx_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out - d, p_cold_out, fluid_cold)) / (2 * d),
        "p_cold_in": (eps_real_ihx_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in + d, h_cold_out, p_cold_out, fluid_cold)
                      - eps_real_ihx_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in - d, h_cold_out, p_cold_out, fluid_cold)) / (2 * d),
        "p_cold_out": (eps_real_ihx_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out + d, fluid_cold)
                       - eps_real_ihx_func(epsilon, h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out - d, fluid_cold)) / (2 * d),
    }


def eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot,
                       h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold):
    # 1 = ((E_cold_out - E_cold_in) + power) / (E_hot_in - E_hot_out)
    #   = (m_cold * (h_cold_out - h_cold_in - T0 * (s_cold_out - s_cold_in)) + power) / 
    #     (m_hot * (h_hot_in - h_hot_out - T0 * (s_hot_in - s_hot_out)))
    s_hot_in = PSI("S", "H", h_hot_in, "P", p_hot_in, fluid_hot)
    s_hot_out = PSI("S", "H", h_hot_out, "P", p_hot_out, fluid_hot)
    s_cold_in = PSI("S", "H", h_cold_in, "P", p_cold_in, fluid_cold)
    s_cold_out = PSI("S", "H", h_cold_out, "P", p_cold_out, fluid_cold)
    return (m_hot * (h_hot_in - h_hot_out - t0 * (s_hot_in - s_hot_out))
            - (m_cold * (h_cold_out - h_cold_in - t0 * (s_cold_out - s_cold_in)) + power))


def eps_ideal_ihx_deriv(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot,
                        h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold):
    d_high = 1e-2
    d_low = 1e-3
    return {
        "power": (eps_ideal_ihx_func(power + d_high, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                  - eps_ideal_ihx_func(power - d_high, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)),
        "h_hot_in": (eps_ideal_ihx_func(power, h_hot_in + d_high, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                     - eps_ideal_ihx_func(power, h_hot_in - d_high, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "h_hot_out": (eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out + d_high, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                      - eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out - d_high, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "p_hot_in": (eps_ideal_ihx_func(power, h_hot_in, p_hot_in + d_high, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                     - eps_ideal_ihx_func(power, h_hot_in, p_hot_in - d_high, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "p_hot_out": (eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out + d_high, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                      - eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out - d_high, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "h_cold_in": (eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in + d_high, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                      - eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in - d_high, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "h_cold_out": (eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out + d_high, p_cold_out, m_cold, fluid_cold)
                       - eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out - d_high, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "p_cold_in": (eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in + d_high, h_cold_out, p_cold_out, m_cold, fluid_cold)
                      - eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in - d_high, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_high),
        "p_cold_out": (eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out + d_high, m_cold, fluid_cold)
                       - eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out - d_high, m_cold, fluid_cold)) / (2 * d_high),
        "m_hot": (eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot + d_low, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                  - eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot - d_low, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d_low),
        "m_cold": (eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold + d_low, fluid_cold)
                   - eps_ideal_ihx_func(power, h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold - d_low, fluid_cold)) / (2 * d_low)
    }


def ideal_ihx_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot,
                           h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold):
    # 0 = (S_cold_out - S_cold_in) / (S_hot_in - S_hot_out)
    #   = (m_cold * (s_cold_out - s_cold_in)) / (m_hot * (s_hot_in - s_hot_out))
    #   = (s_cold_out - s_cold_in) / (s_hot_in - s_hot_out)
    s_hot_in = PSI("S", "H", h_hot_in, "P", p_hot_in, fluid_hot)
    s_hot_out = PSI("S", "H", h_hot_out, "P", p_hot_out, fluid_hot)
    s_cold_in = PSI("S", "H", h_cold_in, "P", p_cold_in, fluid_cold)
    s_cold_out = PSI("S", "H", h_cold_out, "P", p_cold_out, fluid_cold)
    return (s_hot_in - s_hot_out) - (s_cold_out - s_cold_in)


def ideal_ihx_entropy_deriv(h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot,
                           h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold):
    d = 1e-2
    return {
        "h_hot_in": (ideal_ihx_entropy_func(h_hot_in + d, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold)
                     - ideal_ihx_entropy_func(h_hot_in - d, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold)) / (2 * d),
        "h_hot_out": (ideal_ihx_entropy_func(h_hot_in, p_hot_in, h_hot_out + d, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold)
                      - ideal_ihx_entropy_func(h_hot_in, p_hot_in, h_hot_out - d, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold)) / (2 * d),
        "p_hot_in": (ideal_ihx_entropy_func(h_hot_in, p_hot_in + d, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold)
                     - ideal_ihx_entropy_func(h_hot_in, p_hot_in - d, h_hot_out, p_hot_out, fluid_hot,h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold)) / (2 * d),
        "p_hot_out": (ideal_ihx_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out + d, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold)
                      - ideal_ihx_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out - d, fluid_hot,h_cold_in, p_cold_in, h_cold_out, p_cold_out, fluid_cold)) / (2 * d),
        "h_cold_in": (ideal_ihx_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in + d, p_cold_in, h_cold_out, p_cold_out, fluid_cold)
                      - ideal_ihx_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in - d, p_cold_in, h_cold_out, p_cold_out, fluid_cold)) / (2 * d),
        "h_cold_out": (ideal_ihx_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out + d, p_cold_out, fluid_cold)
                       - ideal_ihx_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out - d, p_cold_out, fluid_cold)) / (2 * d),
        "p_cold_in": (ideal_ihx_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in + d, h_cold_out, p_cold_out, fluid_cold)
                      - ideal_ihx_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in - d, h_cold_out, p_cold_out, fluid_cold)) / (2 * d),
        "p_cold_out": (ideal_ihx_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out + d, fluid_cold)
                       - ideal_ihx_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out - d, fluid_cold)) / (2 * d)
    }


def ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot,
                          h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold):
    # 0 = (S_cold_out - S_cold_in) / (S_hot_in - S_hot_out)
    #   = (m_cold * (s_cold_out - s_cold_in)) / (m_hot * (s_hot_in - s_hot_out))
    s_hot_in = PSI("S", "H", h_hot_in, "P", p_hot_in, fluid_hot)
    s_hot_out = PSI("S", "H", h_hot_out, "P", p_hot_out, fluid_hot)
    s_cold_in = PSI("S", "H", h_cold_in, "P", p_cold_in, fluid_cold)
    s_cold_out = PSI("S", "H", h_cold_out, "P", p_cold_out, fluid_cold)
    return m_hot * (s_hot_in - s_hot_out) + m_cold * (s_cold_in - s_cold_out)


def ideal_he_entropy_deriv(h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot,
                           h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold):
    d = 1e-2
    return {
        "h_hot_in": (ideal_he_entropy_func(h_hot_in + d, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                     - ideal_he_entropy_func(h_hot_in - d, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d),
        "h_hot_out": (ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out + d, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                      - ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out - d, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d),
        "p_hot_in": (ideal_he_entropy_func(h_hot_in, p_hot_in + d, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                     - ideal_he_entropy_func(h_hot_in, p_hot_in - d, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d),
        "p_hot_out": (ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out + d, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                      - ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out - d, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d),
        "h_cold_in": (ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in + d, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                      - ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in - d, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d),
        "h_cold_out": (ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out + d, p_cold_out, m_cold, fluid_cold)
                       - ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out - d, p_cold_out, m_cold, fluid_cold)) / (2 * d),
        "p_cold_in": (ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in + d, h_cold_out, p_cold_out, m_cold, fluid_cold)
                      - ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in - d, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d),
        "p_cold_out": (ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out + d, m_cold, fluid_cold)
                       - ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out - d, m_cold, fluid_cold)) / (2 * d),
        "m_hot": (ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot + d, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)
                  - ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot - d, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold, fluid_cold)) / (2 * d),
        "m_cold": (ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold + d, fluid_cold)
                   - ideal_he_entropy_func(h_hot_in, p_hot_in, h_hot_out, p_hot_out, m_hot, fluid_hot, h_cold_in, p_cold_in, h_cold_out, p_cold_out, m_cold - d, fluid_cold)) / (2 * d)
    }


def ideal_valve_entropy_func(h_1, p_1, h_2, p_2, fluid):
    s_1 = PSI("S", "H", h_1, "P", p_1, fluid)
    s_2 = PSI("S", "H", h_2, "P", p_2, fluid)
    return s_1 - s_2


def ideal_valve_entropy_deriv(h_1, p_1, h_2, p_2, fluid):
    d = 1e-2
    return {
        "h_1": (ideal_valve_entropy_func(h_1 + d, p_1, h_2, p_2, fluid) - ideal_valve_entropy_func(h_1 - d, p_1, h_2, p_2, fluid)) / (2 * d),
        "h_2": (ideal_valve_entropy_func(h_1, p_1, h_2 + d, p_2, fluid) - ideal_valve_entropy_func(h_1, p_1, h_2 - d, p_2, fluid)) / (2 * d),
        "p_1": (ideal_valve_entropy_func(h_1, p_1 + d, h_2, p_2, fluid) - ideal_valve_entropy_func(h_1, p_1 - d, h_2, p_2, fluid)) / (2 * d),
        "p_2": (ideal_valve_entropy_func(h_1, p_1, h_2, p_2 + d, fluid) - ideal_valve_entropy_func(h_1, p_1, h_2, p_2 - d, fluid)) / (2 * d)
    }

