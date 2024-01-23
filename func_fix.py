import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI as PSI

T_0 = 283.15

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


def calc_vapor_content(p, h, fluid):
    """
    Calculate the vapor content of a given state in a two-phase region.

    Args:
        p (float): Pressure in bar.
        h (float): Enthalpy in kJ/kg.
        fluid (str): Fluid type for property calculations.

    Returns:
        float or None: Vapor content in the two-phase region. Returns None if the state is not in the wet region.

    Note:
        This function calculates the vapor content based on the enthalpy, pressure, and fluid type.
        It checks if the state is in the wet region and returns the vapor content accordingly.
    """

    # Get the saturation properties
    h_sat_liquid = PSI('H', 'P', p * 1e5, 'Q', 0, fluid) * 1e-3
    h_sat_vapor = PSI('H', 'P', p * 1e5, 'Q', 1, fluid) * 1e-3

    # Check if the state is in the wet region
    if h_sat_liquid < h < h_sat_vapor:
        vapor_content = (h - h_sat_liquid) / (h_sat_vapor - h_sat_liquid)
        return vapor_content
    else:
        return None


# --- 5. FUNCTIONS FOR THE COMPONENTS ----------------------------------------------------------------------------------
def pump(df, inlet, outlet, p_out, eta_s, ideal):
    """
    Simulate the operation of a pump in a thermodynamic system with known inlet conditions.

    Args:
        df (pd.DataFrame): DataFrame containing the thermodynamic states.
        inlet (int): Index of the pump's inlet state in the DataFrame.
        outlet (int): Index of the pump's outlet state in the DataFrame.
        p_out (float): Outlet pressure in bar.
        eta_s (float): Isentropic efficiency of the pump.
        ideal (bool): If True, assume the pump operates ideally without losses.

    Returns:
        None

    Note:
        This function updates the properties of the pump's outlet state in the DataFrame based on the pump model.
        If 'ideal' is True, the pump operates isentropically; otherwise, it considers the isentropic efficiency.
    """

    h_in = df.loc[inlet, 'h [kJ/kg]']
    s_in = df.loc[inlet, 's [J/kgK]']

    fluid = df.loc[inlet, 'fluid']

    h_out_s = PSI('H', 'P', p_out * 1e5, 'S', s_in, fluid) * 1e-3

    if ideal:
        h_out = h_out_s
    else:
        h_out = (h_out_s - h_in) / eta_s + h_in
    T_out = PSI('T', 'H', h_out * 1e3, 'P', p_out * 1e5, fluid) - 273.15
    s_out = PSI('S', 'H', h_out * 1e3, 'P', p_out * 1e5, fluid)

    df.loc[outlet, 'T [°C]'] = T_out
    df.loc[outlet, 'p [bar]'] = p_out
    df.loc[outlet, 'h [kJ/kg]'] = h_out
    df.loc[outlet, 's [J/kgK]'] = s_out


def turbine(df, inlet, outlet, p_out, eta_s, ideal):
    """
    Simulate the operation of a turbine in a thermodynamic system with known inlet conditions.

    Args:
        df (pd.DataFrame): DataFrame containing the thermodynamic states.
        inlet (int): Index of the turbine's inlet state in the DataFrame.
        outlet (int): Index of the turbine's outlet state in the DataFrame.
        p_out (float): Outlet pressure in bar.
        eta_s (float): Isentropic efficiency of the turbine.
        ideal (bool): If True, assume the turbine operates ideally without losses.

    Returns:
        None

    Note:
        This function updates the properties of the turbine's outlet state in the DataFrame based on the turbine model.
        If 'ideal' is True, the turbine operates isentropically; otherwise, it considers the isentropic efficiency.
    """

    h_in = df.loc[inlet, 'h [kJ/kg]']
    s_in = df.loc[inlet, 's [J/kgK]']

    fluid = df.loc[inlet, 'fluid']

    h_out_s = PSI('H', 'P', p_out * 1e5, 'S', s_in, fluid) * 1e-3

    if ideal:
        h_out = h_out_s
    else:
        h_out = (h_out_s - h_in) * eta_s + h_in

    T_out = PSI('T', 'H', h_out * 1e3, 'P', p_out * 1e5, fluid) - 273.15
    s_out = PSI('S', 'H', h_out * 1e3, 'P', p_out * 1e5, fluid)

    df.loc[outlet, 'T [°C]'] = T_out
    df.loc[outlet, 'p [bar]'] = p_out
    df.loc[outlet, 'h [kJ/kg]'] = h_out
    df.loc[outlet, 's [J/kgK]'] = s_out


def valve(df, inlet, outlet, p_out, ideal=False):
    """
    Simulate the operation of a valve in a thermodynamic system with known inlet conditions.

    Args:
        df (pd.DataFrame): DataFrame containing the thermodynamic states.
        inlet (int): Index of the valve's inlet state in the DataFrame.
        outlet (int): Index of the valve's outlet state in the DataFrame.
        p_out (float): Outlet pressure in bar.
        ideal (bool): If True, assume the valve operates isentropically without losses.

    Returns:
        None

    Note:
        This function updates the properties of the valve's outlet state in the DataFrame based on the valve model.
        If 'ideal' is True, the valve operates isentropically; otherwise, it assumes no changes in enthalpy.
    """

    h_in = df.loc[inlet, 'h [kJ/kg]']
    s_in = df.loc[inlet, 's [J/kgK]']

    fluid = df.loc[inlet, 'fluid']

    if ideal:
        s_out = s_in
        h_out = PSI('H', 'P', p_out * 1e5, 'S', s_out, fluid) * 1e-3
        T_out = PSI('T', 'H', h_out * 1e3, 'P', p_out * 1e5, fluid) - 273.15

    else:
        h_out = h_in
        s_out = PSI('S', 'P', p_out * 1e5, 'H', h_out * 1e3, fluid)
        T_out = PSI('T', 'H', h_out * 1e3, 'P', p_out * 1e5, fluid) - 273.15

    df.loc[outlet, 'T [°C]'] = T_out
    df.loc[outlet, 'p [bar]'] = p_out
    df.loc[outlet, 'h [kJ/kg]'] = h_out
    df.loc[outlet, 's [J/kgK]'] = s_out


def heat_exchanger(df, inlet_h, outlet_h, inlet_c, outlet_c, pr_hot, pr_cold, ttd_u=None, ttd_l=None, ideal=False):
    """
    Simulate the operation of a heat exchanger in a thermodynamic system.

    Args:
        df (pd.DataFrame): DataFrame containing relevant thermodynamic properties.
        inlet_h (int): Index of the hot fluid inlet state in the DataFrame.
        outlet_h (int): Index of the hot fluid outlet state in the DataFrame.
        inlet_c (int): Index of the cold fluid inlet state in the DataFrame.
        outlet_c (int): Index of the cold fluid outlet state in the DataFrame.
        pr_hot (float): Pressure ratio of the hot fluid.
        pr_cold (float): Pressure ratio of the cold fluid.
        ttd_u (float, optional): Temperature difference in °C for the upper side (hot side). Default is None.
        ttd_l (float, optional): Temperature difference in °C for the lower side (cold side). Default is None.
        ideal (bool, optional): If True, assume ideal behavior with no heat loss or gain. Default is False.

    Returns:
        None

    Note:
        This function updates the properties of the heat exchanger's outlet states in the DataFrame.
        The function considers various scenarios for known and unknown parameters, including temperature differences and pressure ratios.
    """

    # Extracting properties of the hot inlet state
    h_hot_in = df.loc[inlet_h, 'h [kJ/kg]']
    s_hot_in = df.loc[inlet_h, 's [J/kgK]']
    T_hot_in = df.loc[inlet_h, 'T [°C]']
    p_hot_in = df.loc[inlet_h, 'p [bar]']

    # Extracting properties of the hot outlet state
    h_hot_out = df.loc[outlet_h, 'h [kJ/kg]']
    s_hot_out = df.loc[outlet_h, 's [J/kgK]']
    T_hot_out = df.loc[outlet_h, 'T [°C]']
    p_hot_out = df.loc[outlet_h, 'p [bar]']

    # Extracting properties of the cold inlet state
    h_cold_in = df.loc[inlet_c, 'h [kJ/kg]']
    s_cold_in = df.loc[inlet_c, 's [J/kgK]']
    T_cold_in = df.loc[inlet_c, 'T [°C]']
    p_cold_in = df.loc[inlet_c, 'p [bar]']

    # Extracting properties of the cold outlet state
    h_cold_out = df.loc[outlet_c, 'h [kJ/kg]']
    s_cold_out = df.loc[outlet_c, 's [J/kgK]']
    T_cold_out = df.loc[outlet_c, 'T [°C]']
    p_cold_out = df.loc[outlet_c, 'p [bar]']

    # Extracting mass flows and fluid types
    m_hot = df.loc[inlet_h, 'm [kg/s]']
    fluid_hot = df.loc[inlet_h, 'fluid']
    m_cold = df.loc[inlet_c, 'm [kg/s]']
    fluid_cold = df.loc[inlet_c, 'fluid']

    # Creating a decision vector to identify known and unknown parameters
    dec_vect = [1, 1, 1, 1, 1, 1, 1, 1]
    dec_vect.append(ideal)

    # Identifying known and unknown parameters based on the given inputs
    vars = [m_hot, m_cold, h_hot_in, h_hot_out, h_cold_in, h_cold_out, ttd_u, ttd_l]
    for i in range(len(vars)):
        if vars[i] is np.nan or vars[i] is None:
            dec_vect[i] = 0

    # (1) both mass flows are known and only one thermodynamic state is unknown
    #     --> calc missing state with energy balance equation
    # (2) both mass flows are known and two thermodynamic states are unknown (one of each side)
    #     --> ttd_l or ttd_u is needed to calc one of them
    #     --> the other state is calculated with energy balance equation
    # (3) no mass flow or only one mass flow is known and two thermodynamic states are unknown
    #     --> ttd_l or ttd_u is needed to calc one of the missing state
    #     --> the other mass flow and one state are unsolved if possible
    # (4) all thermodynamic states are known, one mass flow is unknown
    #     --> the energy balance equations is solved to calculate the other mass flow
    # TODO same case/scenarios may be missing

    # (1) both mass flows are known and only one thermodynamic state is unknown
    if dec_vect == [1, 1, 1, 1, 1, 0, 0, 0, False]:  # both mass flows are known, only cold_out is unknown, real
        h_cold_out = h_cold_in + m_hot * (h_hot_in - h_hot_out) / m_cold
        p_cold_out = p_cold_in * pr_cold
        T_cold_out = PSI('T', 'H', h_cold_out * 1e3, 'P', p_cold_out * 1e5, fluid_cold) - 273.15
        s_cold_out = PSI('S', 'P', p_cold_out * 1e5, 'T', T_cold_out + 273.15, fluid_cold)

    if dec_vect == [1, 1, 1, 1, 1, 0, 0, 0, True]:  # both mass flows are known, only cold_out is unknown, real
        s_cold_out = s_cold_in + m_hot * (s_hot_in - s_hot_out) / m_cold
        p_cold_out = p_cold_in
        T_cold_out = PSI('T', 'S', s_cold_out, 'P', p_cold_out * 1e5, fluid_cold) - 273.15
        h_cold_out = PSI('H', 'P', p_cold_out * 1e5, 'T', T_cold_out + 273.15, fluid_cold) * 1e-3

    if dec_vect == [1, 1, 1, 1, 0, 1, 0, 0, False]:  # both mass flows are known, only hot_out is unknown, real
        h_cold_in = h_cold_out - m_hot * (h_hot_in - h_hot_out) / m_cold
        p_cold_in = p_cold_in * pr_cold
        T_cold_in = PSI('T', 'H', h_cold_in * 1e3, 'P', p_cold_in * 1e5, fluid_cold) - 273.15
        s_cold_in = PSI('S', 'P', p_cold_in * 1e5, 'T', T_cold_in + 273.15, fluid_cold)

    if dec_vect == [1, 1, 1, 1, 0, 1, 0, 0, True]:  # both mass flows are known, only hot_out is unknown, ideal
        s_cold_in = s_cold_out - m_hot * (s_hot_in - s_hot_out) / m_cold
        p_cold_in = p_cold_in
        T_cold_in = PSI('T', 'S', s_cold_in, 'P', p_cold_in * 1e5, fluid_cold) - 273.15
        h_cold_in = PSI('H', 'P', p_cold_in * 1e5, 'T', T_cold_in + 273.15, fluid_cold) * 1e-3

    if dec_vect == [1, 1, 1, 0, 1, 1, 0, 0, False]:  # both mass flows are known, only hot_out is unknown, real
        h_hot_out = h_hot_in + m_cold * (h_cold_in - h_cold_out) / m_hot
        p_hot_out = p_hot_in * pr_hot
        T_hot_out = PSI('T', 'H', h_hot_out * 1e3, 'P', p_hot_out * 1e5, fluid_hot) - 273.15
        s_hot_out = PSI('S', 'P', p_hot_out * 1e5, 'T', T_hot_out + 273.15, fluid_hot)

    if dec_vect == [1, 1, 1, 0, 1, 1, 0, 0, True]:  # both mass flows are known, only hot_out is unknown, ideal
        s_hot_out = s_hot_in + m_cold * (s_cold_in - s_cold_out) / m_hot
        p_hot_out = p_hot_in
        T_hot_out = PSI('T', 'S', s_hot_out, 'P', p_hot_out * 1e5, fluid_hot) - 273.15
        h_hot_out = PSI('H', 'P', p_hot_out * 1e5, 'T', T_hot_out + 273.15, fluid_hot) * 1e-3

    if dec_vect == [1, 1, 0, 1, 1, 1, 0, 0, False]:  # both mass flows are known, only hot_in is unknown, real
        h_hot_in = h_hot_out - m_cold * (h_cold_in - h_cold_out) / m_hot
        p_hot_in = p_hot_in * pr_hot
        T_hot_in = PSI('T', 'H', h_hot_in * 1e3, 'P', p_hot_in * 1e5, fluid_hot) - 273.15
        s_hot_in = PSI('S', 'P', p_hot_in * 1e5, 'T', T_hot_in + 273.15, fluid_hot)

    if dec_vect == [1, 1, 0, 1, 1, 1, 0, 0, True]:  # both mass flows are known, only hot_in is unknown, ideal
        s_hot_in = s_hot_out - m_cold * (s_cold_in - s_cold_out) / m_hot
        p_hot_in = p_hot_in
        T_hot_in = PSI('T', 'S', s_hot_in, 'P', p_hot_in * 1e5, fluid_hot) - 273.15
        h_hot_in = PSI('H', 'P', p_hot_in * 1e5, 'T', T_hot_in + 273.15, fluid_hot) * 1e-3

    # (2) both mass flows are known and two thermodynamic states are unknown (one of each side)
    if dec_vect == [1, 1, 1, 0, 1, 0, 1, 0, False]:  # both mass flows are known, both outlet unknown, ttd_u given, real
        T_cold_out = T_hot_in - ttd_u
        p_cold_out = p_cold_in * pr_cold
        h_cold_out = PSI('H', 'P', p_cold_out * 1e5, 'T', T_cold_out + 273.15, fluid_cold) * 1e-3
        s_cold_out = PSI('S', 'P', p_cold_out * 1e5, 'T', T_cold_out + 273.15, fluid_cold)
        h_hot_out = h_hot_in + m_cold * (h_cold_in - h_cold_out) / m_hot
        p_hot_out = p_hot_in * pr_hot
        T_hot_out = PSI('T', 'H', h_hot_out * 1e3, 'P', p_hot_out * 1e5, fluid_hot) - 273.15
        s_hot_out = PSI('S', 'P', p_hot_out * 1e5, 'T', T_hot_out + 273.15, fluid_hot)

    if dec_vect == [1, 1, 1, 0, 1, 0, 1, 0, True]:  # both mass flows are known, both outlet unknown, ttd_u given, ideal
        T_cold_out = T_hot_in
        p_cold_out = p_cold_in
        s_cold_out = PSI('S', 'P', p_cold_out * 1e5, 'T', T_cold_out + 273.15, fluid_cold)
        h_cold_out = PSI('H', 'P', p_cold_out * 1e5, 'T', T_cold_out + 273.15, fluid_cold) * 1e-3
        s_hot_out = s_hot_in + m_cold * (s_cold_in - s_cold_out) / m_hot
        p_hot_out = p_hot_in
        T_hot_out = PSI('T', 'S', s_hot_out, 'P', p_hot_out * 1e5, fluid_hot) - 273.15
        h_hot_out = PSI('H', 'P', p_hot_out * 1e5, 'T', T_hot_out + 273.15, fluid_hot) * 1e-3

    if dec_vect == [1, 1, 1, 0, 1, 0, 0, 1, False]:  # both mass flows are known, both outlet unknown, ttd_l given, real
        T_hot_out = T_cold_in + ttd_l
        p_hot_out = p_hot_in * pr_hot
        h_hot_out = PSI('H', 'P', p_hot_out * 1e5, 'T', T_hot_out + 273.15, fluid_hot) * 1e-3
        s_hot_out = PSI('S', 'P', p_hot_out * 1e5, 'T', T_hot_out + 273.15, fluid_hot)
        h_cold_out = m_hot * (h_hot_out - h_hot_in) / m_cold + h_cold_in
        p_cold_out = p_cold_in * pr_cold
        T_cold_out = PSI('T', 'H', h_cold_out * 1e3, 'P', p_cold_out * 1e5, fluid_cold) - 273.15
        s_cold_out = PSI('S', 'P', p_cold_out * 1e5, 'T', T_cold_out + 273.15, fluid_cold)

    if dec_vect == [1, 1, 1, 0, 1, 0, 0, 1, True]:  # both mass flows are known, both outlet unknown, ttd_l given, ideal
        T_hot_out = T_cold_in
        p_hot_out = p_hot_in
        s_hot_out = PSI('S', 'P', p_hot_out * 1e5, 'T', T_hot_out + 273.15, fluid_hot)
        h_hot_out = PSI('H', 'P', p_hot_out * 1e5, 'T', T_hot_out + 273.15, fluid_hot) * 1e-3
        s_cold_out = m_hot * (s_hot_out - s_hot_in) / m_cold + s_cold_in
        p_cold_out = p_cold_in * pr_cold
        T_cold_out = PSI('T', 'S', s_cold_out, 'P', p_cold_out * 1e5, fluid_cold) - 273.15
        h_cold_out = PSI('H', 'P', p_cold_out * 1e5, 'T', T_cold_out + 273.15, fluid_cold) * 1e-3

    if dec_vect == [1, 1, 0, 1, 0, 1, 1, 0, False]:  # both mass flows are known, both inlet unknown, ttd_u given, real
        T_hot_in = T_cold_out + ttd_u
        p_hot_in = p_hot_out / pr_hot
        h_hot_in = PSI('H', 'P', p_hot_in * 1e5, 'T', T_hot_in + 273.15, fluid_hot) * 1e-3
        s_hot_in = PSI('S', 'P', p_hot_in * 1e5, 'T', T_hot_in + 273.15, fluid_hot)
        h_cold_in = h_cold_out - m_hot * (h_hot_in - h_hot_out) / m_cold
        p_cold_in = p_cold_out / pr_cold
        T_cold_in = PSI('T', 'H', h_cold_in * 1e3, 'P', p_cold_in * 1e5, fluid_cold) - 273.15
        s_cold_in = PSI('S', 'P', p_cold_in * 1e5, 'T', T_cold_in + 273.15, fluid_cold)

    if dec_vect == [1, 1, 0, 1, 0, 1, 1, 0, True]:  # both mass flows are known, both inlet unknown, ttd_u given, ideal
        T_hot_in = T_cold_out
        p_hot_in = p_hot_out
        h_hot_in = PSI('H', 'P', p_hot_in * 1e5, 'T', T_hot_in + 273.15, fluid_hot) * 1e-3
        s_hot_in = PSI('S', 'P', p_hot_in * 1e5, 'T', T_hot_in + 273.15, fluid_hot)
        s_cold_in = s_cold_out - m_hot * (s_hot_in - s_hot_out) / m_cold
        p_cold_in = p_cold_out
        T_cold_in = PSI('T', 'H', h_cold_in * 1e3, 'P', p_cold_in * 1e5, fluid_cold) - 273.15
        h_cold_in = PSI('H', 'P', p_cold_in * 1e5, 'T', T_cold_in + 273.15, fluid_cold) * 1e-3

    if dec_vect == [1, 1, 0, 1, 0, 1, 0, 1, False]:  # both mass flows are known, both inlet unknown, ttd_u given, real
        T_cold_in = T_hot_out - ttd_l
        p_cold_in = p_cold_out / pr_cold
        h_cold_in = PSI('H', 'P', p_cold_in * 1e5, 'T', T_cold_in + 273.15, fluid_cold) * 1e-3
        s_cold_in = PSI('S', 'P', p_cold_in * 1e5, 'T', T_cold_in + 273.15, fluid_cold)
        h_hot_in = h_cold_in - m_cold * (h_cold_in - h_cold_out) / m_hot
        p_hot_in = p_hot_out / pr_hot
        T_hot_in = PSI('T', 'H', h_hot_in * 1e3, 'P', p_hot_in * 1e5, fluid_hot) - 273.15
        s_hot_in = PSI('S', 'P', p_hot_in * 1e5, 'T', T_hot_in + 273.15, fluid_hot)

    if dec_vect == [1, 1, 0, 1, 0, 1, 0, 1, True]:  # both mass flows are known, both inlet unknown, ttd_u given, ideal
        T_cold_in = T_hot_out
        p_cold_in = p_cold_out
        h_cold_in = PSI('H', 'P', p_cold_in * 1e5, 'T', T_cold_in + 273.15, fluid_cold) * 1e-3
        s_cold_in = PSI('S', 'P', p_cold_in * 1e5, 'T', T_cold_in + 273.15, fluid_cold)
        s_hot_in = s_cold_in - m_cold * (s_cold_in - s_cold_out) / m_hot
        p_hot_in = p_hot_out
        T_hot_in = PSI('T', 'H', h_hot_in * 1e3, 'P', p_hot_in * 1e5, fluid_hot) - 273.15
        h_hot_in = PSI('H', 'P', p_hot_in * 1e5, 'T', T_hot_in + 273.15, fluid_hot) * 1e-3

    # (3) no mass flow or only one mass flow is known and two thermodynamic states are unknown
    if dec_vect == [1, 0, 1, 0, 1, 0, 1, 0, False] \
            or dec_vect == [0, 1, 1, 0, 1, 0, 1, 0, False] \
            or dec_vect == [0, 0, 1, 0, 1, 0, 1, 0,
                            False]:  # only one mass flow is known, both outlet unknown, ttd_u given, real
        T_cold_out = T_hot_in - ttd_u
        p_cold_out = p_cold_in * pr_cold
        h_cold_out = PSI('H', 'P', p_cold_out * 1e5, 'T', T_cold_out + 273.15, fluid_cold) * 1e-3
        s_cold_out = PSI('S', 'P', p_cold_out * 1e5, 'T', T_cold_out + 273.15, fluid_cold)

    if dec_vect == [1, 0, 1, 0, 1, 0, 1, 0, True] \
            or dec_vect == [0, 1, 1, 0, 1, 0, 1, 0, True] \
            or dec_vect == [0, 0, 1, 0, 1, 0, 1, 0,
                            True]:  # only one mass flow is known, both outlet unknown, ttd_u given, ideal
        T_cold_out = T_hot_in
        p_cold_out = p_cold_in
        s_cold_out = PSI('S', 'P', p_cold_out * 1e5, 'T', T_cold_out + 273.15, fluid_cold)
        h_cold_out = PSI('H', 'P', p_cold_out * 1e5, 'T', T_cold_out + 273.15, fluid_cold) * 1e-3

    if dec_vect == [1, 0, 1, 0, 1, 0, 0, 1, False] \
            or dec_vect == [0, 1, 1, 0, 1, 0, 0, 1, False] \
            or dec_vect == [0, 0, 1, 0, 1, 0, 0, 1,
                            False]:  # only one flow is known, both outlet unknown, ttd_l given, real
        T_hot_out = T_cold_in + ttd_l
        p_hot_out = p_hot_in * pr_hot
        h_hot_out = PSI('H', 'P', p_hot_out * 1e5, 'T', T_hot_out + 273.15, fluid_hot) * 1e-3
        s_hot_out = PSI('S', 'P', p_hot_out * 1e5, 'T', T_hot_out + 273.15, fluid_hot)

    if dec_vect == [1, 0, 1, 0, 1, 0, 0, 1, True] \
            or dec_vect == [0, 1, 1, 0, 1, 0, 0, 1, True] \
            or dec_vect == [0, 0, 1, 0, 1, 0, 0, 1,
                            True]:  # only one flow is known, both outlet unknown, ttd_l given, ideal
        T_hot_out = T_cold_in
        p_hot_out = p_hot_in
        h_hot_out = PSI('H', 'P', p_hot_out * 1e5, 'T', T_hot_out + 273.15, fluid_hot) * 1e-3
        s_hot_out = PSI('S', 'P', p_hot_out * 1e5, 'T', T_hot_out + 273.15, fluid_hot)

    if dec_vect == [1, 0, 0, 1, 0, 1, 1, 0, False] \
            or dec_vect == [0, 1, 0, 1, 0, 1, 1, 0, False] \
            or dec_vect == [0, 0, 0, 1, 0, 1, 1, 0,
                            False]:  # only one mass flow is known, both inlet unknown, ttd_u given, real
        T_hot_in = T_cold_out + ttd_u
        p_hot_in = p_hot_out / pr_hot
        h_hot_in = PSI('H', 'P', p_hot_in * 1e5, 'T', T_hot_in + 273.15, fluid_hot) * 1e-3
        s_hot_in = PSI('S', 'P', p_hot_in * 1e5, 'T', T_hot_in + 273.15, fluid_hot)

    if dec_vect == [1, 0, 0, 1, 0, 1, 1, 0, True] \
            or dec_vect == [0, 1, 0, 1, 0, 1, 1, 0, True] \
            or dec_vect == [0, 0, 0, 1, 0, 1, 1, 0,
                            True]:  # only one mass flow is known, both inlet unknown, ttd_u given, ideal
        T_hot_in = T_cold_out
        p_hot_in = p_hot_out
        h_hot_in = PSI('H', 'P', p_hot_in * 1e5, 'T', T_hot_in + 273.15, fluid_hot) * 1e-3
        s_hot_in = PSI('S', 'P', p_hot_in * 1e5, 'T', T_hot_in + 273.15, fluid_hot)

    if dec_vect == [1, 0, 0, 1, 0, 1, 0, 1, False] \
            or dec_vect == [0, 1, 0, 1, 0, 1, 0, 1, False] \
            or dec_vect == [0, 0, 0, 1, 0, 1, 0, 1,
                            False]:  # only one flow is known, both inlet unknown, ttd_l given, real
        T_cold_in = T_hot_out - ttd_l
        p_cold_in = p_cold_out / pr_cold
        h_cold_in = PSI('H', 'P', p_cold_in * 1e5, 'T', T_cold_in + 273.15, fluid_cold) * 1e-3
        s_cold_in = PSI('S', 'P', p_cold_in * 1e5, 'T', T_cold_in + 273.15, fluid_cold)

    if dec_vect == [1, 0, 0, 1, 0, 1, 0, 1, True]:  # only one flow is known, both inlet unknown, ttd_l given, ideal
        T_cold_in = T_hot_out
        p_cold_in = p_cold_out
        h_cold_in = PSI('H', 'P', p_cold_in * 1e5, 'T', T_cold_in + 273.15, fluid_cold) * 1e-3
        s_cold_in = PSI('S', 'P', p_cold_in * 1e5, 'T', T_cold_in + 273.15, fluid_cold)

    # (4) all thermodynamic states are known, one mass flow is unknown
    if dec_vect == [0, 1, 1, 1, 1, 1, 0, 0, False]:
        m_hot = m_cold * (h_cold_in - h_cold_out) / (h_hot_out - h_hot_in)  # m_hot missing, real

    if dec_vect == [0, 1, 1, 1, 1, 1, 0, 0, True]:
        m_hot = m_cold * (s_cold_in - s_cold_out) / (s_hot_out - s_hot_in)  # m_hot missing, ideal

    if dec_vect == [1, 0, 1, 1, 1, 1, 0, 0, False]:
        m_cold = m_hot * (h_hot_out - h_hot_in) / (h_cold_in - h_cold_out)  # m_cold missing, real

    if dec_vect == [1, 0, 1, 1, 1, 1, 0, 0, True]:
        m_cold = m_hot * (s_hot_out - s_hot_in) / (s_cold_in - s_cold_out)  # m_cold missing, ideal

    # Assign outlet cold fluid properties to DataFrame
    df.loc[outlet_c, 'T [°C]'] = T_cold_out
    df.loc[outlet_c, 'p [bar]'] = p_cold_out
    df.loc[outlet_c, 'h [kJ/kg]'] = h_cold_out
    df.loc[outlet_c, 's [J/kgK]'] = s_cold_out
    df.loc[outlet_c, 'm [kg/s]'] = m_cold

    # Assign outlet hot fluid properties to DataFrame
    df.loc[outlet_h, 'T [°C]'] = T_hot_out
    df.loc[outlet_h, 'p [bar]'] = p_hot_out
    df.loc[outlet_h, 'h [kJ/kg]'] = h_hot_out
    df.loc[outlet_h, 's [J/kgK]'] = s_hot_out
    df.loc[outlet_h, 'm [kg/s]'] = m_hot

    # Assign inlet cold fluid properties to DataFrame
    df.loc[inlet_c, 'T [°C]'] = T_cold_in
    df.loc[inlet_c, 'p [bar]'] = p_cold_in
    df.loc[inlet_c, 'h [kJ/kg]'] = h_cold_in
    df.loc[inlet_c, 's [J/kgK]'] = s_cold_in
    df.loc[inlet_c, 'm [kg/s]'] = m_cold

    # Assign inlet hot fluid properties to DataFrame
    df.loc[inlet_h, 'T [°C]'] = T_hot_in
    df.loc[inlet_h, 'p [bar]'] = p_hot_in
    df.loc[inlet_h, 'h [kJ/kg]'] = h_hot_in
    df.loc[inlet_h, 's [J/kgK]'] = s_hot_in
    df.loc[inlet_h, 'm [kg/s]'] = m_hot


def turbo_mach_bal(df_comps, df_conns, inlet, outlet, label):
    """
    Perform thermodynamic balance equations for a turbo machine and update the specified component in the given DataFrames.

    Parameters:
    - df_comps (pd.DataFrame): DataFrame containing component information.
    - df_conns (pd.DataFrame): DataFrame containing connection information.
    - inlet (str): Label of the inlet connection.
    - outlet (str): Label of the outlet connection.
    - label (str): Label of the component to be updated.

    Updates the specified component with power and entropy generation based on the balance equations.

    Returns:
    None
    """

    df_comps.loc[label, 'P [kW]'] = - df_conns.loc[inlet, 'm [kg/s]'] * (
                df_conns.loc[inlet, 'h [kJ/kg]'] - df_conns.loc[outlet, 'h [kJ/kg]'])
    df_comps.loc[label, 'S_gen [kW/K]'] = - df_conns.loc[inlet, 'm [kg/s]'] * (
                df_conns.loc[inlet, 's [J/kgK]'] - df_conns.loc[outlet, 's [J/kgK]']) * 1e-3


def he_bal(df_comps, df_conns, inlet_h, outlet_h, inlet_c, outlet_c, label):
    """
    Perform thermodynamic balance equations for a heat exchanger and update the specified component in the given DataFrames.

    Parameters:
    - df_comps (pd.DataFrame): DataFrame containing component information.
    - df_conns (pd.DataFrame): DataFrame containing connection information.
    - inlet_h (str): Label of the hot side inlet connection.
    - outlet_h (str): Label of the hot side outlet connection.
    - inlet_c (str): Label of the cold side inlet connection.
    - outlet_c (str): Label of the cold side outlet connection.
    - label (str): Label of the component to be updated.

    Updates the specified component with power, heat transfer, and entropy generation based on the balance equations.

    Returns:
    None
    """

    df_comps.loc[label, 'P [kW]'] = - df_conns.loc[inlet_h, 'm [kg/s]'] * (
                df_conns.loc[inlet_h, 'h [kJ/kg]'] - df_conns.loc[outlet_h, 'h [kJ/kg]']) - df_conns.loc[
                                        inlet_c, 'm [kg/s]'] * (df_conns.loc[inlet_c, 'h [kJ/kg]'] - df_conns.loc[
        outlet_c, 'h [kJ/kg]'])
    df_comps.loc[label, 'Q [kW]'] = df_conns.loc[inlet_c, 'm [kg/s]'] * (
                df_conns.loc[inlet_c, 'h [kJ/kg]'] - df_conns.loc[outlet_c, 'h [kJ/kg]'])
    df_comps.loc[label, 'S_gen [kW/K]'] = - df_conns.loc[inlet_h, 'm [kg/s]'] * (
                df_conns.loc[inlet_h, 's [J/kgK]'] - df_conns.loc[outlet_h, 's [J/kgK]']) * 1e-3 - df_conns.loc[
                                              inlet_c, 'm [kg/s]'] * (df_conns.loc[inlet_c, 's [J/kgK]'] - df_conns.loc[
        outlet_c, 's [J/kgK]']) * 1e-3


def valve_bal(df_comps, df_conns, inlet, outlet, label):
    """
    Perform thermodynamic balance equations for a valve and update the specified component in the given DataFrames.

    Parameters:
    - df_comps (pd.DataFrame): DataFrame containing component information.
    - df_conns (pd.DataFrame): DataFrame containing connection information.
    - inlet (str): Label of the valve's inlet connection.
    - outlet (str): Label of the valve's outlet connection.
    - label (str): Label of the component to be updated.

    Updates the specified component with power and entropy generation based on the balance equations.

    Returns:
    None
    """

    df_comps.loc[label, 'P [kW]'] = - df_conns.loc[inlet, 'm [kg/s]'] * (
                df_conns.loc[inlet, 'h [kJ/kg]'] - df_conns.loc[outlet, 'h [kJ/kg]'])
    df_comps.loc[label, 'S_gen [kW/K]'] = - df_conns.loc[inlet, 'm [kg/s]'] * (
                df_conns.loc[inlet, 's [J/kgK]'] - df_conns.loc[outlet, 's [J/kgK]']) * 1e-3


#
#
#
#
# FUNCTIONS FOR SIMULTANEOUS SOLUTION
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


def ttd_func(h, p, ttd, fluid):
    T = PSI("T", "H", h, "P", p, fluid)
    return T + ttd


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
    return (h_2 - h_1) * epsilon - (h_2 - h_1 - T_0 * (s_2 - s_1))


def eps_compressor_deriv(epsilon, h_1, p_1, h_2, p_2, fluid):
    d = 1e-2
    return {
        "h_1": (eps_compressor_func(epsilon, h_1 + d, p_1, h_2, p_2, fluid) - eps_compressor_func(epsilon, h_1 - d, p_1, h_2, p_2, fluid)) / (2 * d),
        "h_2": (eps_compressor_func(epsilon, h_1, p_1, h_2 + d, p_2, fluid) - eps_compressor_func(epsilon, h_1, p_1, h_2 - d, p_2, fluid)) / (2 * d),
        "p_1": (eps_compressor_func(epsilon, h_1, p_1 + d, h_2, p_2, fluid) - eps_compressor_func(epsilon, h_1, p_1 - d, h_2, p_2, fluid)) / (2 * d),
        "p_2": (eps_compressor_func(epsilon, h_1, p_1, h_2, p_2 + d, fluid) - eps_compressor_func(epsilon, h_1, p_1, h_2, p_2 - d, fluid)) / (2 * d),
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
    return (m_hot * (h_hot_in - h_hot_out - T_0 * (s_hot_in - s_hot_out)) * epsilon
            - m_cold * (h_cold_out - h_cold_in - T_0 * (s_cold_out - s_cold_in)))


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
    return (m_hot * (h_hot_in - h_hot_out - T_0 * (s_hot_in - s_hot_out))
            - m_cold * (h_cold_out - h_cold_in - T_0 * (s_cold_out - s_cold_in)))


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
    return ((h_hot_in - h_hot_out - T_0 * (s_hot_in - s_hot_out)) * epsilon
            - (h_cold_out - h_cold_in - T_0 * (s_cold_out - s_cold_in)))


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
    return (m_hot * (h_hot_in - h_hot_out - T_0 * (s_hot_in - s_hot_out))
            - (m_cold * (h_cold_out - h_cold_in - T_0 * (s_cold_out - s_cold_in)) + power))


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

