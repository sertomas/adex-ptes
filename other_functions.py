import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI as PSI


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

def create_df_from_tespy(component):
    """
    Creates a DataFrame from TESPy component connections for QT diagram plotting.
    """
    import pandas as pd

    # Get connections
    cold_side = next(
        conn for conn in component.inl if isinstance(conn.fluid.val, dict) and list(conn.fluid.val.values())[1] == 1)
    cold_out = next(
        conn for conn in component.outl if isinstance(conn.fluid.val, dict) and list(conn.fluid.val.values())[1] == 1)
    hot_side = next(
        conn for conn in component.inl if isinstance(conn.fluid.val, dict) and list(conn.fluid.val.values())[1] == 0)
    hot_out = next(
        conn for conn in component.outl if isinstance(conn.fluid.val, dict) and list(conn.fluid.val.values())[1] == 0)

    # Create data dictionary
    data = {
        'connection': ['hot_in', 'hot_out', 'cold_in', 'cold_out'],
        'fluid': [
            next(name for name, val in hot_side.fluid.val.items() if val == 1),
            next(name for name, val in hot_out.fluid.val.items() if val == 1),
            next(name for name, val in cold_side.fluid.val.items() if val == 1),
            next(name for name, val in cold_out.fluid.val.items() if val == 1)
        ],
        'T [°C]': [
            hot_side.T.val,
            hot_out.T.val,
            cold_side.T.val,
            cold_out.T.val
        ],
        'p [bar]': [
            hot_side.p.val,
            hot_out.p.val,
            cold_side.p.val,
            cold_out.p.val
        ],
        'h [kJ/kg]': [
            hot_side.h.val,
            hot_out.h.val,
            cold_side.h.val,
            cold_out.h.val
        ],
        'm [kg/s]': [
            hot_side.m.val,
            hot_out.m.val,
            cold_side.m.val,
            cold_out.m.val
        ]
    }

    return pd.DataFrame(data)


# Usage example:
def plot_qt_diagram(component, delta_t_min, system_name="Heat Pump", case_name="Base Case", plot=True):
    df = create_df_from_tespy(component)
    return qt_diagram(
        df=df,
        component_name=component.label,
        hot_in=0,
        hot_out=1,
        cold_in=2,
        cold_out=3,
        delta_t_min=delta_t_min,
        system=system_name,
        case=case_name,
        plot=plot
    )
