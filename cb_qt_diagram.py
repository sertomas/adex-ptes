from CoolProp.CoolProp import PropsSI as PSI
import CoolProp.CoolProp as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def qt_sens_latent(sens_hot, sens_cold, latent_hot, latent_cold, delta_t_min, plot):

    # read the fluid of the streams
    fluid_latent = next(key for key, value in latent_cold.fluid.val.items() if value == 1)
    fluid_sens = next(key for key, value in sens_hot.fluid.val.items() if value == 1)

    # set number of points calculated with CoolProp for each segment of sensible heat transfer
    set_points = 5

    # --- SENSIBLE STREAM ----------------------------------------------------------------

    # discretize the pressure by a number of set points and assume pressure drops linearly
    p_sens = np.linspace(sens_hot.p.val, sens_cold.p.val, set_points)

    # discretize the enthalpy by same number of set points
    enthalpy_sens = np.linspace(sens_hot.h.val, sens_cold.h.val, set_points)

    # calculate the temperature T=f(p,h) using CoolProp
    temp_sens = np.zeros(set_points)
    temp_sens[0] = sens_hot.T.val
    temp_sens[-1] = sens_cold.T.val
    for i in range(1, set_points-1):
        temp_sens[i] = cp.PropsSI('T', 'P', p_sens[i] * 1e5, 'H', enthalpy_sens[i] * 1e3, fluid_sens) - 273.15

    # calculate the enthalpy flow
    m_enthalpy_sens = enthalpy_sens*sens_hot.m.val - min(enthalpy_sens)*sens_hot.m.val

    # --- LATENT STREAM ----------------------------------------------------------------

    # assume the latent stream is not affected by pressure drops
    p_latent = max(latent_hot.p.val, latent_cold.p.val)

    # calculate h' and h'' (boiling liquid and saturated vapour) as well as the boiling temperature
    enthalpy_boil = PSI("H", "Q", 0, 'P', p_latent * 1e5, fluid_latent) * 1e-3
    enthalpy_vap = PSI("H", "Q", 1, 'P', p_latent * 1e5, fluid_latent) * 1e-3
    temp_boil = PSI("T", "Q", 1, 'P', p_latent * 1e5, fluid_latent) - 273.15

    # discretize the enthalpy by same number of set points and
    # check if the inlet stream is already boiling / the outlet stream is not saturated
    if latent_cold.h.val < enthalpy_boil:
        enthalpy_liq_lat = np.linspace(latent_cold.h.val, enthalpy_boil, set_points)
    else:
        enthalpy_liq_lat = latent_cold.h.val
    if latent_hot.h.val > enthalpy_vap:
        enthalpy_vap_lat = np.linspace(enthalpy_vap, latent_hot.h.val, set_points)
    else:
        enthalpy_vap_lat = latent_hot.h.val

    # calculate the temperature T=f(p,h) using CoolProp
    temp_liq_lat = np.zeros(set_points)
    temp_vap_lat = np.zeros(set_points)
    temp_liq_lat[0] = latent_cold.T.val
    temp_liq_lat[-1] = temp_boil
    temp_vap_lat[0] = temp_boil
    temp_vap_lat[-1] = latent_hot.T.val
    for i in range(1, set_points - 1):
        if latent_cold.h.val < enthalpy_boil:
            temp_liq_lat[i] = cp.PropsSI('T', 'P', p_latent * 1e5, 'H', enthalpy_liq_lat[i] * 1e3, fluid_latent) - 273.15
        if latent_hot.h.val > enthalpy_vap:
            temp_vap_lat[i] = cp.PropsSI('T', 'P', p_latent * 1e5, 'H', enthalpy_vap_lat[i] * 1e3, fluid_latent) - 273.15

    # calculate the enthalpy flow
    if latent_cold.h.val < enthalpy_boil:
        m_enthalpy_liq_lat = enthalpy_liq_lat * latent_hot.m.val - latent_cold.h.val * latent_hot.m.val
    else:
        m_enthalpy_liq_lat = (enthalpy_liq_lat * latent_hot.m.val - latent_cold.h.val * latent_hot.m.val) * np.ones(set_points)
    if latent_hot.h.val > enthalpy_vap:
        m_enthalpy_vap_lat = enthalpy_vap_lat * latent_hot.m.val - latent_cold.h.val * latent_hot.m.val
    else:
        m_enthalpy_vap_lat = enthalpy_vap_lat * latent_hot.m.val - latent_cold.h.val * latent_hot.m.val * np.ones(set_points)

    # --- ANALYZE AND PLOT ----------------------------------------------------------------

    # check which stream is hot / cold and calculate the differences at the four crucial points
    # (1: cold inlet, 2: cold outlet, 3: beginning of evaporation, 4: end of evaporation)

    # get the temperature of the sensible stream at the boundary points of the evaporation/condensation
    p3 = abs(sens_hot.p.val - sens_cold.p.val) / max(m_enthalpy_sens) * max(m_enthalpy_liq_lat) + min(sens_hot.p.val, sens_cold.p.val)
    p4 = abs(sens_hot.p.val - sens_cold.p.val) / max(m_enthalpy_sens) * min(m_enthalpy_vap_lat) + min(sens_hot.p.val, sens_cold.p.val)
    h3 = (max(m_enthalpy_liq_lat) - min(m_enthalpy_liq_lat)) / sens_cold.m.val + sens_cold.h.val
    h4 = sens_hot.h.val - (max(m_enthalpy_vap_lat) - min(m_enthalpy_vap_lat)) / sens_cold.m.val
    t3 = cp.PropsSI('T', 'P', p3 * 1e5, 'H', h3 * 1e3, fluid_sens) - 273.15
    t4 = cp.PropsSI('T', 'P', p4 * 1e5, 'H', h4 * 1e3, fluid_sens) - 273.15

    # if latent_hot.T.val > sens_hot.T.val:
    difference = [sens_cold.T.val - latent_cold.T.val, t3 - temp_boil, t4 - temp_boil, sens_hot.T.val - latent_hot.T.val ]
    if latent_cold.T.val > sens_cold.T.val:
        difference = [- d for d in difference]

    # plot the results
    mpl.rcParams['font.size'] = 16
    plt.figure(figsize=(14, 7))
    plt.plot(m_enthalpy_sens, temp_sens, color='red')
    plt.plot(m_enthalpy_liq_lat, temp_liq_lat, color='blue')
    plt.plot(m_enthalpy_vap_lat, temp_vap_lat, color='blue')
    plt.plot([min(m_enthalpy_vap_lat), max(m_enthalpy_liq_lat)], [temp_boil, temp_boil], color='blue')
    plt.legend(["Hot side", "Cold side"])
    plt.xlabel('Q [kW]')
    plt.grid(True)
    plt.ylabel('T [°C]')
    plt.xlim(0, max(m_enthalpy_sens))
    plt.title("QT diagram (no pressure losses in the latent stream)")
    if plot:
        plt.show()

    if min(difference) > delta_t_min - 1e-5:
        print("The minimum temperature difference is: " + str(round(min(difference), 2)) + "K and is equal or higher than the allowed minimum temperature difference in a heat exchanger (" + str(delta_t_min) + "K).")
    else:
        print("WARNING: The minimum temperature difference is: " + str(round(min(difference), 2)) + "K and is lower than the allowed minimum temperature difference in a heat exchanger (" + str(delta_t_min) + "K).")

    return min(difference)


