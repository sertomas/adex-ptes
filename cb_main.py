import numpy as np
from cb_set_pars import hp_settings, orc_settings
from cb_network import hp_network, orc_network
from cb_qt_diagram import qt_sens_latent
from cb_set_pressure import set_pressure
from cb_exergy import exerg_an_hp, exerg_an_orc

from config import delta_t_min, T_amb, p_amb, P_in, P_out

# TODO CODE:
#  - Give the results as variables/arrays and not from CSV files (function / global variables?)
#  - Find better way to create data frames / show results
#  - Try to reduce number of functions / make the code more organic and simple

# TODO MODEL:
#  - Perform further sensitivity analyses to find better parameters set
#  - Perform adv. exergy analysis of ORC
#  - Simulate the TES (connecting HP and ORC?)
#  - Simulate time series (change ambient conditions)
#  - Assess flexibility of model by changing pressures, fluid, sizes


# 1) set the networks
hp = hp_network(['R245fa', 'water'])
orc = orc_network(['R245fa', 'water'])

# 2) set the parameters, the initial values and the design variables
# as well as the ambient conditions and the minimum temperature difference
hp_settings(hp)
orc_settings(orc)

# 3) solve the problem and store the results in .csv files
hp.solve(mode='design')
hp.results['Connection'].round(5).to_csv("hp_results.csv")

orc.solve(mode='design')
orc.results['Connection'].round(5).to_csv("orc_results.csv")

# 4a) check QT-diagrams and correct the pressures
delta_t_he = [
    qt_sens_latent(hp.get_conn('5'), hp.get_conn('6'), hp.get_conn('2'), hp.get_conn('1'), delta_t_min, False),
    qt_sens_latent(hp.get_conn('8'), hp.get_conn('7'), hp.get_conn('3'), hp.get_conn('4'), delta_t_min, False),
    qt_sens_latent(orc.get_conn('17'), orc.get_conn('18'), orc.get_conn('14'), orc.get_conn('13'), delta_t_min, False),
    qt_sens_latent(orc.get_conn('16'), orc.get_conn('15'), orc.get_conn('11'), orc.get_conn('12'), delta_t_min, False)
    ]

# in case one or more heat exchangers have too small temperature difference,
# ask the user to perform a sensitivity analysis
if min(delta_t_he) < delta_t_min - 1e-5:
    print("\nWARNING: The minimum temperature difference in one or more heat exchangers is lower than allowed!",
          "Set new parameters or perform a sensitivity analysis.")

    answer = input("Do you want to perform a sensitivity analysis? (y/n)")

    if answer.lower() == "y":

        data = {
            'p_high_hp': np.arange(20, 31),  # [bar]
            'p_high_orc': np.arange(3, 15),  # [bar]
            'p_low_orc': np.arange(0.8, 1.5, 0.1),  # [bar]
        }

        # check the results in the .csv files and select manually the optimal pressures!
        set_pressure(hp, orc, data, delta_t_min)

# 4b) set the optimal pressures and run the model again
# this step can be skipped if the optimal parameters have been found already
hp.get_conn('4').set_attr(p=25.5)
orc.get_conn('14').set_attr(p=7)
orc.get_conn('12').set_attr(p=1.2)

hp.solve('design')
hp.results['Connection'].round(5).to_csv("hp_base_case.csv")

orc.solve('design')
orc.results['Connection'].round(5).to_csv("orc_base_case.csv")

# check if the minimum temperature difference is still respected
delta_t_he = [
    qt_sens_latent(hp.get_conn('5'), hp.get_conn('6'), hp.get_conn('2'), hp.get_conn('1'), delta_t_min, False),
    qt_sens_latent(hp.get_conn('8'), hp.get_conn('7'), hp.get_conn('3'), hp.get_conn('4'), delta_t_min, False),
    qt_sens_latent(orc.get_conn('17'), orc.get_conn('18'), orc.get_conn('14'), orc.get_conn('13'), delta_t_min, False),
    qt_sens_latent(orc.get_conn('16'), orc.get_conn('15'), orc.get_conn('11'), orc.get_conn('12'), delta_t_min, False)
    ]

if min(delta_t_he) < delta_t_min - 1e-5:
    print("\nWARNING: The minimum temperature difference in one or more heat exchangers is lower than allowed!",
          "Set new parameters or perform a sensitivity analysis.")

# 5) conventional exergy analysis
ex_an_hp = exerg_an_hp(hp)
(ex_an_hp.connection_data / 1e3).round(5).to_csv("hp_exan_connections.csv")  # in kJ/kg
ex_an_hp.component_data[ex_an_hp.component_data["E_F"] > 0].round(5).to_csv("hp_exan_components.csv")

ex_an_orc =exerg_an_orc(orc)
(ex_an_orc.connection_data / 1e3).round(5).to_csv("orc_exan_connections.csv")  # in kJ/kg
ex_an_orc.component_data[ex_an_orc.component_data["E_F"] > 0].round(5).to_csv("orc_exan_components.csv")

