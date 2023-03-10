# --- OVERALL ------------------------------------------------------------

# set here the ambient conditions
T_amb = 10  # temperature of ambient [°C]
p_amb = 1.013  # pressure of ambient [bar]

# set here the tolerated minimal temperature difference
DeltaT_min = 5  # [K]

# set here the pressure drops in all heat exchangers (for now the same everywhere)
p_loss_rel = 0.95  # relative pressure drop in the heat exchangers

# set here the values for the thermal energy storage
T_low_TES = 70  # low temperature of the water in the TES [°C]
T_high_TES = 130  # high temperature of the water in the TES [°C]


# --- HEAT PUMP ------------------------------------------------------------
inlet_power = 1.5e6  # power to compressor [W]
p_low_hp = 0.45  # inlet pressure of compressor [bar]
p_high_hp = 26  # outlet pressure of condenser [bar]
eta_s_compressor = 0.85  # isentropic efficiency of compressor


# --- ORC ------------------------------------------------------------
outlet_power = 0.4e6  # power from turbine [W]
p_low_orc = 1  # inlet pressure of pump [bar]
p_high_orc = 6.5  # outlet pressure of evaporator [bar]
eta_s_pump = 0.85  # isentropic efficiency of pump
eta_s_turbine = 0.85  # isentropic efficiency of turbine


def hp_settings(network):

    # pressure drops and isentropic efficiencies
    network.get_comp('evaporator hp').set_attr(pr1=p_loss_rel, pr2=p_loss_rel)
    network.get_comp('condenser hp').set_attr(pr1=p_loss_rel, pr2=p_loss_rel)
    network.get_comp('compressor').set_attr(eta_s=eta_s_compressor)

    # heat pump cycle
    network.get_comp('compressor').set_attr(P=inlet_power)
    network.get_conn('2').set_attr(T=T_amb-DeltaT_min, p=p_low_hp, fluid={network.fluids[0]: 1, network.fluids[1]: 0})
    network.get_conn('4').set_attr(p=p_high_hp)
    network.get_comp('condenser hp').set_attr(ttd_l=5)

    # ambient water
    network.get_conn('5').set_attr(T=T_amb, p=p_amb, fluid={network.fluids[0]: 0, network.fluids[1]: 1})
    network.get_conn('6').set_attr(T=5)

    # TES charging water
    network.get_conn('7').set_attr(T=T_low_TES, p=3, fluid={network.fluids[0]: 0, network.fluids[1]: 1})
    network.get_conn('8').set_attr(T=T_high_TES)

    return network


def orc_settings(network):

    # pressure drops and isentropic efficiencies
    network.get_comp('evaporator orc').set_attr(pr1=p_loss_rel, pr2=p_loss_rel)
    network.get_comp('condenser orc').set_attr(pr1=p_loss_rel, pr2=p_loss_rel)
    network.get_comp('pump').set_attr(eta_s=eta_s_pump)
    network.get_comp('steam turbine').set_attr(eta_s=eta_s_turbine)

    # ORC
    network.get_comp('steam turbine').set_attr(P=-outlet_power)
    network.get_conn('12').set_attr(p=p_low_orc, x=0, fluid={network.fluids[0]: 1, network.fluids[1]: 0})
    network.get_conn('14').set_attr(T=120, p=p_high_orc)

    # ambient water
    network.get_conn('15').set_attr(T=10, p=p_amb, fluid={network.fluids[0]: 0, network.fluids[1]: 1})
    network.get_conn('16').set_attr(T=15)

    # TES discharging water
    network.get_conn('17').set_attr(T=T_high_TES, p=3, fluid={network.fluids[0]: 0, network.fluids[1]: 1})
    network.get_conn('18').set_attr(T=T_low_TES)

    return network
