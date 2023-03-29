# "exerg_an_hp" perform a conventional exergy analysis of the heat pump
# "exerg_an_orc" perform a conventional exergy analysis of the ORC

from config import delta_t_min, T_amb, p_amb
from tespy.connections import Connection, Bus, Ref
from tespy.tools import ExergyAnalysis


def exerg_an_hp(hp):

    # 1) set exergy streams and add them to the networks

    power_bus_hp = Bus('power input')
    power_bus_hp.add_comps({'comp': hp.get_comp('compressor'), 'char': 0.95, 'base': 'bus'})

    charging_water_bus = Bus('charging water')
    charging_water_bus.add_comps({'comp': hp.get_comp('outlet water charging'), 'base': 'bus'},
                                 {'comp': hp.get_comp('inlet water charging')})

    amb_water_hp_bus = Bus('ambient water hp')
    amb_water_hp_bus.add_comps({'comp': hp.get_comp('inlet ambient water hp'), 'base': 'bus'},
                               {'comp': hp.get_comp('outlet ambient water hp')})

    hp.add_busses(power_bus_hp, charging_water_bus, amb_water_hp_bus)

    # 2) carry out exergy analysis

    ex_an_hp = ExergyAnalysis(hp, E_F=[power_bus_hp], E_P=[charging_water_bus], E_L=[amb_water_hp_bus])
    ex_an_hp.analyse(pamb=p_amb, Tamb=T_amb)

    return ex_an_hp


def exerg_an_orc(orc):

    # 1) set exergy streams and add them to the networks

    power_bus_orc = Bus('power output')
    power_bus_orc.add_comps({'comp': orc.get_comp('steam turbine'), 'char': 0.95, 'base': 'component'},
                            {'comp': orc.get_comp('pump'), 'char': 0.95, 'base': 'bus'})

    discharging_water_bus = Bus('discharging water')
    discharging_water_bus.add_comps({'comp': orc.get_comp('outlet water discharging'), 'base': 'bus'},
                                    {'comp': orc.get_comp('inlet water discharging')})

    amb_water_orc_bus = Bus('ambient water orc')
    amb_water_orc_bus.add_comps({'comp': orc.get_comp('inlet ambient water orc'), 'base': 'bus'},
                                {'comp': orc.get_comp('outlet ambient water orc')})

    orc.add_busses(power_bus_orc, discharging_water_bus, amb_water_orc_bus)

    # 2) carry out exergy analysis

    ex_an_orc = ExergyAnalysis(orc, E_F=[discharging_water_bus], E_P=[power_bus_orc], E_L=[amb_water_orc_bus])
    ex_an_orc.analyse(pamb=p_amb, Tamb=T_amb)

    return ex_an_orc

