# "hp_network" create a simple heat pump using the given working fluids
# "orc_network" create a simple ORC cycle using the given working fluids


from tespy.networks import Network
from tespy.connections import Connection
from tespy.components import (
    CycleCloser,
    Compressor,
    Pump,
    Turbine,
    Drum,
    Valve,
    HeatExchanger,
    Source,
    Sink
)


def hp_network(working_fluids):

    heat_pump = Network(
            fluids=working_fluids,
            T_unit="C", p_unit="bar", h_unit="kJ / kg", m_unit="kg / s"
    )

    cycle_closer = CycleCloser('cycle closer')
    condenser_hp = HeatExchanger('condenser hp')
    evaporator_hp = HeatExchanger('evaporator hp')
    valve = Valve('expansion valve')
    compressor = Compressor('compressor')
    amb_water_in_hp = Source('inlet ambient water hp')
    amb_water_out_hp = Sink('outlet ambient water hp')
    water_charge_in = Source('inlet water charging')
    water_charge_out = Sink('outlet water charging')

    # Heat pump cycle
    c1 = Connection(cycle_closer, 'out1', evaporator_hp, 'in2', label='1')
    c2 = Connection(evaporator_hp, 'out2', compressor, 'in1', label='2')
    c3 = Connection(compressor, 'out1', condenser_hp, 'in1', label='3')
    c4 = Connection(condenser_hp, 'out1', valve, 'in1', label='4')
    c0 = Connection(valve, 'out1', cycle_closer, 'in1', label='0')

    # Ambient water
    c5 = Connection(amb_water_in_hp, 'out1', evaporator_hp, 'in1', label='5')
    c6 = Connection(evaporator_hp, 'out1', amb_water_out_hp, 'in1', label='6')

    # TES charging water
    c7 = Connection(water_charge_in, 'out1', condenser_hp, 'in2', label='7')
    c8 = Connection(condenser_hp, 'out2', water_charge_out, 'in1', label='8')

    # all connections have to be added to the network
    heat_pump.add_conns(c1, c2, c3, c4, c0, c5, c6, c7, c8)

    return heat_pump


def orc_network(working_fluids):

    orc = Network(
        fluids=working_fluids,
        T_unit="C", p_unit="bar", h_unit="kJ / kg", m_unit="kg / s"
    )

    cycle_closer_orc = CycleCloser('cycle closer orc')
    condenser_orc = HeatExchanger('condenser orc')
    evaporator_orc = HeatExchanger('evaporator orc')
    turbine = Turbine('steam turbine')
    pump = Pump('pump')
    drum = Drum('drum')
    amb_water_in_orc = Source('inlet ambient water orc')
    amb_water_out_orc = Sink('outlet ambient water orc')
    water_discharge_in = Source('inlet water discharging')
    water_discharge_out = Sink('outlet water discharging')

    # ORC
    c11 = Connection(cycle_closer_orc, 'out1', condenser_orc, 'in1', label='11')
    c12 = Connection(condenser_orc, 'out1', pump, 'in1', label='12')
    c13 = Connection(pump, 'out1', evaporator_orc, 'in2', label='13')
    c14 = Connection(evaporator_orc, 'out2', turbine, 'in1', label='14')
    c10 = Connection(turbine, 'out1', cycle_closer_orc, 'in1', label='10')

    # Ambient water
    c15 = Connection(amb_water_in_orc, 'out1', condenser_orc, 'in2', label='15')
    c16 = Connection(condenser_orc, 'out2', amb_water_out_orc, 'in1', label='16')

    # TES charging water
    c17 = Connection(water_discharge_in, 'out1', evaporator_orc, 'in1', label='17')
    c18 = Connection(evaporator_orc, 'out1', water_discharge_out, 'in1', label='18')

    # all connections have to be added to the network
    orc.add_conns(c11, c12, c13, c14, c10, c15, c16, c17, c18)

    return orc
