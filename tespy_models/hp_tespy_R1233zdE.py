from tespy.networks import Network
from tespy.components import (Compressor, HeatExchanger, SimpleHeatExchanger, Valve, Source, Sink, CycleCloser)
from tespy.connections import Connection, Bus
from other_functions import plot_qt_diagram
import numpy as np

# Create network and specify fluids
nw = Network(T_unit='C', p_unit='bar', h_unit='kJ / kg', s_unit='kJ / kgK', m_unit='kg / s')

water = 'REFPROP::water'
wf = 'REFPROP::R1233zdE'
water_dict = {wf: 0, water: 1}
wf_dict = {wf: 1, water: 0}

# Components
comp = Compressor('compressor')
cond = HeatExchanger('condenser')
ihx = HeatExchanger('internal heat exchanger')
valve = Valve('valve')
eva = SimpleHeatExchanger('evaporator')
cc = CycleCloser('cycle closer')

# Heat sink and source
source_cold = Source('source cold')
sink_cold = Sink('sink cold')
source_hot = Source('source hot')
sink_hot = Sink('sink hot')

# Connections working fluid
c11 = Connection(cc, 'out1', cond, 'in1', label='11')
c12 = Connection(cond, 'out1', ihx, 'in1', label='12')
c13 = Connection(ihx, 'out1', valve, 'in1', label='13')
c14 = Connection(valve, 'out1', eva, 'in1', label='14')
c15 = Connection(eva, 'out1', ihx, 'in2', label='15')
c16 = Connection(ihx, 'out2', comp, 'in1', label='16')
c10 = Connection(comp, 'out1', cc, 'in1', label='10')

# Connections water
c21 = Connection(source_hot, 'out1', cond, 'in2', label='21')
c22 = Connection(cond, 'out2', sink_hot, 'in1', label='22')

# Add connections to network
nw.add_conns(c10, c11, c12, c13, c14, c15, c16, c21, c22)

# Starting value first
comp.set_attr(eta_s=0.85)
cond.set_attr(pr1=0.95, pr2=1)
ihx.set_attr(pr1=0.985, pr2=0.985)
eva.set_attr(pr=0.95)

c11.set_attr(p=24, fluid=wf_dict)
c12.set_attr(h=290)
c14.set_attr(p=0.6)
c15.set_attr(x=1)
c16.set_attr(h=410)
c21.set_attr(T=70, p=5, m=10, fluid=water_dict)
c22.set_attr(T=140)

nw.solve(mode='design')

# Now setting the desired parameters
c12.set_attr(h=None)
c16.set_attr(h=None)
cond.set_attr(ttd_l=5)
ihx.set_attr(ttd_u=5)

# Add busses for power and heat
power = Bus('power input')
power.add_comps({'comp': comp, 'base': 'bus'})

heat_cond = Bus('heat condenser')
heat_cond.add_comps({'comp': cond, 'base': 'bus'})

nw.add_busses(power, heat_cond)

# Solve network
nw.solve(mode='design')
nw.print_results()

print('COP:', round(abs(heat_cond.P.val / power.P.val),2))


plot_qt_diagram(cond, delta_t_min=5)