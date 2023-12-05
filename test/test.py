from tespy.networks import Network
from CoolProp.CoolProp import PropsSI as PSI

# create a network object with R245fa as fluid
my_plant = Network()
my_plant.set_attr(T_unit='C', p_unit='bar', h_unit='kJ / kg')

from tespy.components import (
    CycleCloser, Compressor, Valve, SimpleHeatExchanger, HeatExchanger
)

cc = CycleCloser('cycle closer')

cond = SimpleHeatExchanger('condenser')
eva = SimpleHeatExchanger('evaporator')
val = Valve('expansion valve')
comp = Compressor('compressor')
ihx = HeatExchanger('internal heat exchanger')

from tespy.connections import Connection

# connections of heat pump
c31 = Connection(cc, 'out1', cond, 'in1', label='31')
c32 = Connection(cond, 'out1', ihx, 'in1', label='32')
c33 = Connection(ihx, 'out1', val, 'in1', label='33')
c34 = Connection(val, 'out1', eva, 'in1', label='34')
c35 = Connection(eva, 'out1', ihx, 'in2', label='35')
c36 = Connection(ihx, 'out2', comp, 'in1', label='36')
c30 = Connection(comp, 'out1', cc, 'in1', label='30')

my_plant.add_conns(c31, c32, c33, c34, c35, c36, c30)

cond.set_attr(pr=0.98)
eva.set_attr(pr=0.98)
ihx.set_attr(pr1=0.98, pr2=0.98)

comp.set_attr(eta_s=0.85)

c35.set_attr(T=10, x=1, fluid={'REFPROP::R245fa': 1}, m=10)
c36.set_attr(h=468)  # change heat flow of IHX
c31.set_attr(p=18)
c32.set_attr(T=75)

my_plant.solve(mode='design')
my_plant.results['Connection'].round(5).to_csv('test_results.csv')

print(f'COP = {abs(cond.Q.val) / comp.P.val}')
print('eta_s = ' + str(round(comp.eta_s.val, 3)) + '. It should be 0.85.')