import time

from QCP.parser.parseQCP import parseQCP
from QCP.template.svtemplate import SVtemplate
from QCP.template.utils.equivalence_testing import equivalence_check_sv

circuit_name = "find_period_2_x_mod_15"

c = parseQCP(f"QCP/circuits/benchmarks/{circuit_name}.qcp")
# print_circuit(c)
simulator = SVtemplate(c)
a = time.perf_counter()
simulator.simulate()
b = time.perf_counter()
print(f"DOES IT WORK: {equivalence_check_sv(simulator.psi, circuit_name)}")
print(simulator.psi)
print("Time taken: ", b - a)
