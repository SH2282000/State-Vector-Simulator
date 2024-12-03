"""Benchmarking the simulator."""

import os
import time
from pathlib import Path

import numpy as np

from QCP.parser.parseQCP import QCPcircuit, parseQCP
from QCP.template.svtemplate import SVtemplate
from QCP.template.utils.equivalence_testing import equivalence_check_sv

FILE_PATH = Path("QCP/circuits/benchmarks")
FILE_NAMES = [
    "bernstein_16",
    "grover_n3",
    "grover_n5",
    "grover_n10",
    "haar_random_n5_d3",
    "haar_random_n20_d10",
    "find_period_2_x_mod_15",
    "find_period_2_x_mod_21",
    "grover_n30",
]


def _simulate(circuit: QCPcircuit, file: str):
    simulator = SVtemplate(circuit)
    a = time.perf_counter()
    simulator.simulate()
    b = time.perf_counter()
    duration = b - a
    print(f"DURATION PROCESS for {file}: {duration}")
    circuit_name = file.replace(str(FILE_PATH), "").replace(".qcp", "")

    formatted_psi = np.array2string(
        simulator.psi,
        formatter={"float_kind": lambda x: "%.4f" % x},
        separator=",",
    )
    print(
        "IT JUST WORKS!"
        if equivalence_check_sv(simulator.psi, circuit_name)
        else "IT JUST WORKS! (maybe)"
    )
    try:
        os.mkdir("outputs")
    except FileExistsError:
        pass
    with Path.open(
        f"outputs/{circuit_name}_output.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(formatted_psi)
    try:
        os.mkdir("outputs")
    except FileExistsError:
        pass
    with Path.open(
        f"outputs/{circuit_name}_output.txt",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(f"\nDURATION PROCESS for {file}: {duration}")


def benchmark_circuit_sv():
    """Test all the different circuit with different initial states."""

    print(FILE_PATH)
    for file in FILE_NAMES:
        circuit = parseQCP(FILE_PATH / f"{file}.qcp")

        print(f"START PROCESS: {file}")
        _simulate(circuit, file)


if __name__ == "__main__":
    benchmark_circuit_sv()
