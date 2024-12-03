import numpy as np
import pytest

from QCP.parser.parseQCP import parseQCP
from QCP.template.dmtemplate import DMtemplate
from QCP.template.svtemplate import SVtemplate
from QCP.template.utils.files import get_all_files


@pytest.mark.parametrize("file", get_all_files("QCP/circuits/test"))
def test_all_circuits_basic(file):
    """Test all the different circuit with default initial states. |00...0>"""
    c = parseQCP(file)
    simulator = SVtemplate(c)
    simulator.simulate()
    # TODO: Implement the assertion for the result of the simulation.


@pytest.mark.parametrize("file", get_all_files("QCP/circuits/test"))
def test_circuit_sv(file):
    """Test all the different circuit with different initial states."""
    c = parseQCP(file)

    with open(file, "r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            line = line.strip()
            if line.startswith("// TEST:"):
                line_content = line[8:].strip()
                try:
                    evaluated_content = eval(line_content)
                except (SyntaxError, NameError, TypeError) as e:
                    exit(f"Could not evaluate line '{line_content}' in '{file}': {e}")

                simulator = SVtemplate(c)

                simulator.set_psi(evaluated_content[0])
                result = simulator.simulate()
                np.testing.assert_array_almost_equal(
                    desired=np.array(evaluated_content[1], dtype=complex, ndmin=2).T,
                    actual=result,
                    err_msg=f"ERROR LINE {index}: {line_content}",
                )


@pytest.mark.parametrize("file", get_all_files("QCP/circuits/test"))
def test_circuit_dm(file):
    """Test all the different circuit with different initial states."""
    c = parseQCP(file)

    with open(file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("// TEST:"):
                line_content = line[8:].strip()
                try:
                    evaluated_content = eval(line_content)
                except (SyntaxError, NameError, TypeError) as e:
                    exit(f"Could not evaluate line '{line_content}' in '{file}': {e}")

                simulator_dm = DMtemplate(c)
                print("Eval content: ", evaluated_content)
                psi_start, psi_end = (
                    np.array(evaluated_content[0], ndmin=2).T,
                    np.array(evaluated_content[1], ndmin=2).T,
                )

                simulator_dm.rho = psi_start * psi_start.conj().T
                result = simulator_dm.simulate()
                np.testing.assert_array_almost_equal(
                    desired=psi_end * psi_end.conj().T,
                    actual=result,
                    err_msg=f"line {line_content}",
                )
