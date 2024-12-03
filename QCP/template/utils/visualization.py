import math
from enum import Enum
from random import choice

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from QCP.parser.parseQCP import QCPcircuit
from QCP.template.utils.n_weighted_coinflip import n_weighted_coinflip


def plot_prob(prob, phase, hide_legend=False, hide_phase=False):
    x_labels = []
    bar_colors = []

    plot_prob = []
    plot_phase = []

    for i in range(len(prob)):
        if not math.isclose(prob[i], 0, abs_tol=1e-4):
            plot_prob.append(prob[i])
            plot_phase.append(phase[i])
            x_labels.append(
                format(i, "0" + str(math.floor(math.log2(len(prob)))) + "b")
            )

    if not hide_phase:
        if not hide_legend:
            red_patch = mpatches.Patch(color="red", label="Negative")
            blue_patch = mpatches.Patch(color="blue", label="Positive")
            plt.legend(title="Phase", handles=[blue_patch, red_patch])

        for i in range(len(plot_phase)):
            if plot_phase[i] < 0:
                bar_colors.append("red")
            else:
                bar_colors.append("blue")

        plt.bar(x_labels, plot_prob, color=bar_colors)
    else:
        plt.bar(x_labels, plot_prob)

    plt.title("Probability distribution")
    plt.show()


def plot(simulator, hide_legend=False, hide_phase=False):
    res = simulator.get_probabilities_and_phases()
    plot_prob(res[0].flatten(), res[1].flatten(), hide_legend, hide_phase)


def plot_histogram(
    simulator,
    amount=50,
    noise={"bitflip": 0, "phaseflip": 0, "amplitude_damping": (1, 0)},
):
    # save original noise of the simulator
    noise_og = simulator.noise

    def with_error_sim(simulator, noise):
        simulator.noise = noise
        simulator.reset()
        simulator.simulate()
        return simulator.get_probabilities_and_phases()

    probs = simulator.get_probabilities_and_phases()

    def no_error_sim(simulator, noise):
        return probs

    if noise["bitflip"] or noise["phaseflip"]:
        fct = with_error_sim
    else:
        fct = no_error_sim

    data = {}

    for i in range(amount):
        res = fct(simulator, noise)
        res_measured = n_weighted_coinflip(res[0])
        if data.get(res_measured, "") != "":
            data[res_measured] += 1
        else:
            data[res_measured] = 1

    # Insert original noise data into simulator
    simulator.noise = noise_og

    # Plot data
    arr = dict(sorted(data.items()))
    print(arr.values())
    plt.bar(arr.keys(), arr.values())
    plt.title("Results from " + str(amount) + " tries")
    plt.show()


class Colors(Enum):
    """The Colors."""

    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    RESET = "\033[0m"


def print_circuit(circuit: QCPcircuit):
    num_qubits = circuit.numQubits

    qubit_lines = [[f"\u03c8{i}\t"] for i in range(num_qubits - 1, -1, -1)]

    for gate_batch in circuit.gates:
        evaluated_qubits = []
        for gate in gate_batch:
            if gate.name.startswith("cc"):
                random_color = choice(
                    [color for color in Colors if color != Colors.RESET]
                )

                evaluated_qubits += [gate.target] + gate.control
                # Control and target gate
                control1, control2 = gate.control[0], gate.control[1]
                target = gate.target

                # Add control symbol (blacksquare) at control qubit
                control_qubit_idx_1 = (num_qubits - 1) - control1
                control_qubit_idx_2 = (num_qubits - 1) - control2

                # Add gate name at target qubit
                target_qubit_idx = (num_qubits - 1) - target
                gate_name = f"{random_color.value} {gate.name[2:].capitalize()} {Colors.RESET.value}"

                qubit_lines[control_qubit_idx_1].append(
                    f"{random_color.value} \u25a0 {Colors.RESET.value}"
                )
                qubit_lines[control_qubit_idx_2].append(
                    f"{random_color.value} \u25a0 {Colors.RESET.value}"
                )
                qubit_lines[target_qubit_idx].append(gate_name)

                pass
            elif gate.name.startswith("c"):
                random_color = choice(
                    [color for color in Colors if color != Colors.RESET]
                )

                evaluated_qubits += [gate.target] + gate.control
                # Control and target gate
                control1 = gate.control[0]
                target = gate.target

                # Add control symbol (blacksquare) at control qubit
                control_qubit_idx_1 = (num_qubits - 1) - control1

                # Add gate name at target qubit
                target_qubit_idx = (num_qubits - 1) - target
                gate_name = f"{gate.name[1:].capitalize()}"

                qubit_lines[control_qubit_idx_1].append(
                    f"{random_color.value} \u25a0 {Colors.RESET.value}"
                )
                qubit_lines[target_qubit_idx].append(
                    f"{random_color.value} {gate_name} {Colors.RESET.value}"
                )
                pass
            else:
                target = gate.target
                evaluated_qubits += [target]

                target_qubit_idx = num_qubits - 1 - target
                qubit_lines[target_qubit_idx].append(f" {gate.name.capitalize()} ")

        for qubit in range(num_qubits):
            if qubit not in evaluated_qubits:
                qubit_idx = num_qubits - 1 - qubit
                qubit_lines[qubit_idx].append(" - ")

    res = "\n".join("-".join(x) for x in qubit_lines)
    print(res)
