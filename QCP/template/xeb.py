from copy import deepcopy

import numpy as np
import itertools
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from QCP.parser.parseQCP import QCPcircuit, Gate
from QCP.template.svtemplate import SVtemplate
from QCP.template.dmtemplate import DMtemplate
from QCP.template.utils.n_weighted_coinflip import n_weighted_coinflip

# Define xeb parameters
exponents = np.linspace(0, 7 / 4, 8)
rotation_zero_qubit = [
    Gate("phasedXZ", param=[0.5, z, a], target=0)
    for a, z in itertools.product(exponents, repeat=2)
]
rotation_one_qubit = [
    Gate("phasedXZ", param=[0.5, z, a], target=1)
    for a, z in itertools.product(exponents, repeat=2)
]
depth = 4
random_generator = np.random.default_rng(seed=1337)


def build_random_circuit_xeb(
    random_generator, depth, rotations_zero_qubit, rotations_one_qubit
):
    # Create a random circuit
    # Apply rotation in the beginning
    zero_qubit_rotation = random_generator.choice(rotations_zero_qubit, 1)[0]
    one_qubit_rotation = random_generator.choice(rotations_one_qubit, 1)[0]
    gates = [[zero_qubit_rotation, one_qubit_rotation]]
    for _ in range(depth):
        # Apply two qubit gate
        gates += [[Gate("sqrt_iswap", target=[0, 1])]]
        # Apply rotation after
        zero_qubit_rotation = random_generator.choice(rotations_zero_qubit, 1)[0]
        one_qubit_rotation = random_generator.choice(rotations_one_qubit, 1)[0]
        gates += [[zero_qubit_rotation, one_qubit_rotation]]
    return QCPcircuit(numQubits=2, gates=gates)


# Create 10 random circuits of depth 100
max_depth = 100
n_circuits = 10
circuits = [
    build_random_circuit_xeb(
        random_generator, max_depth, rotation_zero_qubit, rotation_one_qubit
    )
    for _ in range(n_circuits)
]
# We will truncate to these lengths
cycle_depths = np.arange(1, max_depth + 1, 9)

# Set Noise for DensityMatrix Simulator
e_depolarizing = 5e-3

# Run simulations
records = []
noisy_runs = 100
for cycle_depth in cycle_depths:
    for circuit_i, circuit in enumerate(circuits):
        circuit_copy = deepcopy(circuit)

        # Initialize the simulator
        pure_sim = SVtemplate(circuit_copy)
        noisy_sim = DMtemplate(
            circuit_copy,
            noise={
                "bitflip": 0,
                "phaseflip": 0,
                "depolarization": e_depolarizing,
                "amplitude_damping": (1, 0),
            },
        )

        # Truncate the long circuit to the requested cycle_depth
        circuit_depth = cycle_depth * 2 + 1
        assert circuit_depth <= len(circuit_copy.gates)
        circuit_copy.gates = circuit_copy.gates[:circuit_depth]

        # Pure-state simulation
        psi = pure_sim.simulate()
        pure_probs = pure_sim.get_probabilities_and_phases()[0]

        # Noisy execution
        circuit_copy.gates = circuit_copy.gates

        noisy_results = {}
        for _ in range(noisy_runs):
            noisy_sim.simulate()
            res = noisy_sim.get_probabilities_and_phases()
            res_measured = n_weighted_coinflip(res[0])
            if noisy_results.get(res_measured, "") != "":
                noisy_results[res_measured] += 1
            else:
                noisy_results[res_measured] = 1
            # Reset simulator
            noisy_sim.reset()

        # Ensure all results are present
        if len(noisy_results) != 2**2:
            noisy_results.update(
                {f"{i:02b}": 0 for i in range(2**2) if f"{i:02b}" not in noisy_results}
            )
        noisy_results = dict(sorted(noisy_results.items()))

        sampled_probs = np.array(list(noisy_results.values())) / noisy_runs

        # Save the results
        records += [
            {
                "circuit_i": circuit_i,
                "cycle_depth": cycle_depth,
                "circuit_depth": circuit_depth,
                "pure_probs": pure_probs.flatten(),
                "sampled_probs": sampled_probs,
            }
        ]
        print(".", end="", flush=True)

for record in records:
    e_u = np.sum(record["pure_probs"] ** 2)
    u_u = np.sum(record["pure_probs"]) / 4
    m_u = np.sum(record["pure_probs"] * record["sampled_probs"])
    record.update(
        e_u=e_u,
        u_u=u_u,
        m_u=m_u,
    )

df = pd.DataFrame(records)
df["y"] = df["m_u"] - df["u_u"]
df["x"] = df["e_u"] - df["u_u"]

df["numerator"] = df["x"] * df["y"]
df["denominator"] = df["x"] ** 2


# Color by cycle depth
colors = sns.cubehelix_palette(n_colors=len(cycle_depths))
colors = {k: colors[i] for i, k in enumerate(cycle_depths)}

_lines = []


def per_cycle_depth(df):
    fid_lsq = df["numerator"].sum() / df["denominator"].sum()

    cycle_depth = df.name
    xx = np.linspace(0, df["x"].max())
    (line,) = plt.plot(xx, fid_lsq * xx, color=colors[cycle_depth])
    plt.scatter(df["x"], df["y"], color=colors[cycle_depth])

    global _lines
    _lines += [line]  # for legend
    return pd.Series({"fidelity": fid_lsq})


fids = df.groupby("cycle_depth").apply(per_cycle_depth).reset_index()
plt.xlabel(r"$e_U - u_U$", fontsize=18)
plt.ylabel(r"$m_U - u_U$", fontsize=18)
_lines = np.asarray(_lines)
plt.legend(_lines[[0, -1]], cycle_depths[[0, -1]], loc="best", title="Cycle depth")
plt.tight_layout()
plt.show()


plt.plot(fids["cycle_depth"], fids["fidelity"], marker="o", label="Least Squares")

xx = np.linspace(0, fids["cycle_depth"].max())

# In XEB, we extract the depolarizing fidelity, which is
# related to (but not equal to) the Pauli error.
# For the latter, an error involves doing X, Y, or Z with E_PAULI/3
# but for the former, an error involves doing I, X, Y, or Z with e_depol/4
e_depol = e_depolarizing / (1 - 1 / 4**2)

# The additional factor of four in the exponent is because each layer
# involves two moments of two qubits (so each layer has four applications
# of a single-qubit single-moment depolarizing channel).
plt.plot(xx, (1 - e_depol) ** (4 * xx), label=r"$(1-\mathrm{e\_depol})^{4d}$")

plt.ylabel("Circuit fidelity", fontsize=18)
plt.xlabel("Cycle Depth $d$", fontsize=18)
plt.legend(loc="best")
plt.yscale("log")
plt.tight_layout()
plt.show()
