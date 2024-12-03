import cmath
import math
import re
from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class Gate:
    name: str = None
    param: Union[float, List[float]] = None
    target: int = None
    control: Union[int, List[int]] = None

    def __str__(self):
        strg = self.name
        if self.param is not None:
            strg += f" {self.param}"
        if self.control is not None:
            if isinstance(self.control, list):
                strg += " " + ", ".join(map(str, self.control))
            else:
                strg += f" {self.control}"

        return strg + f" {self.target}"


@dataclass
class QCPcircuit:
    numQubits: int = None
    gates: list[list[Gate]] = field(default_factory=list)

    def __str__(self):
        str = f"{self.numQubits}\n"
        str += "\n".join([i.__str__() for i in self.gates])
        return str


def add_non_trivial_gate(
    circ_string_per_qbit: [], target: [], control: [], name: str, param: []
):
    affected = control + target
    id = name + "".join(map(str, affected))
    is_redundant = True

    for i in affected:
        last_gate = circ_string_per_qbit[i][-1]
        if last_gate["id"] != id:
            is_redundant = False
            break

    for i in affected:
        if is_redundant:
            res = circ_string_per_qbit[i].pop()

            if id.startswith("r"):
                param_new = (res["param"][0] + param[0]) % (2 * math.pi)

                if param_new != 0:
                    circ_string_per_qbit[i].append(
                        {
                            "id": id,
                            "name": name,
                            "target": target,
                            "control": control,
                            "param": [param_new],
                        }
                    )

            if id.startswith("m"):
                circ_string_per_qbit[i].append(
                    {
                        "id": id,
                        "name": name,
                        "target": target,
                        "control": control,
                        "param": [0],
                    }
                )

        else:
            circ_string_per_qbit[i].append(
                {
                    "id": id,
                    "name": name,
                    "target": target,
                    "control": control,
                    "param": param,
                }
            )


def merge_gates_to_circ(circ, gates_per_qbit):
    while max(len(inner_list) for inner_list in gates_per_qbit) > 1:
        popable = [False for _ in range(circ.numQubits)]
        gate_list = []
        for qbit_num in range(circ.numQubits):
            gates = gates_per_qbit[qbit_num]

            if len(gates) <= 1:
                continue

            gate_str = gates[1]

            if len(gates[1]["control"]) == 0:
                gate = Gate(gate_str["name"])
                gate.target = gate_str["target"][0]
                gate.param = gate_str["param"]
                gate_list.append(gate)
                popable[qbit_num] = True
            else:
                # case cx or ccx
                affected = gate_str["control"] + gate_str["target"]
                can_apply = True
                for a in affected:
                    if gates_per_qbit[a][1]["id"] != gate_str["id"]:
                        can_apply = False

                if can_apply:
                    gates[1]["id"] += "applied" + str(qbit_num)
                    gate = Gate(gate_str["name"])
                    gate.target = gate_str["target"][0]
                    gate.param = gate_str["param"]
                    gate.control = gate_str["control"]
                    gate_list.append(gate)
                    for a in affected:
                        popable[a] = True

        for ind in range(circ.numQubits):
            if popable[ind]:
                gates_per_qbit[ind].pop(1)

        circ.gates.append(gate_list)


def parseQCP(path):
    with open(path, "r") as fp:
        gates_per_qbit = []
        circ = QCPcircuit()
        for line in fp.read().splitlines():
            # ignore comments
            if line.startswith("//"):
                continue

            # first line that is no comment has to be num of used qbits
            if circ.numQubits is None:
                circ.numQubits = int(line)
                gates_per_qbit = [
                    [{"id": "", "name": "", "target": [], "control": [], "param": [0]}]
                    for _ in range(circ.numQubits)
                ]
                continue

            gate_comp = line.split()

            # gates with parameters
            if line.startswith("r"):
                add_non_trivial_gate(
                    gates_per_qbit,
                    [int(gate_comp[2])],
                    [],
                    gate_comp[0],
                    [float(eval(gate_comp[1].replace("pi", str(cmath.pi))))],
                )
                continue

            # Toffoli gates
            if line.startswith("cc"):
                # Use regex to split by commas and spaces (benchmark)
                gate_comp = re.split(r"[\s,]+", line.strip())
                # Call amend_for_redundancy with parsed values
                add_non_trivial_gate(
                    gates_per_qbit,
                    [int(gate_comp[3])],
                    [int(gate_comp[1]), int(gate_comp[2])],
                    gate_comp[0],
                    [0],
                )
                continue

            # controlled gates
            if line.startswith("c"):
                add_non_trivial_gate(
                    gates_per_qbit,
                    [int(gate_comp[2])],
                    [int(gate_comp[1])],
                    gate_comp[0],
                    [0],
                )
                continue

            if line.startswith("m"):
                add_non_trivial_gate(
                    gates_per_qbit, [int(gate_comp[1])], [], "measure", [0]
                )
                continue

            if line.startswith("p"):
                add_non_trivial_gate(
                    gates_per_qbit,
                    [int(gate_comp[2])],
                    [],
                    "phasedXZ",
                    list(map(int, gate_comp[1].split(","))),
                )
                continue

            if line.startswith("sqrt_iswap"):
                # sqrt_iswap is special case only used of xeb
                add_non_trivial_gate(gates_per_qbit, [-1], [], "sqrt_iswap", [])
                continue

            # single qbit gates without parameters
            add_non_trivial_gate(
                gates_per_qbit, [int(gate_comp[1])], [], gate_comp[0], [0]
            )

    merge_gates_to_circ(circ, gates_per_qbit)
    return circ
