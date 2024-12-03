import ast
import json


def get_equality_dict(output_observed, output_expected, pval, reverse=False):
    # Loop over all expected non-zero probability measurement results
    for measurement_result, prob_expected in output_expected.items():
        # Get the observed probability for the given measurement result
        prob_observed = (
            output_observed[measurement_result[::-1]]
            if reverse
            else output_observed[measurement_result]
        )
        # The threshold is the maximum absolute difference between the observed and expected probabilities
        threshold = prob_observed * pval
        if abs(prob_observed - prob_expected) > threshold:
            print(
                measurement_result,
                "\n Your difference:",
                abs(prob_observed - prob_expected),
                "Threshold:",
                threshold,
                "\n Not okay yet.",
            )
            # If the difference of the amplitudes is over the threshold, then the probability distribution for this order is not correct
            return False
    return True


def equivalence_check_dict(output_observed, algorithm, pval=0.04):
    with open("QCP/circuits/solution_benchmark.json") as algorithms_file:
        # Load json file and find the expected output for the given algorithm
        output_expected = json.load(algorithms_file)[algorithm]

        # Check whether the observed and expected results are equal with one qubit order
        equality = get_equality_dict(output_observed, output_expected, pval)
        if equality:
            return True
        print("\n Trying the other qubit order...")
        # ... If they are not equal, check with the other qubit order
        if not equality:
            equality = get_equality_dict(
                output_observed, output_expected, pval, reverse=True
            )
        return equality


def equivalence_check_dict_txt(output_observed, algorithm, pval=0.04):
    with open("QCP/circuits/solution_benchmark.json") as algorithms_file:
        with open(output_observed, "r") as output_file:
            output_observed = ast.literal_eval(output_file.read())
            # Load json file and find the expected output for the given algorithm
            output_expected = json.load(algorithms_file)[algorithm]

            # Check whether the observed and expected results are equal with one qubit order
            equality = get_equality_dict(output_observed, output_expected, pval)
            if equality:
                return True
            print("\n Trying the other qubit order...")
            # ... If they are not equal, check with the other qubit order
            if not equality:
                equality = get_equality_dict(
                    output_observed, output_expected, pval, reverse=True
                )
            return equality


def get_equality_sv(sv_observed, output_expected, pval, reverse=False):
    # Loop over all expected non-zero probability measurement results
    for measurement_result, prob in output_expected.items():
        # Get the given measurement result as decimal number for the key of the statevector
        key_dec = (
            int(measurement_result, 2)
            if not reverse
            else int(measurement_result[::-1], 2)
        )

        # Calculate the absolute probability using the amplitude
        abs_probability = abs(sv_observed[key_dec]) ** 2
        # The threshold is the maximum absolute difference between the observed and expected probabilities
        threshold = abs_probability * pval
        if abs(abs_probability - prob) > threshold:
            print(
                measurement_result,
                "\n Your difference:",
                abs(abs_probability - prob),
                "Threshold:",
                threshold,
                "\n Not okay yet.",
            )
            # If the difference of the amplitudes is over the threshold, then the probability distribution for this order is not correct
            return False
    return True


def equivalence_check_sv(sv_observed, algorithm, pval=0.04):
    with open("QCP/circuits/solution_benchmark.json") as algorithms_file:
        # Load json file and find the expected output for the given algorithm
        output_expected = json.load(algorithms_file)[algorithm]

        # Check whether the observed and expected results are equal with one qubit order
        equality = get_equality_sv(sv_observed, output_expected, pval)
        if equality:
            return True
        print("\n Trying the other qubit order...")
        # ... If they are not equal, check with the other qubit order
        equality = get_equality_sv(sv_observed, output_expected, pval, reverse=True)
        return equality


def equivalence_check_sv_txt(sv_observed, algorithm, pval=0.04):
    with open("QCP/circuits/solution_benchmark.json") as algorithms_file:
        with open(sv_observed, "r") as output_file:
            sv_observed = ast.literal_eval(output_file.read())
            # Load json file and find the expected output for the given algorithm
            output_expected = json.load(algorithms_file)[algorithm]

            # Check whether the observed and expected results are equal with one qubit order
            equality = get_equality_sv(sv_observed, output_expected, pval)
            if equality:
                return True
            print("\n Trying the other qubit order...")
            # ... If they are not equal, check with the other qubit order
            equality = get_equality_sv(sv_observed, output_expected, pval, reverse=True)
            return equality
