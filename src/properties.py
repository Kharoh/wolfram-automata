def is_legal_wolfram(rule_number: int) -> bool:
    """
    Determines if a Wolfram rule is 'legal'.

    A rule is considered legal if it satisfies two conditions:
    1. A quiescent state of all zeros remains unchanged.
    2. The rule is reflection-symmetric (e.g., '100' produces the same
       output as '001').

    Args:
        rule_number (int): The elementary cellular automaton rule (0-255).

    Returns:
        bool: True if the rule is legal, False otherwise.
    """
    if not 0 <= rule_number <= 255:
        raise ValueError("Rule number must be between 0 and 255.")

    # Convert the rule number to its 8-bit binary representation.
    # The bits correspond to neighborhoods from '111' down to '000'.
    rule_bin = format(rule_number, "08b")

    # Condition 1: The '000' neighborhood must produce 0.
    # This corresponds to the last bit of the rule's binary form.
    if rule_bin[7] != "0":
        return False

    # Condition 2: The rule must be reflection-symmetric.
    # '110' (bit 1) must equal '011' (bit 4).
    # '100' (bit 3) must equal '001' (bit 6).
    is_symmetric = (rule_bin[1] == rule_bin[4]) and (rule_bin[3] == rule_bin[6])

    return is_symmetric


def is_additive(rule_number: int) -> bool:
    """
    Determines if a Wolfram rule is additive.

    A rule is additive if its output can be described by a linear function
    (using XOR as addition) of the three input cells.
    f(p,q,r) = (c1 AND p) XOR (c2 AND q) XOR (c3 AND r) XOR c4

    Args:
        rule_number (int): The elementary cellular automaton rule (0-255).

    Returns:
        bool: True if the rule is additive, False otherwise.
    """
    if not 0 <= rule_number <= 255:
        raise ValueError("Rule number must be between 0 and 255.")

    # An additive rule can be defined by checking if it is in the known set.
    # The 8 additive elementary rules are explicitly identified.
    additive_rules = {0, 60, 90, 102, 150, 170, 204, 240}
    return rule_number in additive_rules


def is_additive_by_test(rule_number: int) -> bool:
    """
    Determines if a Wolfram rule is additive by testing its linear properties.

    An additive rule must be expressible as a linear combination (XOR sum)
    of its three input cells. This function derives the potential linear
    coefficients and verifies if they hold for all 8 neighborhoods.

    Args:
        rule_number (int): The elementary cellular automaton rule (0-255).

    Returns:
        bool: True if the rule is computationally verified as additive, False otherwise.
    """
    if not 0 <= rule_number <= 255:
        raise ValueError("Rule number must be between 0 and 255.")

    # Convert the rule number to its 8-bit binary representation.
    # The bits correspond to neighborhoods from '111' (index 0) down to '000' (index 7).
    rule_bin = format(rule_number, "08b")

    # --- Step 1: Deduce the potential linear coefficients ---
    # If the rule is additive, its behavior on the "basis vectors"
    # reveals the coefficients of the linear function.
    # The output for (1,0,0) gives the left coefficient, (0,1,0) the center, etc.
    # Neighborhood '100' (4) -> bit 3
    # Neighborhood '010' (2) -> bit 5
    # Neighborhood '001' (1) -> bit 6
    w_left = int(rule_bin[3])
    w_center = int(rule_bin[5])
    w_right = int(rule_bin[6])

    # --- Step 2: Verify the derived linear formula for all 8 neighborhoods ---
    for i in range(8):
        # Get the neighborhood (l, c, r) from the index i
        # Example: i=0 -> neighborhood '111'
        l = (i >> 2) & 1
        c = (i >> 1) & 1
        r = i & 1

        # Calculate the expected output based on the linear formula
        expected_output = (w_left & l) ^ (w_center & c) ^ (w_right & r)

        # Get the actual output from the rule's definition
        # The rule's binary string is ordered from '111' down to '000'.
        actual_output = int(rule_bin[7 - i])

        # If the actual output does not match the predicted linear output,
        # the rule is not additive.
        if expected_output != actual_output:
            return False

    # If the formula holds for all 8 neighborhoods, the rule is additive.
    return True
