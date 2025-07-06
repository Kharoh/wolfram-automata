from src.properties import is_legal_wolfram, is_additive, is_additive_by_test


def demo_properties():
    """
    Run demonstrations for properties of Wolfram rules.
    This function tests the legality and additivity of various Wolfram rules,
    finds all legal and additive rules, and tests individual rules for additivity.
    It also dynamically finds all additive rules in the range of 0-255.
    """

    # --- Test Cases ---
    print("--- Testing Rules for Legality ---")
    legal_rule_candidate = 90
    illegal_rule_candidate = 30
    print(
        f"Is Rule {legal_rule_candidate} legal? {is_legal_wolfram(legal_rule_candidate)}"
    )
    print(
        f"Is Rule {illegal_rule_candidate} legal? {is_legal_wolfram(illegal_rule_candidate)}"
    )

    print("\n--- Testing Rules for Additivity ---")
    additive_rule = 150
    non_additive_rule = 110
    print(f"Is Rule {additive_rule} additive? {is_additive(additive_rule)}")
    print(f"Is Rule {non_additive_rule} additive? {is_additive(non_additive_rule)}")

    # --- Find all legal and additive rules ---
    all_legal_rules = [r for r in range(256) if is_legal_wolfram(r)]
    all_additive_rules = [r for r in range(256) if is_additive(r)]

    print(f"\nThere are {len(all_legal_rules)} legal rules:")
    print(all_legal_rules)

    print(f"\nThere are {len(all_additive_rules)} additive rules:")
    print(all_additive_rules)

    print("--- Testing Individual Rules for Additivity ---")

    # Test a known additive rule (Rule 90: output = left XOR right)
    rule_90 = 90
    print(f"Is Rule {rule_90} additive? {is_additive_by_test(rule_90)}")

    # Test a known non-additive rule (Rule 30)
    rule_30 = 30
    print(f"Is Rule {rule_30} additive? {is_additive_by_test(rule_30)}")

    # Test another known additive rule (Rule 150: output = left XOR center XOR right)
    rule_150 = 150
    print(f"Is Rule {rule_150} additive? {is_additive_by_test(rule_150)}")

    # Test another known non-additive rule (Rule 110)
    rule_110 = 110
    print(f"Is Rule {rule_110} additive? {is_additive_by_test(rule_110)}")

    # --- Generate the complete list of additive rules using the test ---
    print("\n--- Dynamically Finding All Additive Rules (0-255) ---")
    all_additive_rules = [r for r in range(256) if is_additive_by_test(r)]

    print(f"Found {len(all_additive_rules)} additive rules:")
    print(all_additive_rules)
