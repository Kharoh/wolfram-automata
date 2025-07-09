from demos.demo_entropy import demo_entropy
from demos.demo_metrics import demo_metrics
from demos.demo_properties import demo_properties
from demos.demo_correlation import demo_correlation
from demos.demo_lempel_ziv import demo_lempel_ziv

from config.rules import RULE_CATEGORIES

from src.wolfram_ca import WolframCA

if __name__ == "__main__":
    # Run the metrics demonstration
    # demo_metrics()

    # Run the entropy demonstration
    # demo_entropy()

    # Run the correlation demonstration
    # demo_correlation()

    # Run the Lempel-Ziv complexity demonstration
    # demo_lempel_ziv()

    # Run the properties demonstration
    # demo_properties()

    for rule_category, rules in RULE_CATEGORIES.items():
        print(f"{rule_category}: {rules}")

        for rule in rules:
            print(f"  Processing Rule {rule}")

            size = 13
            generations = 300
            runs = 5000
            n_first = 13
            n_last = 13

            ca = WolframCA(
                rule_number=rule,
                size=size,
                initial_condition="single",
                device="cpu",
            )

            ca.run_experiments_to_csv(
                generations=generations,
                runs=runs,
                n_first=n_first,
                n_last=n_last,
                initial_condition="random",
                output_dir="experiments",
            )

            ca.run_analysis_to_md(
                input_csv=f"experiments/{rule}_random_s{size}_g{generations}_r{runs}_f{n_first}_l{n_last}.csv",
                output_dir="experiments",
            )
