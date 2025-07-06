# Wolfram Cellular Automata Analysis

This project provides a comprehensive toolkit for analyzing and visualizing Wolfram's elementary cellular automata (ECA) using PyTorch. It includes demonstrations of entropy, statistical metrics, and rule properties for various ECA rules, with plotting and summary outputs.

## Features
- **Metrics Demonstration**: Calculates and visualizes density, sequence density, triangular density, and two-point correlation for a range of Wolfram rules.
- **Entropy Experiment**: Runs multiple random initializations for each rule and computes the entropy of the final state distribution.
- **Rule Properties**: Tests legality and additivity of rules, and dynamically finds all additive rules.
- **Configurable Rule Categories**: Easily select and sample rules from different Wolfram classes (homogeneous, periodic, chaotic, complex).
- **Modern Plotting**: Generates informative plots for automaton evolution and statistical metrics.
- **CUDA Support**: Automatically uses GPU if available for faster computation.

## Project Structure
```
main.py                       # Entry point for running demonstrations
config/rules.py               # Rule categories for demonstration
src/wolfram_ca.py             # WolframCA class (PyTorch implementation)
src/metrics.py                # MetricsCalculator for CA statistics
src/entropy_experiment.py     # Entropy experiment logic
src/plotting.py               # Plotting utilities
src/properties.py             # Rule legality and additivity checks
demos/demo_metrics.py         # Metrics demonstration script
demos/demo_entropy.py         # Entropy demonstration script
demos/demo_properties.py      # Rule properties demonstration script
```

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

Install dependencies with:
```bash
pip install torch numpy matplotlib
```

## Usage
Run the main script to execute demonstrations:
```bash
python main.py
```
By default, the metrics demonstration runs. To run entropy or properties demos, uncomment the relevant lines in `main.py`.

## Customization
- **Rule Categories**: Edit `config/rules.py` to change which rules are included in each category.
- **Parameters**: Adjust size, generations, and number of runs in the demo scripts for different experiments.

## License
This project is provided for educational and research purposes.
