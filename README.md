# Quantum Simulator Comparison Tool

This tool compares different quantum simulators by running QASM files and analyzing their performance in terms of accuracy, statevector representation, and execution time.

## Supported Simulators

- **Qiskit Aer (Statevector)** - IBM's quantum simulator (statevector_simulator)
- **Qiskit Aer (MPS)** - IBM's Matrix Product State simulator (AerSimulator with MPS method)
- **Qiskit Aer (GPU)** - IBM's GPU-accelerated quantum simulator (AerSimulator with GPU device)
- **Quimb** - Tensor network-based quantum simulator
- **QsimCirq** - Google's quantum simulator
- **TensorNetwork** - Tensor network library for quantum computing

**Note**: MPS and GPU simulators are optional and can be enabled/disabled via the `enable_mps` and `enable_gpu` parameters.

## Features

- **Accuracy Comparison**: Calculates fidelity between different simulators
- **Statevector Analysis**: Extracts dominant basis states and amplitudes
- **Performance Metrics**: Measures execution time for each simulator
- **Comprehensive Reporting**: Generates detailed CSV reports and summary statistics
- **QASM Support**: Parses and converts OpenQASM 2.0 files to different formats

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the setup**:
   ```bash
   python test_setup.py
   ```

## Usage

### Basic Usage

Run the comparison tool on all QASM files in the `sample_circuits` directory:

```bash
python quantum_simulator_comparison.py
```

### Programmatic Usage

```python
from quantum_simulator_comparison import QuantumSimulatorComparison

# Create comparison object
comparison = QuantumSimulatorComparison(circuits_dir="sample_circuits")

# Run comparisons on all QASM files
comparison.run_all_comparisons()

# Generate report
df = comparison.generate_report()
```

### Single File Analysis

```python
from quantum_simulator_comparison import QuantumSimulatorComparison

comparison = QuantumSimulatorComparison()

# Analyze a single QASM file
result = comparison.run_comparison("sample_circuits/peaked_circuit/example.qasm")
print(f"Execution time: {result['qiskit_time']:.4f}s")
```

## Output Files

The tool generates several output files:

1. **`quantum_simulator_results.csv`** - Detailed results for each QASM file
2. **`statevector_analysis.csv`** - Analysis of quantum state vectors
3. **Console Report** - Summary statistics and comparisons

## Sample Output

```
QUANTUM SIMULATOR COMPARISON REPORT
================================================================================

TIMING COMPARISON:
----------------------------------------
Qiskit Time:
  Mean: 0.1234s
  Min:  0.0456s
  Max:  0.2345s
  Count: 45/45

Qiskit Mps Time:
  Mean: 0.0987s
  Min:  0.0321s
  Max:  0.1876s
  Count: 42/45

Qsimcirq Time:
  Mean: 0.0987s
  Min:  0.0321s
  Max:  0.1876s
  Count: 42/45

FIDELITY ANALYSIS:
----------------------------------------
Qiskit vs Qiskit Mps:
  Mean fidelity: 0.999987
  Min fidelity:  0.999945
  Max fidelity:  0.999999
  Count: 42

Qiskit vs Qsimcirq:
  Mean fidelity: 0.999987
  Min fidelity:  0.999945
  Max fidelity:  0.999999
  Count: 42

CIRCUIT STATISTICS:
----------------------------------------
Number of qubits:
  Mean: 31.0
  Min:  31
  Max:  31

Number of gates:
  Mean: 1800.5
  Min:  1750
  Max:  1850
```

## File Structure

```
TestQuantum/
├── quantum_simulator_comparison.py  # Main comparison tool
├── qasm_converter.py               # QASM parsing and conversion
├── test_setup.py                   # Dependency testing
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── sample_circuits/                # QASM files directory
│   └── peaked_circuit/            # Sample quantum circuits
└── output/                         # Generated reports (created automatically)
    ├── quantum_simulator_results.csv
    └── statevector_analysis.csv
```

## QASM File Format

The tool supports OpenQASM 2.0 files. Example:

```qasm
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
```

## Supported Gates

The QASM converter supports common quantum gates:

- **Single-qubit gates**: `u3`, `u2`, `u1`, `rx`, `ry`, `rz`, `x`, `y`, `z`, `h`, `s`, `t`
- **Two-qubit gates**: `cx`, `cy`, `cz`, `swap`

## Performance Considerations

- **Memory Usage**: Large circuits (30+ qubits) may require significant memory
- **Execution Time**: Complex circuits can take several minutes to simulate
- **Simulator Limitations**: Some simulators may not support all gate types

## Troubleshooting

### Common Issues

1. **Import Errors**: Run `python test_setup.py` to check dependencies
2. **Memory Errors**: Reduce circuit size or use fewer qubits
3. **Timeout Errors**: Increase timeout limits for large circuits

### Getting Help

1. Check the console output for error messages
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Test with a simple QASM file first

## Contributing

To extend the tool:

1. **Add New Simulators**: Implement conversion in `qasm_converter.py`
2. **Add New Metrics**: Extend the `QuantumSimulatorComparison` class
3. **Improve Parsing**: Enhance the `QASMParser` class

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- IBM Qiskit team for the quantum computing framework
- Google Cirq team for the quantum simulator
- Quimb developers for the tensor network library
- TensorNetwork contributors for the tensor operations library 