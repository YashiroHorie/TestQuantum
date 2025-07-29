#!/usr/bin/env python3
"""
Example Script for Quantum Simulator Comparison

This script demonstrates how to use the quantum simulator comparison tool
with a small subset of circuits for quick testing.
"""

import os
import glob
from quantum_simulator_comparison import QuantumSimulatorComparison

def main():
    """Run example comparison with a few circuits"""
    print("Quantum Simulator Comparison - Example Run")
    print("=" * 50)
    
    # Create comparison object
    comparison = QuantumSimulatorComparison()
    
    # Find QASM files
    qasm_files = comparison.find_qasm_files()
    
    if not qasm_files:
        print("No QASM files found in sample_circuits directory")
        return
    
    print(f"Found {len(qasm_files)} QASM files")
    
    # Select first 3 files for quick testing
    test_files = qasm_files[:3]
    print(f"Testing with {len(test_files)} files:")
    for i, file in enumerate(test_files):
        print(f"  {i+1}. {os.path.basename(file)}")
    
    print("\n" + "="*50)
    print("Running comparisons...")
    print("="*50)
    
    # Run comparisons on selected files
    for i, qasm_file in enumerate(test_files):
        print(f"\n[{i+1}/{len(test_files)}] Processing: {os.path.basename(qasm_file)}")
        
        # Run comparison
        result = comparison.run_comparison(qasm_file)
        
        # Print quick summary
        if result:
            print(f"  Qubits: {result['num_qubits']}")
            print(f"  Gates: {result['num_gates']}")
            print(f"  Qiskit (Statevector) time: {result['qiskit_time']:.4f}s" if result['qiskit_time'] else "  Qiskit (Statevector) time: N/A")
            print(f"  Qiskit (MPS) time: {result['qiskit_mps_time']:.4f}s" if result['qiskit_mps_time'] else "  Qiskit (MPS) time: N/A")
            print(f"  QsimCirq time: {result['qsimcirq_time']:.4f}s" if result['qsimcirq_time'] else "  QsimCirq time: N/A")
            
            # Print fidelity if available
            if result['fidelities']:
                for key, value in result['fidelities'].items():
                    if value is not None:
                        print(f"  Fidelity {key}: {value:.6f}")
    
    # Generate report
    print("\n" + "="*50)
    print("Generating report...")
    print("="*50)
    
    df = comparison.generate_report()
    
    print("\nExample run completed!")
    print("Check the generated CSV files for detailed results.")

if __name__ == "__main__":
    main() 