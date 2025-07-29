#!/usr/bin/env python3
"""
Example Script for Quantum Simulator Comparison

This script demonstrates how to use the quantum simulator comparison tool
with a small subset of circuits for quick testing.
"""

import os
import glob
import re
from quantum_simulator_comparison import QuantumSimulatorComparison

def main():
    """Run example comparison with a few circuits"""
    print("Quantum Simulator Comparison - Example Run")
    print("=" * 50)
    
    # Create comparison object
    comparison = QuantumSimulatorComparison()
    
        # Find QASM files (sorted by difficulty)
    qasm_files = comparison.find_qasm_files()

    if not qasm_files:
        print("No QASM files found in sample_circuits directory")
        return

    print(f"Found {len(qasm_files)} QASM files")
    print("Files are sorted by increasing difficulty level")

    # Find the simple circuit specifically
    simple_circuit = None
    for file in qasm_files:
        if "peaked_circuit_diff=2.500_PUBLIC_98054206" in file:
            simple_circuit = file
            break
    
    if not simple_circuit:
        print("Simple circuit not found, using first file instead")
        simple_circuit = qasm_files[0]
    
    # Test with only the simple circuit
    test_files = [simple_circuit]
    print(f"Testing with simple circuit:")
    filename = os.path.basename(simple_circuit)
    difficulty_match = re.search(r'diff=([\d.]+)', filename)
    difficulty = difficulty_match.group(1) if difficulty_match else "unknown"
    print(f"  {filename} (difficulty: {difficulty})")
    
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
            print(f"  Qiskit (GPU) time: {result['qiskit_gpu_time']:.4f}s" if result['qiskit_gpu_time'] else "  Qiskit (GPU) time: N/A")
            print(f"  QsimCirq time: {result['qsimcirq_time']:.4f}s" if result['qsimcirq_time'] else "  QsimCirq time: N/A")
            
            # Print expected values if available
            if result.get('expected_data'):
                expected = result['expected_data']
                print(f"  Target State: {expected.get('target_state', 'N/A')}")
                print(f"  Peak Probability: {expected.get('peak_prob', 'N/A'):.2e}")
                print(f"  Difficulty Level: {expected.get('difficulty_level', 'N/A')}")
            
            # Print target accuracy if available
            if result.get('target_accuracies'):
                for key, value in result['target_accuracies'].items():
                    if value is not None:
                        print(f"  Target Accuracy {key}: {value:.8f}")
            
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