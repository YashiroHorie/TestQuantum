#!/usr/bin/env python3
"""
Test script to demonstrate bitstring and accuracy display functionality
"""

import os
import multiprocessing
from quantum_simulator_comparison import QuantumSimulatorComparison

def main():
    print("Bitstring and Accuracy Display Test")
    print("=" * 50)
    
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    # Create comparison object
    comparison = QuantumSimulatorComparison(enable_mps=True, enable_gpu=True)
    
    # Find a circuit to test
    qasm_files = comparison.find_qasm_files()
    if not qasm_files:
        print("No QASM files found!")
        return
    
    # Test with the first circuit
    test_circuit = qasm_files[0]
    print(f"Testing with: {os.path.basename(test_circuit)}")
    
    # Run comparison with bitstring display
    print("\nRunning comparison with bitstring and accuracy display...")
    comparison.run_comparison(test_circuit)
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 