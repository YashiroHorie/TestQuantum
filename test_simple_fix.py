#!/usr/bin/env python3
"""
Simple test to verify the direct timeout implementation
"""

import time
import os
from quantum_simulator_comparison import QuantumSimulatorComparison

def main():
    print("Simple Timeout Test")
    print("=" * 40)
    
    # Create comparison object
    comparison = QuantumSimulatorComparison(enable_mps=True, enable_gpu=True)
    
    # Find a circuit to test
    qasm_files = comparison.find_qasm_files()
    if not qasm_files:
        print("No QASM files found!")
        return
    
    test_circuit = qasm_files[0]
    print(f"Testing with: {os.path.basename(test_circuit)}")
    
    # Test Qiskit simulation with timeout
    print("\nTesting Qiskit (Statevector) simulation...")
    start_time = time.time()
    
    try:
        state, sim_time = comparison.run_qiskit_simulation(test_circuit)
        end_time = time.time()
        total_time = end_time - start_time
        
        if state is not None and sim_time is not None:
            print(f"  ✓ Qiskit completed in {sim_time:.4f}s (total: {total_time:.4f}s)")
        else:
            print(f"  ⏰ Qiskit timed out or failed (total: {total_time:.4f}s)")
            
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"  ✗ Qiskit failed with error: {e} (total: {total_time:.4f}s)")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 