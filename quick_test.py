#!/usr/bin/env python3
"""
Quick test to isolate the Qiskit simulation timeout issue
"""

import time
import os
from quantum_simulator_comparison import QuantumSimulatorComparison

def test_simple_circuit():
    """Test with the simple 3-qubit circuit"""
    print("Testing with simple 3-qubit circuit...")
    
    # Create comparison object
    comparison = QuantumSimulatorComparison(enable_mps=True, enable_gpu=True)
    
    # Find the simple circuit
    qasm_files = comparison.find_qasm_files()
    simple_circuit = None
    for file in qasm_files:
        if "simple_circuit_diff=-1.000_PUBLIC_simple3" in file:
            simple_circuit = file
            break
    
    if not simple_circuit:
        print("Simple circuit not found, using first file")
        simple_circuit = qasm_files[0] if qasm_files else None
    
    if not simple_circuit:
        print("No QASM files found!")
        return
    
    print(f"Testing with: {os.path.basename(simple_circuit)}")
    
    # Test only Qiskit statevector simulation
    print("\nTesting Qiskit (Statevector) with timeout...")
    start_time = time.time()
    
    try:
        state, sim_time = comparison.run_qiskit_simulation(simple_circuit)
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

def test_timeout_mechanism():
    """Test the timeout mechanism directly"""
    print("\n" + "="*50)
    print("Testing Timeout Mechanism")
    print("="*50)
    
    from quantum_simulator_comparison import run_with_timeout
    
    @run_with_timeout(timeout_seconds=10)  # 10 second timeout for testing
    def slow_operation():
        print("    Starting slow operation...")
        time.sleep(15)  # This should timeout
        print("    Slow operation completed (should not see this)")
        return "success"
    
    print("Testing 10-second timeout on 15-second operation...")
    start_time = time.time()
    result = slow_operation()
    end_time = time.time()
    
    print(f"Result: {result}")
    print(f"Time taken: {end_time - start_time:.2f}s")
    
    if result == (None, None):
        print("✓ Timeout mechanism working correctly!")
    else:
        print("✗ Timeout mechanism failed!")

if __name__ == "__main__":
    print("Quick Timeout Test")
    print("="*50)
    
    # Test timeout mechanism first
    test_timeout_mechanism()
    
    # Test with actual circuit
    test_simple_circuit()
    
    print("\nTest completed!") 