#!/usr/bin/env python3
"""
Test script to verify the multiprocessing-based timeout works correctly
"""

import time
import multiprocessing
from quantum_simulator_comparison import run_with_timeout

def slow_function():
    """A function that takes a long time"""
    print("    Starting slow function...")
    time.sleep(35)  # This should timeout after 30 seconds
    print("    Slow function completed (this should not print)")
    return "success"

def fast_function():
    """A function that completes quickly"""
    print("    Starting fast function...")
    time.sleep(2)
    print("    Fast function completed")
    return "success"

@run_with_timeout(timeout_seconds=30)
def test_slow():
    return slow_function()

@run_with_timeout(timeout_seconds=30)
def test_fast():
    return fast_function()

def test_qiskit_simulation():
    """Test with actual Qiskit simulation"""
    print("\n" + "="*50)
    print("Testing Qiskit Simulation with Timeout")
    print("="*50)
    
    from quantum_simulator_comparison import QuantumSimulatorComparison
    
    # Create comparison object
    comparison = QuantumSimulatorComparison(enable_mps=True, enable_gpu=True)
    
    # Find a circuit to test
    qasm_files = comparison.find_qasm_files()
    if not qasm_files:
        print("No QASM files found!")
        return
    
    test_circuit = qasm_files[0]
    print(f"Testing with: {test_circuit}")
    
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

def main():
    print("Testing Multiprocessing-Based Timeout")
    print("=" * 50)
    
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    print("\n1. Testing fast function (should complete):")
    start_time = time.time()
    result = test_fast()
    end_time = time.time()
    print(f"   Result: {result}")
    print(f"   Time taken: {end_time - start_time:.2f}s")
    
    print("\n2. Testing slow function (should timeout):")
    start_time = time.time()
    result = test_slow()
    end_time = time.time()
    print(f"   Result: {result}")
    print(f"   Time taken: {end_time - start_time:.2f}s")
    
    # Test with actual Qiskit simulation
    test_qiskit_simulation()
    
    print("\nTimeout test completed!")
    print("If the slow function timed out correctly, you should see:")
    print("- '⏰ Timeout after 30s - process terminated' message")
    print("- Result should be (None, None)")
    print("- Total time should be around 30 seconds")

if __name__ == "__main__":
    main() 