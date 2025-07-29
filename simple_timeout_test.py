#!/usr/bin/env python3
"""
Simple timeout test with manual timeout checking
"""

import time
import os
from quantum_simulator_comparison import QuantumSimulatorComparison

def run_with_manual_timeout(func, timeout_seconds=30):
    """Run a function with manual timeout checking"""
    start_time = time.time()
    
    def check_timeout():
        if time.time() - start_time > timeout_seconds:
            print(f"  ⏰ Manual timeout after {timeout_seconds}s")
            return True
        return False
    
    try:
        # Run the function and check timeout periodically
        result = func()
        
        # Check if we exceeded timeout
        if check_timeout():
            return None, None
            
        return result
        
    except Exception as e:
        print(f"  ✗ Function failed: {e}")
        return None, None

def test_qiskit_with_manual_timeout():
    """Test Qiskit simulation with manual timeout"""
    print("Testing Qiskit with Manual Timeout")
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
    
    # Test Qiskit simulation with manual timeout
    print("\nTesting Qiskit (Statevector) simulation...")
    start_time = time.time()
    
    def qiskit_simulation():
        return comparison._run_qiskit(test_circuit)
    
    result = run_with_manual_timeout(qiskit_simulation, timeout_seconds=30)
    end_time = time.time()
    total_time = end_time - start_time
    
    if result and result[0] is not None:
        state, sim_time = result
        print(f"  ✓ Qiskit completed in {sim_time:.4f}s (total: {total_time:.4f}s)")
    else:
        print(f"  ⏰ Qiskit timed out or failed (total: {total_time:.4f}s)")

def test_simple_timeout():
    """Test simple timeout mechanism"""
    print("\n" + "="*40)
    print("Testing Simple Timeout")
    print("="*40)
    
    def slow_function():
        print("    Starting slow function...")
        time.sleep(35)  # This should timeout
        print("    Slow function completed (should not see this)")
        return "success"
    
    def fast_function():
        print("    Starting fast function...")
        time.sleep(2)
        print("    Fast function completed")
        return "success"
    
    print("1. Testing fast function:")
    result = run_with_manual_timeout(fast_function, timeout_seconds=30)
    print(f"   Result: {result}")
    
    print("\n2. Testing slow function:")
    result = run_with_manual_timeout(slow_function, timeout_seconds=30)
    print(f"   Result: {result}")

if __name__ == "__main__":
    print("Simple Timeout Test")
    print("=" * 40)
    
    # Test simple timeout first
    test_simple_timeout()
    
    # Test with Qiskit
    test_qiskit_with_manual_timeout()
    
    print("\nTest completed!") 