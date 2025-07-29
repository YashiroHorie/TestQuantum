#!/usr/bin/env python3
"""
Direct timeout test - bypassing decorators
"""

import time
import os
import threading
from quantum_simulator_comparison import QuantumSimulatorComparison

class DirectTimeoutComparison(QuantumSimulatorComparison):
    """Comparison class with direct timeout implementation"""
    
    def run_qiskit_simulation_with_timeout(self, qasm_file, timeout_seconds=30):
        """Run Qiskit simulation with direct timeout"""
        if not QISKIT_AVAILABLE:
            return None, None
        
        result = [None]
        exception = [None]
        
        def run_simulation():
            try:
                result[0] = self._run_qiskit(qasm_file)
            except Exception as e:
                exception[0] = e
        
        # Start simulation in a thread
        thread = threading.Thread(target=run_simulation)
        thread.daemon = True
        thread.start()
        
        # Wait for completion or timeout
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            print(f"  ⏰ Qiskit simulation timed out after {timeout_seconds}s")
            return None, None
        
        if exception[0]:
            print(f"  ✗ Qiskit simulation failed: {exception[0]}")
            return None, None
        
        return result[0]

def test_direct_timeout():
    """Test direct timeout implementation"""
    print("Direct Timeout Test")
    print("=" * 40)
    
    # Create comparison object
    comparison = DirectTimeoutComparison(enable_mps=True, enable_gpu=True)
    
    # Find a circuit to test
    qasm_files = comparison.find_qasm_files()
    if not qasm_files:
        print("No QASM files found!")
        return
    
    test_circuit = qasm_files[0]
    print(f"Testing with: {os.path.basename(test_circuit)}")
    
    # Test with different timeout values
    timeouts = [10, 30, 60]  # seconds
    
    for timeout in timeouts:
        print(f"\nTesting with {timeout}s timeout:")
        start_time = time.time()
        
        result = comparison.run_qiskit_simulation_with_timeout(test_circuit, timeout)
        end_time = time.time()
        total_time = end_time - start_time
        
        if result and result[0] is not None:
            state, sim_time = result
            print(f"  ✓ Completed in {sim_time:.4f}s (total: {total_time:.4f}s)")
            break
        else:
            print(f"  ⏰ Timed out or failed (total: {total_time:.4f}s)")

def test_simple_direct_timeout():
    """Test simple direct timeout"""
    print("\n" + "="*40)
    print("Testing Simple Direct Timeout")
    print("="*40)
    
    def slow_function():
        print("    Starting slow function...")
        time.sleep(35)
        print("    Slow function completed (should not see this)")
        return "success"
    
    result = [None]
    exception = [None]
    
    def run_func():
        try:
            result[0] = slow_function()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=run_func)
    thread.daemon = True
    thread.start()
    
    print("Waiting for 30 seconds...")
    thread.join(30)
    
    if thread.is_alive():
        print("  ⏰ Function timed out after 30s")
    else:
        print(f"  ✓ Function completed: {result[0]}")

if __name__ == "__main__":
    print("Direct Timeout Test")
    print("=" * 40)
    
    # Test simple timeout first
    test_simple_direct_timeout()
    
    # Test with Qiskit
    test_direct_timeout()
    
    print("\nTest completed!") 