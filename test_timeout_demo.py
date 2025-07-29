#!/usr/bin/env python3
"""
Timeout Demonstration Script

This script demonstrates how the 30-second timeout works for each simulator.
It shows that simulators are stopped and skipped if they take longer than 30 seconds.
"""

import time
import os
from quantum_simulator_comparison import QuantumSimulatorComparison

def create_slow_circuit():
    """Create a circuit that might be slow to simulate"""
    qasm_content = """OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];
creg c[5];

// Create a circuit with many gates that might be slow
// This is designed to potentially trigger timeouts

// Apply many gates to create computational load
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];

// Many CNOT operations
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[0];

// More rotations
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[4];

// More CNOTs
cx q[0],q[2];
cx q[1],q[3];
cx q[2],q[4];
cx q[3],q[0];
cx q[4],q[1];

// Final measurements
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
"""
    
    # Create test circuit file
    test_file = "sample_circuits/peaked_circuit/test_timeout_circuit.qasm"
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    with open(test_file, 'w') as f:
        f.write(qasm_content)
    
    return test_file

def test_timeout_behavior():
    """Test the timeout behavior for each simulator"""
    print("Timeout Behavior Test")
    print("=" * 50)
    print("This test demonstrates how simulators are stopped and skipped")
    print("if they take longer than 30 seconds.")
    print()
    
    # Create comparison object
    comparison = QuantumSimulatorComparison(enable_mps=True, enable_gpu=True)
    
    # Use the simple circuit for testing
    qasm_files = comparison.find_qasm_files()
    if not qasm_files:
        print("No QASM files found. Creating a test circuit...")
        test_file = create_slow_circuit()
        qasm_files = [test_file]
    
    # Use the first available circuit
    test_circuit = qasm_files[0]
    print(f"Testing with: {os.path.basename(test_circuit)}")
    print()
    
    # Test each simulator individually
    simulators = [
        ("Qiskit (Statevector)", comparison.run_qiskit_simulation),
        ("Qiskit (MPS)", comparison.run_qiskit_mps_simulation),
        ("Qiskit (GPU)", comparison.run_qiskit_gpu_simulation),
        ("Quimb", comparison.run_quimb_simulation),
        ("QsimCirq", comparison.run_qsimcirq_simulation),
        ("TensorNetwork", comparison.run_tensornetwork_simulation)
    ]
    
    results = {}
    
    for name, simulator_func in simulators:
        print(f"Testing {name}...")
        start_time = time.time()
        
        try:
            state, sim_time = simulator_func(test_circuit)
            end_time = time.time()
            total_time = end_time - start_time
            
            if state is not None and sim_time is not None:
                print(f"  ✓ {name} completed in {sim_time:.4f}s (total: {total_time:.4f}s)")
                results[name] = {"status": "success", "time": sim_time, "total_time": total_time}
            else:
                print(f"  ⏰ {name} timed out or failed (>30s)")
                results[name] = {"status": "timeout", "time": None, "total_time": total_time}
                
        except Exception as e:
            end_time = time.time()
            total_time = end_time - start_time
            print(f"  ✗ {name} failed with error: {e}")
            results[name] = {"status": "error", "time": None, "total_time": total_time}
        
        print()
    
    # Summary
    print("=" * 50)
    print("TIMEOUT TEST SUMMARY:")
    print("=" * 50)
    
    successful = 0
    timed_out = 0
    failed = 0
    
    for name, result in results.items():
        status = result["status"]
        if status == "success":
            successful += 1
            print(f"✓ {name}: {result['time']:.4f}s")
        elif status == "timeout":
            timed_out += 1
            print(f"⏰ {name}: TIMED OUT (>30s)")
        else:
            failed += 1
            print(f"✗ {name}: FAILED")
    
    print()
    print(f"Successful: {successful}")
    print(f"Timed out: {timed_out}")
    print(f"Failed: {failed}")
    print(f"Total: {len(results)}")
    
    # Clean up test file if created
    if "test_timeout_circuit.qasm" in test_circuit:
        try:
            os.remove(test_circuit)
            print(f"\nCleaned up test file: {test_circuit}")
        except:
            pass

def demonstrate_timeout_mechanism():
    """Demonstrate the timeout mechanism with a simple example"""
    print("Timeout Mechanism Demonstration")
    print("=" * 50)
    print()
    print("How the 30-second timeout works:")
    print()
    print("1. Each simulator has a 30-second timeout decorator:")
    print("   @run_with_timeout(timeout_seconds=30)")
    print("   def _run_simulator(self, qasm_file):")
    print()
    print("2. If a simulator takes >30 seconds:")
    print("   - Signal SIGALRM is triggered")
    print("   - TimeoutError is raised")
    print("   - Simulator is stopped immediately")
    print("   - Function returns (None, None)")
    print("   - Tool continues with next simulator")
    print()
    print("3. Expected output for timeout:")
    print("   ⏰ SimulatorName simulation timed out (>30s), skipping...")
    print("   ✗ SimulatorName simulation failed")
    print()
    print("4. Expected output for success:")
    print("   ✓ SimulatorName simulation completed in 0.1234s")
    print()

if __name__ == "__main__":
    print("Quantum Simulator Timeout Test")
    print("=" * 50)
    
    # Demonstrate the mechanism
    demonstrate_timeout_mechanism()
    
    # Run the actual test
    test_timeout_behavior()
    
    print("\nTimeout test completed!")
    print("All simulators should complete within 30 seconds or be skipped.") 