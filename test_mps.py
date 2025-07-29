#!/usr/bin/env python3
"""
Test MPS Simulator

This script tests the Matrix Product State (MPS) simulator functionality
and compares it with the standard statevector simulator.
"""

import time
from quantum_simulator_comparison import QuantumSimulatorComparison

def test_mps_vs_statevector():
    """Test MPS vs Statevector simulators"""
    print("Testing MPS vs Statevector Simulators")
    print("=" * 50)
    
    # Create comparison objects
    comparison_with_mps = QuantumSimulatorComparison(enable_mps=True)
    comparison_without_mps = QuantumSimulatorComparison(enable_mps=False)
    
    # Find QASM files
    qasm_files = comparison_with_mps.find_qasm_files()
    
    if not qasm_files:
        print("No QASM files found")
        return
    
    # Test with first file
    test_file = qasm_files[0]
    print(f"Testing with: {test_file}")
    
    # Test with MPS enabled
    print("\n1. Testing with MPS enabled:")
    result_with_mps = comparison_with_mps.run_comparison(test_file)
    
    if result_with_mps:
        print(f"  Qiskit (Statevector) time: {result_with_mps['qiskit_time']:.4f}s")
        print(f"  Qiskit (MPS) time: {result_with_mps['qiskit_mps_time']:.4f}s")
        
        if result_with_mps['qiskit_mps_time'] and result_with_mps['qiskit_time']:
            speedup = result_with_mps['qiskit_time'] / result_with_mps['qiskit_mps_time']
            print(f"  MPS speedup: {speedup:.2f}x")
        
        # Check fidelity
        if result_with_mps['fidelities'] and 'qiskit_qiskit_mps' in result_with_mps['fidelities']:
            fidelity = result_with_mps['fidelities']['qiskit_qiskit_mps']
            print(f"  Fidelity (Statevector vs MPS): {fidelity:.8f}")
    
    # Test with MPS disabled
    print("\n2. Testing with MPS disabled:")
    result_without_mps = comparison_without_mps.run_comparison(test_file)
    
    if result_without_mps:
        print(f"  Qiskit (Statevector) time: {result_without_mps['qiskit_time']:.4f}s")
        print(f"  Qiskit (MPS) time: {result_without_mps['qiskit_mps_time']}")
    
    print("\nMPS test completed!")

def test_mps_performance():
    """Test MPS performance on different circuit sizes"""
    print("\nTesting MPS Performance on Different Circuits")
    print("=" * 50)
    
    comparison = QuantumSimulatorComparison(enable_mps=True)
    qasm_files = comparison.find_qasm_files()
    
    if len(qasm_files) < 3:
        print("Need at least 3 QASM files for performance testing")
        return
    
    # Test with first 3 files
    test_files = qasm_files[:3]
    
    results = []
    for i, qasm_file in enumerate(test_files):
        print(f"\nCircuit {i+1}: {qasm_file}")
        
        result = comparison.run_comparison(qasm_file)
        if result:
            results.append({
                'file': qasm_file,
                'qubits': result['num_qubits'],
                'gates': result['num_gates'],
                'statevector_time': result['qiskit_time'],
                'mps_time': result['qiskit_mps_time'],
                'speedup': result['qiskit_time'] / result['qiskit_mps_time'] if result['qiskit_mps_time'] else None
            })
    
    # Print summary
    print("\nPerformance Summary:")
    print("-" * 30)
    for result in results:
        print(f"Circuit: {result['qubits']} qubits, {result['gates']} gates")
        print(f"  Statevector: {result['statevector_time']:.4f}s")
        print(f"  MPS: {result['mps_time']:.4f}s")
        if result['speedup']:
            print(f"  Speedup: {result['speedup']:.2f}x")
        print()

if __name__ == "__main__":
    test_mps_vs_statevector()
    test_mps_performance() 