#!/usr/bin/env python3
"""
Test QASM Converter Fixes

This script tests the fixes for Quimb and Cirq conversion errors.
"""

import os
from qasm_converter import QASMConverter

def test_converter_fixes():
    """Test the converter fixes"""
    print("Testing QASM Converter Fixes")
    print("=" * 40)
    
    # Create converter
    converter = QASMConverter()
    
    # Find a QASM file to test with
    sample_dir = "sample_circuits/peaked_circuit"
    if not os.path.exists(sample_dir):
        print(f"Sample directory {sample_dir} not found")
        return
    
    qasm_files = [f for f in os.listdir(sample_dir) if f.endswith('.qasm')]
    if not qasm_files:
        print("No QASM files found")
        return
    
    test_file = os.path.join(sample_dir, qasm_files[0])
    print(f"Testing with: {test_file}")
    
    # Test Qiskit conversion (should work)
    print("\n1. Testing Qiskit conversion:")
    try:
        qiskit_circuit = converter.qasm_to_qiskit(test_file)
        if qiskit_circuit:
            print(f"  ✓ Qiskit conversion successful")
            print(f"  Qubits: {qiskit_circuit.num_qubits}")
            print(f"  Gates: {qiskit_circuit.size()}")
        else:
            print("  ✗ Qiskit conversion failed")
    except Exception as e:
        print(f"  ✗ Qiskit conversion error: {e}")
    
    # Test Cirq conversion (should work now)
    print("\n2. Testing Cirq conversion:")
    try:
        cirq_circuit = converter.qasm_to_cirq(test_file)
        if cirq_circuit:
            print(f"  ✓ Cirq conversion successful")
            print(f"  Qubits: {len(cirq_circuit.all_qubits())}")
            print(f"  Operations: {len(cirq_circuit.all_operations())}")
        else:
            print("  ✗ Cirq conversion failed")
    except Exception as e:
        print(f"  ✗ Cirq conversion error: {e}")
    
    # Test Quimb conversion (should work now)
    print("\n3. Testing Quimb conversion:")
    try:
        quimb_circuit = converter.qasm_to_quimb(test_file)
        if quimb_circuit:
            print(f"  ✓ Quimb conversion successful")
            print(f"  Qubits: {quimb_circuit['num_qubits']}")
            print(f"  Instructions: {len(quimb_circuit['instructions'])}")
        else:
            print("  ✗ Quimb conversion failed")
    except Exception as e:
        print(f"  ✗ Quimb conversion error: {e}")
    
    # Test TensorNetwork conversion
    print("\n4. Testing TensorNetwork conversion:")
    try:
        tn_circuit = converter.qasm_to_tensornetwork(test_file)
        if tn_circuit:
            print(f"  ✓ TensorNetwork conversion successful")
            print(f"  Tensors: {len(tn_circuit)}")
        else:
            print("  ✗ TensorNetwork conversion failed")
    except Exception as e:
        print(f"  ✗ TensorNetwork conversion error: {e}")
    
    print("\nConverter fixes test completed!")

if __name__ == "__main__":
    test_converter_fixes() 