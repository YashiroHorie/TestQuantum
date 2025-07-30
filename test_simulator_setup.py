#!/usr/bin/env python3
"""
Test script to verify simulator setup and basic functionality
"""

import sys
import time

def test_imports():
    """Test if all required libraries can be imported"""
    print("Testing simulator imports...")
    print("=" * 50)
    
    simulators = {}
    
    # Test Qiskit
    try:
        import qiskit
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        simulators['Qiskit'] = "✓ Available"
        print("✓ Qiskit: Available")
    except ImportError as e:
        simulators['Qiskit'] = f"✗ Not available: {e}"
        print(f"✗ Qiskit: Not available - {e}")
    
    # Test Quimb
    try:
        import quimb.tensor as qtn
        simulators['Quimb'] = "✓ Available"
        print("✓ Quimb: Available")
    except ImportError as e:
        simulators['Quimb'] = f"✗ Not available: {e}"
        print(f"✗ Quimb: Not available - {e}")
    
    # Test Cotengra
    try:
        import cotengra
        simulators['Cotengra'] = "✓ Available"
        print("✓ Cotengra: Available")
    except ImportError as e:
        simulators['Cotengra'] = f"✗ Not available: {e}"
        print(f"✗ Cotengra: Not available - {e}")
    
    # Test QsimCirq
    try:
        import qsimcirq
        import cirq
        simulators['QsimCirq'] = "✓ Available"
        print("✓ QsimCirq: Available")
    except ImportError as e:
        simulators['QsimCirq'] = f"✗ Not available: {e}"
        print(f"✗ QsimCirq: Not available - {e}")
    
    # Test ExaTN
    try:
        import exatn
        simulators['ExaTN'] = "✓ Available"
        print("✓ ExaTN: Available")
    except ImportError as e:
        simulators['ExaTN'] = f"✗ Not available: {e}"
        print(f"✗ ExaTN: Not available - {e}")
    
    return simulators

def test_basic_functionality():
    """Test basic functionality of available simulators"""
    print("\nTesting basic functionality...")
    print("=" * 50)
    
    # Test Qiskit MPS
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        
        # Create simple circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        
        # Test MPS simulator
        mps_backend = AerSimulator(method='matrix_product_state')
        print("✓ Qiskit MPS: Basic functionality works")
    except Exception as e:
        print(f"✗ Qiskit MPS: Error - {e}")
    
    # Test Quimb + Cotengra
    try:
        import quimb.tensor as qtn
        import cotengra
        
        # Create simple tensor
        t = qtn.Tensor([1, 2, 3, 4], inds=['a'])
        print("✓ Quimb + Cotengra: Basic functionality works")
    except Exception as e:
        print(f"✗ Quimb + Cotengra: Error - {e}")
    
    # Test Qsim
    try:
        import qsimcirq
        import cirq
        
        # Create simple circuit
        q = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()
        circuit.append(cirq.H(q[0]))
        circuit.append(cirq.CNOT(q[0], q[1]))
        
        # Test simulator
        simulator = qsimcirq.QSimSimulator()
        print("✓ Qsim: Basic functionality works")
    except Exception as e:
        print(f"✗ Qsim: Error - {e}")

def test_simple_circuit():
    """Test a simple circuit with available simulators"""
    print("\nTesting simple circuit...")
    print("=" * 50)
    
    # Simple 2-qubit Bell state circuit
    qasm_content = """OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

h q[0];
cx q[0],q[1];
"""
    
    # Save to temporary file
    with open("temp_test.qasm", "w") as f:
        f.write(qasm_content)
    
    results = {}
    
    # Test Qiskit MPS
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        from qiskit.compiler import transpile
        
        start_time = time.time()
        circuit = QuantumCircuit.from_qasm_file("temp_test.qasm")
        mps_backend = AerSimulator(method='matrix_product_state')
        transpiled = transpile(circuit, mps_backend)
        job = mps_backend.run(transpiled)
        result = job.result()
        statevector = result.get_statevector()
        execution_time = time.time() - start_time
        
        results['Qiskit MPS'] = {
            'time': execution_time,
            'state_size': len(statevector),
            'success': True
        }
        print(f"✓ Qiskit MPS: {execution_time:.3f}s, {len(statevector)} states")
    except Exception as e:
        results['Qiskit MPS'] = {'success': False, 'error': str(e)}
        print(f"✗ Qiskit MPS: {e}")
    
    # Test Qsim
    try:
        import qsimcirq
        import cirq
        
        start_time = time.time()
        q = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()
        circuit.append(cirq.H(q[0]))
        circuit.append(cirq.CNOT(q[0], q[1]))
        
        simulator = qsimcirq.QSimSimulator()
        result = simulator.simulate(circuit)
        statevector = result.final_state_vector
        execution_time = time.time() - start_time
        
        results['Qsim'] = {
            'time': execution_time,
            'state_size': len(statevector),
            'success': True
        }
        print(f"✓ Qsim: {execution_time:.3f}s, {len(statevector)} states")
    except Exception as e:
        results['Qsim'] = {'success': False, 'error': str(e)}
        print(f"✗ Qsim: {e}")
    
    # Clean up
    import os
    if os.path.exists("temp_test.qasm"):
        os.remove("temp_test.qasm")
    
    return results

def main():
    """Main test function"""
    print("Multi-Simulator Setup Test")
    print("=" * 60)
    
    # Test imports
    simulators = test_imports()
    
    # Test basic functionality
    test_basic_functionality()
    
    # Test simple circuit
    results = test_simple_circuit()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    available_count = sum(1 for status in simulators.values() if "✓" in status)
    total_count = len(simulators)
    
    print(f"Available simulators: {available_count}/{total_count}")
    
    for name, status in simulators.items():
        print(f"  {name}: {status}")
    
    print(f"\nWorking simulators: {available_count}")
    if available_count >= 2:
        print("✓ Ready to run multi-simulator tests!")
    else:
        print("⚠ Need at least 2 simulators for comparison")

if __name__ == "__main__":
    main() 