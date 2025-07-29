#!/usr/bin/env python3
"""
Test Qiskit Imports

This script tests that Qiskit imports work correctly with version 1.4.3.
"""

def test_qiskit_imports():
    """Test Qiskit imports"""
    print("Testing Qiskit Imports")
    print("=" * 30)
    
    try:
        import qiskit
        print(f"✓ Qiskit version: {qiskit.__version__}")
    except ImportError as e:
        print(f"✗ Qiskit import failed: {e}")
        return False
    
    try:
        from qiskit import QuantumCircuit
        from qiskit.compiler import transpile
        print("✓ QuantumCircuit and transpile imported")
    except ImportError as e:
        print(f"✗ QuantumCircuit/transpile import failed: {e}")
        return False
    
    try:
        from qiskit.quantum_info import Statevector, state_fidelity
        print("✓ Statevector and state_fidelity imported")
    except ImportError as e:
        print(f"✗ Statevector/state_fidelity import failed: {e}")
        return False
    
    try:
        from qiskit_aer import Aer
        print("✓ Aer imported from qiskit_aer")
    except ImportError as e:
        print(f"✗ Aer import failed: {e}")
        return False
    
    try:
        from qiskit_aer import AerSimulator
        print("✓ AerSimulator imported")
    except ImportError as e:
        print(f"✗ AerSimulator import failed: {e}")
        return False
    
    return True

def test_simple_circuit():
    """Test creating and running a simple circuit"""
    print("\nTesting Simple Circuit")
    print("=" * 30)
    
    try:
        from qiskit import QuantumCircuit
        from qiskit.compiler import transpile
        from qiskit_aer import Aer
        
        # Create a simple 2-qubit circuit
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
        
        print("✓ Circuit created successfully")
        print(f"  Qubits: {circuit.num_qubits}")
        print(f"  Gates: {circuit.size()}")
        
        # Run simulation
        backend = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(circuit, backend)
        job = backend.run(transpiled_circuit, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        print("✓ Simulation completed successfully")
        print(f"  Results: {counts}")
        
        return True
        
    except Exception as e:
        print(f"✗ Circuit test failed: {e}")
        return False

def test_statevector_simulation():
    """Test statevector simulation"""
    print("\nTesting Statevector Simulation")
    print("=" * 30)
    
    try:
        from qiskit import QuantumCircuit
        from qiskit.compiler import transpile
        from qiskit_aer import Aer
        
        # Create a simple circuit
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        
        # Run statevector simulation
        backend = Aer.get_backend('statevector_simulator')
        transpiled_circuit = transpile(circuit, backend)
        job = backend.run(transpiled_circuit)
        result = job.result()
        statevector = result.get_statevector()
        
        # Handle new Statevector object type
        if hasattr(statevector, 'data'):
            statevector_array = statevector.data
        else:
            statevector_array = statevector
        
        print("✓ Statevector simulation completed")
        print(f"  Statevector type: {type(statevector)}")
        print(f"  Statevector shape: {statevector_array.shape}")
        print(f"  First few amplitudes: {statevector_array[:4]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Statevector test failed: {e}")
        return False

if __name__ == "__main__":
    print("Qiskit Import and Functionality Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_qiskit_imports()
    
    if imports_ok:
        # Test simple circuit
        circuit_ok = test_simple_circuit()
        
        # Test statevector simulation
        statevector_ok = test_statevector_simulation()
        
        if circuit_ok and statevector_ok:
            print("\n🎉 All tests passed! Qiskit is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Check the errors above.")
    else:
        print("\n❌ Import tests failed. Qiskit is not properly installed.") 