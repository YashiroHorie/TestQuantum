#!/usr/bin/env python3
"""
Test Setup Script

This script tests that all required dependencies are properly installed
and can be imported successfully.
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            module = importlib.import_module(package_name)
        else:
            module = importlib.import_module(module_name)
        print(f"✓ {module_name} - OK")
        return True
    except ImportError as e:
        print(f"✗ {module_name} - FAILED: {e}")
        return False

def main():
    """Test all required dependencies"""
    print("Testing Quantum Simulator Dependencies")
    print("=" * 50)
    
    # Core dependencies
    print("\nCore Dependencies:")
    core_modules = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm")
    ]
    
    core_success = 0
    for name, module in core_modules:
        if test_import(name, module):
            core_success += 1
    
    # Quantum computing libraries
    print("\nQuantum Computing Libraries:")
    quantum_modules = [
        ("qiskit", "qiskit"),
        ("qiskit-aer", "qiskit_aer"),
        ("quimb", "quimb"),
        ("cirq", "cirq"),
        ("qsimcirq", "qsimcirq"),
        ("tensornetwork", "tensornetwork")
    ]
    
    quantum_success = 0
    for name, module in quantum_modules:
        if test_import(name, module):
            quantum_success += 1
    
    # Test QASM converter
    print("\nCustom Modules:")
    custom_modules = [
        ("qasm_converter", "qasm_converter"),
        ("quantum_simulator_comparison", "quantum_simulator_comparison")
    ]
    
    custom_success = 0
    for name, module in custom_modules:
        if test_import(name, module):
            custom_success += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Core dependencies: {core_success}/{len(core_modules)}")
    print(f"Quantum libraries: {quantum_success}/{len(quantum_modules)}")
    print(f"Custom modules: {custom_success}/{len(custom_modules)}")
    
    total_success = core_success + quantum_success + custom_success
    total_modules = len(core_modules) + len(quantum_modules) + len(custom_modules)
    
    print(f"\nOverall: {total_success}/{total_modules} modules available")
    
    if total_success == total_modules:
        print("🎉 All dependencies are properly installed!")
    else:
        print("⚠️  Some dependencies are missing. Please install them using:")
        print("   pip install -r requirements.txt")
    
    # Test basic functionality
    print("\n" + "=" * 50)
    print("Testing Basic Functionality:")
    
    # Test Qiskit
    try:
        import qiskit
        from qiskit import QuantumCircuit, Aer
        print("✓ Qiskit basic functionality - OK")
    except Exception as e:
        print(f"✗ Qiskit basic functionality - FAILED: {e}")
    
    # Test QASM converter
    try:
        from qasm_converter import QASMConverter
        converter = QASMConverter()
        print("✓ QASM converter - OK")
    except Exception as e:
        print(f"✗ QASM converter - FAILED: {e}")
    
    # Test main comparison tool
    try:
        from quantum_simulator_comparison import QuantumSimulatorComparison
        comparison = QuantumSimulatorComparison()
        print("✓ Quantum simulator comparison - OK")
    except Exception as e:
        print(f"✗ Quantum simulator comparison - FAILED: {e}")

if __name__ == "__main__":
    main() 