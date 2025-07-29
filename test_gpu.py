#!/usr/bin/env python3
"""
Test script for Qiskit GPU functionality

This script tests the GPU-accelerated quantum simulator and compares
its performance with other simulators.
"""

import time
import numpy as np
from quantum_simulator_comparison import QuantumSimulatorComparison

def test_gpu_functionality():
    """Test GPU functionality with a simple circuit"""
    print("Testing Qiskit GPU Functionality")
    print("=" * 50)
    
    # Create comparison object with GPU enabled
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
        print("No QASM files found for testing")
        return
    
    print(f"Testing with: {simple_circuit}")
    
    # Test individual simulators
    print("\n1. Testing Qiskit (Statevector)...")
    qiskit_state, qiskit_time = comparison.run_qiskit_simulation(simple_circuit)
    print(f"   Time: {qiskit_time:.4f}s" if qiskit_time else "   Failed")
    
    print("\n2. Testing Qiskit (MPS)...")
    qiskit_mps_state, qiskit_mps_time = comparison.run_qiskit_mps_simulation(simple_circuit)
    print(f"   Time: {qiskit_mps_time:.4f}s" if qiskit_mps_time else "   Failed")
    
    print("\n3. Testing Qiskit (GPU)...")
    qiskit_gpu_state, qiskit_gpu_time = comparison.run_qiskit_gpu_simulation(simple_circuit)
    print(f"   Time: {qiskit_gpu_time:.4f}s" if qiskit_gpu_time else "   Failed")
    
    # Compare results
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON:")
    print("=" * 50)
    
    if qiskit_time and qiskit_mps_time and qiskit_gpu_time:
        print(f"Qiskit (Statevector): {qiskit_time:.4f}s")
        print(f"Qiskit (MPS):        {qiskit_mps_time:.4f}s")
        print(f"Qiskit (GPU):        {qiskit_gpu_time:.4f}s")
        
        # Calculate speedup
        if qiskit_time > 0:
            mps_speedup = qiskit_time / qiskit_mps_time
            gpu_speedup = qiskit_time / qiskit_gpu_time
            print(f"\nSpeedup vs Statevector:")
            print(f"MPS: {mps_speedup:.2f}x")
            print(f"GPU: {gpu_speedup:.2f}x")
    
    # Test fidelity
    print("\n" + "=" * 50)
    print("FIDELITY COMPARISON:")
    print("=" * 50)
    
    if qiskit_state is not None and qiskit_mps_state is not None:
        fidelity_mps = comparison.calculate_fidelity(qiskit_state, qiskit_mps_state)
        print(f"Statevector vs MPS: {fidelity_mps:.8f}" if fidelity_mps else "Failed")
    
    if qiskit_state is not None and qiskit_gpu_state is not None:
        fidelity_gpu = comparison.calculate_fidelity(qiskit_state, qiskit_gpu_state)
        print(f"Statevector vs GPU:  {fidelity_gpu:.8f}" if fidelity_gpu else "Failed")
    
    if qiskit_mps_state is not None and qiskit_gpu_state is not None:
        fidelity_mps_gpu = comparison.calculate_fidelity(qiskit_mps_state, qiskit_gpu_state)
        print(f"MPS vs GPU:         {fidelity_mps_gpu:.8f}" if fidelity_mps_gpu else "Failed")
    
    # Test target accuracy
    print("\n" + "=" * 50)
    print("TARGET ACCURACY TEST:")
    print("=" * 50)
    
    expected_data = comparison.get_expected_value(simple_circuit)
    if expected_data and expected_data.get('target_state'):
        target_state = expected_data['target_state']
        print(f"Target state: {target_state}")
        
        if qiskit_state is not None:
            accuracy = comparison.calculate_target_accuracy(qiskit_state, target_state)
            print(f"Statevector accuracy: {accuracy:.8f}" if accuracy else "Failed")
        
        if qiskit_mps_state is not None:
            accuracy = comparison.calculate_target_accuracy(qiskit_mps_state, target_state)
            print(f"MPS accuracy:        {accuracy:.8f}" if accuracy else "Failed")
        
        if qiskit_gpu_state is not None:
            accuracy = comparison.calculate_target_accuracy(qiskit_gpu_state, target_state)
            print(f"GPU accuracy:        {accuracy:.8f}" if accuracy else "Failed")
    
    print("\n" + "=" * 50)
    print("GPU TEST COMPLETED")
    print("=" * 50)

def test_gpu_availability():
    """Test if GPU is available for Qiskit"""
    print("Testing GPU Availability")
    print("=" * 30)
    
    try:
        from qiskit_aer import AerSimulator
        
        # Try to create GPU backend
        gpu_backend = AerSimulator(method='statevector', device='GPU')
        print("✓ GPU backend created successfully")
        
        # Check available devices
        try:
            from qiskit_aer.backends import AerSimulator
            backend = AerSimulator()
            print(f"✓ Available devices: {backend.available_devices()}")
        except Exception as e:
            print(f"⚠ Could not check available devices: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ GPU not available: {e}")
        print("  This could be due to:")
        print("  - No CUDA-compatible GPU")
        print("  - CUDA not installed")
        print("  - qiskit-aer-gpu not installed")
        return False

if __name__ == "__main__":
    print("Qiskit GPU Functionality Test")
    print("=" * 50)
    
    # Test GPU availability first
    gpu_available = test_gpu_availability()
    
    if gpu_available:
        print("\nGPU is available, running functionality test...")
        test_gpu_functionality()
    else:
        print("\nGPU is not available. The tool will still work with CPU simulators.")
        print("To enable GPU support, ensure you have:")
        print("1. CUDA-compatible GPU")
        print("2. CUDA toolkit installed")
        print("3. qiskit-aer-gpu package installed") 