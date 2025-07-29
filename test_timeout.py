#!/usr/bin/env python3
"""
Test Timeout Functionality

This script tests that the 300-second timeout is working correctly
for quantum simulations.
"""

import time
import os
from quantum_simulator_comparison import QuantumSimulatorComparison

def test_timeout_functionality():
    """Test that timeout works correctly"""
    print("Testing Timeout Functionality")
    print("=" * 50)
    
    # Create comparison object
    comparison = QuantumSimulatorComparison()
    
    # Find QASM files
    qasm_files = comparison.find_qasm_files()
    
    if not qasm_files:
        print("No QASM files found")
        return
    
    print(f"Found {len(qasm_files)} QASM files")
    print("Timeout is set to 300 seconds per simulation")
    
    # Test with first file
    test_file = qasm_files[0]
    filename = os.path.basename(test_file)
    difficulty_match = re.search(r'diff=([\d.]+)', filename)
    difficulty = difficulty_match.group(1) if difficulty_match else "unknown"
    
    print(f"\nTesting timeout with: {filename} (difficulty: {difficulty})")
    print("This will run each simulator with a 300-second timeout")
    
    # Run comparison with timeout
    start_time = time.time()
    result = comparison.run_comparison(test_file)
    total_time = time.time() - start_time
    
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    
    if result:
        print("\nResults summary:")
        print(f"  Qiskit time: {result['qiskit_time']:.4f}s" if result['qiskit_time'] else "  Qiskit time: N/A")
        print(f"  Qiskit MPS time: {result['qiskit_mps_time']:.4f}s" if result['qiskit_mps_time'] else "  Qiskit MPS time: N/A")
        print(f"  QsimCirq time: {result['qsimcirq_time']:.4f}s" if result['qsimcirq_time'] else "  QsimCirq time: N/A")
        
        # Check if any simulations timed out
        timeout_occurred = False
        for key, value in result.items():
            if key.endswith('_time') and value is None:
                print(f"  {key}: TIMEOUT (300s exceeded)")
                timeout_occurred = True
        
        if not timeout_occurred:
            print("  ✓ All simulations completed within timeout")
    
    print("\nTimeout test completed!")

def test_timeout_with_multiple_files():
    """Test timeout with multiple files"""
    print("\n" + "=" * 50)
    print("Testing Timeout with Multiple Files")
    print("=" * 50)
    
    comparison = QuantumSimulatorComparison()
    qasm_files = comparison.find_qasm_files()
    
    if len(qasm_files) < 3:
        print("Need at least 3 files for this test")
        return
    
    # Test with first 3 files
    test_files = qasm_files[:3]
    
    print(f"Testing timeout with {len(test_files)} files:")
    for i, file in enumerate(test_files):
        filename = os.path.basename(file)
        difficulty_match = re.search(r'diff=([\d.]+)', filename)
        difficulty = difficulty_match.group(1) if difficulty_match else "unknown"
        print(f"  {i+1}. {filename} (difficulty: {difficulty})")
    
    print("\nProcessing files with 300s timeout each...")
    
    total_start_time = time.time()
    
    for i, qasm_file in enumerate(test_files):
        print(f"\n[{i+1}/{len(test_files)}] Processing: {os.path.basename(qasm_file)}")
        
        file_start_time = time.time()
        result = comparison.run_comparison(qasm_file)
        file_time = time.time() - file_start_time
        
        print(f"  File processing time: {file_time:.2f}s")
        
        if result:
            # Count successful simulations
            successful_sims = 0
            total_sims = 0
            for key, value in result.items():
                if key.endswith('_time'):
                    total_sims += 1
                    if value is not None:
                        successful_sims += 1
            
            print(f"  Successful simulations: {successful_sims}/{total_sims}")
    
    total_time = time.time() - total_start_time
    print(f"\nTotal processing time for all files: {total_time:.2f}s")
    print("Multiple file timeout test completed!")

if __name__ == "__main__":
    import re
    test_timeout_functionality()
    test_timeout_with_multiple_files() 