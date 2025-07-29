#!/usr/bin/env python3
"""
Test script using quimb.tensor to analyze peaked circuits.
This script loads a QASM circuit, converts it to MPS, and samples to find peak bitstrings.
"""

import quimb.tensor as qtn
from collections import Counter
import time
import os

def test_peaked_circuit_analysis():
    """Test analysis of peaked circuit using quimb.tensor"""
    
    # Define the QASM file path
    qasm_file = "sample_circuits/peaked_circuit/peaked_circuit_diff=0.000_PUBLIC_58b8244b.qasm"
    
    # Check if file exists
    if not os.path.exists(qasm_file):
        print(f"Error: QASM file not found: {qasm_file}")
        return
    
    print(f"Loading QASM circuit from: {qasm_file}")
    print("=" * 60)
    
    try:
        # Load QASM circuit
        start_time = time.time()
        qc = qtn.Circuit.from_qasm_file(qasm_file)
        load_time = time.time() - start_time
        print(f"Circuit loaded successfully in {load_time:.2f} seconds")
        print(f"Circuit has {qc.num_qubits} qubits")
        
        # Convert to MPS (Matrix Product State)
        print("\nConverting to MPS...")
        start_time = time.time()
        psi = qc.psi
        mps_time = time.time() - start_time
        print(f"MPS conversion completed in {mps_time:.2f} seconds")
        
        # Sample from the circuit
        num_shots = 10_000_000
        print(f"\nSampling {num_shots:,} shots from the circuit...")
        start_time = time.time()
        samples = psi.sample(num_shots)
        sample_time = time.time() - start_time
        print(f"Sampling completed in {sample_time:.2f} seconds")
        
        # Count the most frequent bitstring
        print("\nAnalyzing results...")
        start_time = time.time()
        counter = Counter(samples)
        analysis_time = time.time() - start_time
        print(f"Analysis completed in {analysis_time:.2f} seconds")
        
        # Get the peak state
        peak_state, count = counter.most_common(1)[0]
        estimated_prob = count / num_shots
        
        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)
        print(f"Peak bitstring: {peak_state}")
        print(f"Count: {count:,}")
        print(f"Estimated probability: {estimated_prob:.8f} ({estimated_prob*100:.6f}%)")
        
        # Show top 5 most frequent bitstrings
        print(f"\nTop 5 most frequent bitstrings:")
        print("-" * 40)
        for i, (bitstring, freq) in enumerate(counter.most_common(5), 1):
            prob = freq / num_shots
            print(f"{i}. {bitstring}: {freq:,} times ({prob:.8f} = {prob*100:.6f}%)")
        
        # Calculate total time
        total_time = load_time + mps_time + sample_time + analysis_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        
        return {
            'peak_state': peak_state,
            'count': count,
            'estimated_prob': estimated_prob,
            'total_time': total_time,
            'num_shots': num_shots
        }
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_multiple_circuits():
    """Test multiple circuits with different difficulty levels"""
    
    # Test a few circuits with different difficulty levels
    test_circuits = [
        "sample_circuits/peaked_circuit/peaked_circuit_diff=0.000_PUBLIC_58b8244b.qasm",
        "sample_circuits/peaked_circuit/peaked_circuit_diff=0.100_PUBLIC_0f894320.qasm",
        "sample_circuits/peaked_circuit/peaked_circuit_diff=0.500_PUBLIC_04ea9dc5.qasm",
        "sample_circuits/peaked_circuit/simple_circuit_diff=-1.000_PUBLIC_simple3.qasm"
    ]
    
    results = {}
    
    for circuit_file in test_circuits:
        if os.path.exists(circuit_file):
            print(f"\n{'='*80}")
            print(f"Testing circuit: {os.path.basename(circuit_file)}")
            print(f"{'='*80}")
            
            # Extract difficulty from filename
            filename = os.path.basename(circuit_file)
            if 'diff=' in filename:
                difficulty = filename.split('diff=')[1].split('_')[0]
                print(f"Difficulty: {difficulty}")
            
            result = test_peaked_circuit_analysis()
            if result:
                results[circuit_file] = result
        else:
            print(f"Circuit file not found: {circuit_file}")
    
    return results

if __name__ == "__main__":
    print("Quimb Tensor Peaked Circuit Analysis Test")
    print("=" * 60)
    
    # Test single circuit
    result = test_peaked_circuit_analysis()
    
    if result:
        print(f"\n{'='*60}")
        print("SUMMARY:")
        print(f"{'='*60}")
        print(f"Successfully analyzed peaked circuit")
        print(f"Peak state found: {result['peak_state']}")
        print(f"Probability: {result['estimated_prob']:.8f}")
        print(f"Total time: {result['total_time']:.2f} seconds")
    
    # Uncomment the following line to test multiple circuits
    # test_multiple_circuits() 