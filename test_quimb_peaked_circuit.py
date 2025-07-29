#!/usr/bin/env python3
"""
Test script using quimb.tensor to analyze peaked circuits.
This script loads a QASM circuit, converts it to MPS, and samples to find peak bitstrings.
"""

import quimb.tensor as qtn
from collections import Counter
import time
import os
import numpy as np

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
        # Read QASM file content
        with open(qasm_file, 'r') as f:
            qasm_content = f.read()
        
        print("QASM file loaded successfully")
        
        # Parse QASM to get circuit information
        num_qubits = 0
        instructions = []
        
        for line in qasm_content.split('\n'):
            line = line.strip()
            if line.startswith('qreg q['):
                # Extract number of qubits
                num_qubits = int(line.split('[')[1].split(']')[0])
                print(f"Circuit has {num_qubits} qubits")
            elif line.startswith('h q['):
                # Hadamard gate
                qubit = int(line.split('[')[1].split(']')[0])
                instructions.append(('h', qubit))
            elif line.startswith('cx q['):
                # CNOT gate
                parts = line.split('q[')
                control = int(parts[1].split(']')[0])
                target = int(parts[2].split(']')[0])
                instructions.append(('cx', control, target))
            elif line.startswith('x q['):
                # X gate
                qubit = int(line.split('[')[1].split(']')[0])
                instructions.append(('x', qubit))
            elif line.startswith('y q['):
                # Y gate
                qubit = int(line.split('[')[1].split(']')[0])
                instructions.append(('y', qubit))
            elif line.startswith('z q['):
                # Z gate
                qubit = int(line.split('[')[1].split(']')[0])
                instructions.append(('z', qubit))
            elif line.startswith('rz('):
                # RZ gate
                qubit = int(line.split('q[')[1].split(']')[0])
                angle = float(line.split('(')[1].split(')')[0])
                instructions.append(('rz', qubit, angle))
        
        print(f"Parsed {len(instructions)} instructions")
        
        # Create quimb circuit manually
        print("\nCreating quimb circuit...")
        start_time = time.time()
        
        # Initialize qubits as tensors
        qubits = [qtn.qu() for _ in range(num_qubits)]
        
        # Apply gates
        for instruction in instructions:
            if instruction[0] == 'h':
                qubit_idx = instruction[1]
                qubits[qubit_idx] = qtn.gate('H') @ qubits[qubit_idx]
            elif instruction[0] == 'x':
                qubit_idx = instruction[1]
                qubits[qubit_idx] = qtn.gate('X') @ qubits[qubit_idx]
            elif instruction[0] == 'y':
                qubit_idx = instruction[1]
                qubits[qubit_idx] = qtn.gate('Y') @ qubits[qubit_idx]
            elif instruction[0] == 'z':
                qubit_idx = instruction[1]
                qubits[qubit_idx] = qtn.gate('Z') @ qubits[qubit_idx]
            elif instruction[0] == 'rz':
                qubit_idx = instruction[1]
                angle = instruction[2]
                qubits[qubit_idx] = qtn.gate('RZ', angle) @ qubits[qubit_idx]
            elif instruction[0] == 'cx':
                control_idx = instruction[1]
                target_idx = instruction[2]
                # Apply CNOT using tensor contraction
                cnot_gate = qtn.gate('CNOT')
                qubits[control_idx] = cnot_gate @ qubits[control_idx]
                qubits[target_idx] = cnot_gate @ qubits[target_idx]
        
        # Create the full state by contracting all qubits
        psi = qtn.tensor_contract(*qubits)
        
        circuit_time = time.time() - start_time
        print(f"Circuit creation completed in {circuit_time:.2f} seconds")
        
        # Convert to MPS (Matrix Product State)
        print("\nConverting to MPS...")
        start_time = time.time()
        mps = psi.to_mps()
        mps_time = time.time() - start_time
        print(f"MPS conversion completed in {mps_time:.2f} seconds")
        
        # Sample from the circuit
        num_shots = 1_000_000  # Reduced for testing
        print(f"\nSampling {num_shots:,} shots from the circuit...")
        start_time = time.time()
        
        # Sample from MPS
        samples = []
        for _ in range(num_shots):
            # Sample one bitstring
            sample = mps.sample()
            samples.append(sample)
        
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
        total_time = circuit_time + mps_time + sample_time + analysis_time
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

def test_simple_quimb_circuit():
    """Test with a simple manually created circuit"""
    print("Testing simple quimb circuit creation...")
    print("=" * 60)
    
    try:
        # Create a simple 2-qubit circuit: H on qubit 0, CNOT(0,1)
        num_qubits = 2
        
        # Initialize qubits
        q0 = qtn.qu()
        q1 = qtn.qu()
        
        # Apply Hadamard to qubit 0
        q0 = qtn.gate('H') @ q0
        
        # Apply CNOT with control=0, target=1
        cnot = qtn.gate('CNOT')
        q0 = cnot @ q0
        q1 = cnot @ q1
        
        # Contract to get final state
        psi = qtn.tensor_contract(q0, q1)
        
        print("Simple circuit created successfully")
        print(f"State shape: {psi.shape}")
        
        # Convert to MPS
        mps = psi.to_mps()
        print("MPS conversion successful")
        
        # Sample a few times
        num_shots = 1000
        samples = []
        for _ in range(num_shots):
            sample = mps.sample()
            samples.append(sample)
        
        counter = Counter(samples)
        print(f"\nSampling results ({num_shots} shots):")
        for bitstring, count in counter.most_common():
            prob = count / num_shots
            print(f"  {bitstring}: {count} times ({prob:.3f})")
        
        return True
        
    except Exception as e:
        print(f"Error in simple circuit test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Quimb Tensor Peaked Circuit Analysis Test")
    print("=" * 60)
    
    # First test simple circuit
    print("Testing simple circuit first...")
    simple_success = test_simple_quimb_circuit()
    
    if simple_success:
        print("\n" + "="*60)
        print("Simple circuit test passed, testing peaked circuit...")
        print("="*60)
        
        # Test peaked circuit
        result = test_peaked_circuit_analysis()
        
        if result:
            print(f"\n{'='*60}")
            print("SUMMARY:")
            print(f"{'='*60}")
            print(f"Successfully analyzed peaked circuit")
            print(f"Peak state found: {result['peak_state']}")
            print(f"Probability: {result['estimated_prob']:.8f}")
            print(f"Total time: {result['total_time']:.2f} seconds")
    else:
        print("Simple circuit test failed, skipping peaked circuit test") 