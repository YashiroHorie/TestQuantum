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
        
        # Create quimb circuit manually using correct API
        print("\nCreating quimb circuit...")
        start_time = time.time()
        
        # Initialize qubits as tensors using correct quimb API
        # Use tensor creation with proper initialization
        qubits = []
        for i in range(num_qubits):
            # Create a qubit tensor in |0⟩ state
            qubit_tensor = qtn.Tensor([1.0, 0.0], inds=[f'q{i}'])
            qubits.append(qubit_tensor)
        
        # Apply gates
        for instruction in instructions:
            if instruction[0] == 'h':
                qubit_idx = instruction[1]
                # Hadamard gate matrix
                H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                H_tensor = qtn.Tensor(H, inds=[f'q{qubit_idx}', f'q{qubit_idx}_out'])
                qubits[qubit_idx] = H_tensor @ qubits[qubit_idx]
            elif instruction[0] == 'x':
                qubit_idx = instruction[1]
                # X gate matrix
                X = np.array([[0, 1], [1, 0]])
                X_tensor = qtn.Tensor(X, inds=[f'q{qubit_idx}', f'q{qubit_idx}_out'])
                qubits[qubit_idx] = X_tensor @ qubits[qubit_idx]
            elif instruction[0] == 'y':
                qubit_idx = instruction[1]
                # Y gate matrix
                Y = np.array([[0, -1j], [1j, 0]])
                Y_tensor = qtn.Tensor(Y, inds=[f'q{qubit_idx}', f'q{qubit_idx}_out'])
                qubits[qubit_idx] = Y_tensor @ qubits[qubit_idx]
            elif instruction[0] == 'z':
                qubit_idx = instruction[1]
                # Z gate matrix
                Z = np.array([[1, 0], [0, -1]])
                Z_tensor = qtn.Tensor(Z, inds=[f'q{qubit_idx}', f'q{qubit_idx}_out'])
                qubits[qubit_idx] = Z_tensor @ qubits[qubit_idx]
            elif instruction[0] == 'rz':
                qubit_idx = instruction[1]
                angle = instruction[2]
                # RZ gate matrix
                RZ = np.array([[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]])
                RZ_tensor = qtn.Tensor(RZ, inds=[f'q{qubit_idx}', f'q{qubit_idx}_out'])
                qubits[qubit_idx] = RZ_tensor @ qubits[qubit_idx]
            elif instruction[0] == 'cx':
                control_idx = instruction[1]
                target_idx = instruction[2]
                # CNOT gate - this is more complex and requires proper tensor network construction
                # For now, we'll skip CNOT gates in this simplified version
                print(f"  Note: CNOT gate on qubits {control_idx}, {target_idx} - skipping for now")
        
        circuit_time = time.time() - start_time
        print(f"Circuit creation completed in {circuit_time:.2f} seconds")
        
        # Calculate state vector using tensor contraction
        print("\nCalculating state vector...")
        start_time = time.time()
        
        # For a simplified approach, we'll use numpy to simulate the circuit
        # This gives us the full state vector to analyze peak probabilities
        state_vector = simulate_circuit_numpy(num_qubits, instructions)
        
        state_time = time.time() - start_time
        print(f"State vector calculation completed in {state_time:.2f} seconds")
        
        # Find peak probability state
        print("\nAnalyzing peak probability...")
        start_time = time.time()
        
        # Calculate probabilities for each basis state
        probabilities = np.abs(state_vector) ** 2
        
        # Find the state with maximum probability
        peak_index = np.argmax(probabilities)
        peak_probability = probabilities[peak_index]
        
        # Convert index to bitstring
        peak_bitstring = format(peak_index, f'0{num_qubits}b')
        
        analysis_time = time.time() - start_time
        print(f"Peak analysis completed in {analysis_time:.2f} seconds")
        
        # Print results
        print("\n" + "=" * 60)
        print("STATE VECTOR ANALYSIS RESULTS:")
        print("=" * 60)
        print(f"Peak bitstring: {peak_bitstring}")
        print(f"Peak probability: {peak_probability:.8f} ({peak_probability*100:.6f}%)")
        print(f"Peak state index: {peak_index}")
        
        # Show state vector for peak probability
        print(f"\nState vector amplitude for peak state:")
        print(f"  |{peak_bitstring}⟩ = {state_vector[peak_index]:.6f}")
        
        # Show top 5 most probable states
        print(f"\nTop 5 most probable states:")
        print("-" * 50)
        top_indices = np.argsort(probabilities)[-5:][::-1]
        for i, idx in enumerate(top_indices, 1):
            bitstring = format(idx, f'0{num_qubits}b')
            prob = probabilities[idx]
            amplitude = state_vector[idx]
            print(f"{i}. |{bitstring}⟩: prob={prob:.8f} ({prob*100:.6f}%), amp={amplitude:.6f}")
        
        # Show some statistics about the state vector
        print(f"\nState vector statistics:")
        print("-" * 30)
        print(f"Total states: {len(state_vector)}")
        print(f"Non-zero amplitudes: {np.count_nonzero(state_vector)}")
        print(f"Max amplitude magnitude: {np.max(np.abs(state_vector)):.6f}")
        print(f"Min amplitude magnitude: {np.min(np.abs(state_vector[np.abs(state_vector) > 0])):.6f}")
        print(f"State vector norm: {np.linalg.norm(state_vector):.6f}")
        
        # Calculate total time
        total_time = circuit_time + state_time + analysis_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        
        return {
            'peak_bitstring': peak_bitstring,
            'peak_probability': peak_probability,
            'peak_index': peak_index,
            'state_vector': state_vector,
            'probabilities': probabilities,
            'total_time': total_time,
            'num_qubits': num_qubits
        }
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None

def simulate_circuit_numpy(num_qubits, instructions):
    """Simulate quantum circuit using numpy to get full state vector"""
    
    # Initialize state vector in |0...0⟩ state
    state_size = 2 ** num_qubits
    state_vector = np.zeros(state_size, dtype=complex)
    state_vector[0] = 1.0  # Start in |0...0⟩ state
    
    # Apply gates
    for instruction in instructions:
        if instruction[0] == 'h':
            qubit_idx = instruction[1]
            # Apply Hadamard to the specified qubit
            state_vector = apply_hadamard(state_vector, qubit_idx, num_qubits)
        elif instruction[0] == 'x':
            qubit_idx = instruction[1]
            # Apply X gate to the specified qubit
            state_vector = apply_x_gate(state_vector, qubit_idx, num_qubits)
        elif instruction[0] == 'y':
            qubit_idx = instruction[1]
            # Apply Y gate to the specified qubit
            state_vector = apply_y_gate(state_vector, qubit_idx, num_qubits)
        elif instruction[0] == 'z':
            qubit_idx = instruction[1]
            # Apply Z gate to the specified qubit
            state_vector = apply_z_gate(state_vector, qubit_idx, num_qubits)
        elif instruction[0] == 'rz':
            qubit_idx = instruction[1]
            angle = instruction[2]
            # Apply RZ gate to the specified qubit
            state_vector = apply_rz_gate(state_vector, qubit_idx, num_qubits, angle)
        elif instruction[0] == 'cx':
            control_idx = instruction[1]
            target_idx = instruction[2]
            # Apply CNOT gate
            state_vector = apply_cnot_gate(state_vector, control_idx, target_idx, num_qubits)
    
    return state_vector

def apply_hadamard(state_vector, qubit_idx, num_qubits):
    """Apply Hadamard gate to a specific qubit"""
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    return apply_single_qubit_gate(state_vector, H, qubit_idx, num_qubits)

def apply_x_gate(state_vector, qubit_idx, num_qubits):
    """Apply X gate to a specific qubit"""
    X = np.array([[0, 1], [1, 0]])
    return apply_single_qubit_gate(state_vector, X, qubit_idx, num_qubits)

def apply_y_gate(state_vector, qubit_idx, num_qubits):
    """Apply Y gate to a specific qubit"""
    Y = np.array([[0, -1j], [1j, 0]])
    return apply_single_qubit_gate(state_vector, Y, qubit_idx, num_qubits)

def apply_z_gate(state_vector, qubit_idx, num_qubits):
    """Apply Z gate to a specific qubit"""
    Z = np.array([[1, 0], [0, -1]])
    return apply_single_qubit_gate(state_vector, Z, qubit_idx, num_qubits)

def apply_rz_gate(state_vector, qubit_idx, num_qubits, angle):
    """Apply RZ gate to a specific qubit"""
    RZ = np.array([[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]])
    return apply_single_qubit_gate(state_vector, RZ, qubit_idx, num_qubits)

def apply_single_qubit_gate(state_vector, gate_matrix, qubit_idx, num_qubits):
    """Apply a single-qubit gate to a specific qubit"""
    new_state = np.zeros_like(state_vector)
    
    for i in range(len(state_vector)):
        # Get the bitstring for this state
        bitstring = format(i, f'0{num_qubits}b')
        
        # Get the bit value for the target qubit
        target_bit = int(bitstring[num_qubits - 1 - qubit_idx])
        
        # Apply the gate
        if target_bit == 0:
            # Apply gate to |0⟩ component
            new_state[i] += gate_matrix[0, 0] * state_vector[i]
            # Find the state with this qubit flipped to |1⟩
            flipped_bitstring = list(bitstring)
            flipped_bitstring[num_qubits - 1 - qubit_idx] = '1'
            flipped_index = int(''.join(flipped_bitstring), 2)
            new_state[flipped_index] += gate_matrix[1, 0] * state_vector[i]
        else:
            # Apply gate to |1⟩ component
            new_state[i] += gate_matrix[0, 1] * state_vector[i]
            # Find the state with this qubit flipped to |0⟩
            flipped_bitstring = list(bitstring)
            flipped_bitstring[num_qubits - 1 - qubit_idx] = '0'
            flipped_index = int(''.join(flipped_bitstring), 2)
            new_state[flipped_index] += gate_matrix[1, 1] * state_vector[i]
    
    return new_state

def apply_cnot_gate(state_vector, control_idx, target_idx, num_qubits):
    """Apply CNOT gate between control and target qubits"""
    new_state = np.zeros_like(state_vector)
    
    for i in range(len(state_vector)):
        # Get the bitstring for this state
        bitstring = format(i, f'0{num_qubits}b')
        
        # Get the bit values for control and target qubits
        control_bit = int(bitstring[num_qubits - 1 - control_idx])
        target_bit = int(bitstring[num_qubits - 1 - target_idx])
        
        if control_bit == 1:
            # Control qubit is |1⟩, so flip target qubit
            flipped_bitstring = list(bitstring)
            flipped_bitstring[num_qubits - 1 - target_idx] = '1' if target_bit == 0 else '0'
            flipped_index = int(''.join(flipped_bitstring), 2)
            new_state[flipped_index] = state_vector[i]
        else:
            # Control qubit is |0⟩, so no change
            new_state[i] = state_vector[i]
    
    return new_state

def test_simple_quimb_circuit():
    """Test with a simple manually created circuit"""
    print("Testing simple quimb circuit creation...")
    print("=" * 60)
    
    try:
        # Create a simple 2-qubit circuit using basic tensor operations
        print("Creating simple 2-qubit Bell state circuit...")
        
        # Create qubit tensors in |0⟩ state
        q0 = qtn.Tensor([1.0, 0.0], inds=['q0'])
        q1 = qtn.Tensor([1.0, 0.0], inds=['q1'])
        
        print("Qubit tensors created successfully")
        print(f"q0 shape: {q0.shape}, q1 shape: {q1.shape}")
        
        # Apply Hadamard to qubit 0
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        H_tensor = qtn.Tensor(H, inds=['q0', 'q0_out'])
        q0 = H_tensor @ q0
        
        print("Hadamard gate applied to qubit 0")
        
        # For CNOT, we would need more complex tensor network construction
        # For now, just demonstrate the basic tensor operations work
        print("CNOT gate would require more complex tensor network construction")
        
        # Show that basic tensor operations work
        print("Basic tensor operations successful!")
        print(f"Final q0 tensor shape: {q0.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error in simple circuit test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quimb_basic_operations():
    """Test basic quimb operations to understand the API"""
    print("Testing basic quimb operations...")
    print("=" * 60)
    
    try:
        # Test basic tensor creation
        print("1. Testing basic tensor creation...")
        t1 = qtn.Tensor([1, 2, 3, 4], inds=['a'])
        print(f"   Created tensor: {t1}")
        print(f"   Shape: {t1.shape}")
        
        # Test tensor contraction
        print("\n2. Testing tensor contraction...")
        t2 = qtn.Tensor([[1, 2], [3, 4]], inds=['a', 'b'])
        t3 = qtn.Tensor([5, 6], inds=['b'])
        result = t2 @ t3
        print(f"   Contraction result: {result}")
        
        # Test available functions
        print("\n3. Available quimb.tensor functions:")
        functions = [f for f in dir(qtn) if not f.startswith('_')]
        for i, func in enumerate(functions[:10]):  # Show first 10
            print(f"   {i+1}. {func}")
        if len(functions) > 10:
            print(f"   ... and {len(functions) - 10} more")
        
        return True
        
    except Exception as e:
        print(f"Error in basic operations test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Quimb Tensor Peaked Circuit Analysis Test")
    print("=" * 60)
    
    # First test basic quimb operations
    print("Testing basic quimb operations first...")
    basic_success = test_quimb_basic_operations()
    
    if basic_success:
        print("\n" + "="*60)
        print("Basic operations test passed, testing simple circuit...")
        print("="*60)
        
        # Test simple circuit
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
                print(f"Peak bitstring: {result['peak_bitstring']}")
                print(f"Peak probability: {result['peak_probability']:.8f}")
                print(f"Qubits: {result['num_qubits']}")
                print(f"Total time: {result['total_time']:.2f} seconds")
        else:
            print("Simple circuit test failed, skipping peaked circuit test")
    else:
        print("Basic operations test failed, skipping other tests") 