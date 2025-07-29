#!/usr/bin/env python3
"""
QASM Converter Module

This module provides functionality to parse QASM files and convert them
to different quantum simulator formats (Quimb, Cirq, TensorNetwork).
"""

import re
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False

try:
    import quimb.tensor as qtn
    import quimb.gen as qugen
    QUIMB_AVAILABLE = True
except ImportError:
    QUIMB_AVAILABLE = False

try:
    import tensornetwork as tn
    TENSORNETWORK_AVAILABLE = True
except ImportError:
    TENSORNETWORK_AVAILABLE = False


class QASMParser:
    """Parser for OpenQASM 2.0 files"""
    
    def __init__(self):
        self.gates = {}
        self.registers = {}
        self.instructions = []
        
    def parse_qasm_file(self, filename: str) -> Dict[str, Any]:
        """Parse a QASM file and return structured data"""
        with open(filename, 'r') as f:
            content = f.read()
        
        return self.parse_qasm_content(content)
    
    def parse_qasm_content(self, content: str) -> Dict[str, Any]:
        """Parse QASM content string"""
        lines = content.split('\n')
        parsed_data = {
            'version': None,
            'includes': [],
            'registers': {},
            'instructions': []
        }
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
                
            # Parse version
            if line.startswith('OPENQASM'):
                parsed_data['version'] = line.split(';')[0]
                
            # Parse includes
            elif line.startswith('include'):
                include_match = re.match(r'include\s+"([^"]+)"', line)
                if include_match:
                    parsed_data['includes'].append(include_match.group(1))
                    
            # Parse registers
            elif line.startswith('qreg') or line.startswith('creg'):
                reg_match = re.match(r'(qreg|creg)\s+(\w+)\[(\d+)\]', line)
                if reg_match:
                    reg_type, name, size = reg_match.groups()
                    parsed_data['registers'][name] = {
                        'type': reg_type,
                        'size': int(size)
                    }
                    
            # Parse instructions
            else:
                instruction = self.parse_instruction(line)
                if instruction:
                    parsed_data['instructions'].append(instruction)
        
        return parsed_data
    
    def parse_instruction(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single instruction line"""
        # Remove semicolon
        line = line.rstrip(';')
        
        # Parse gate applications
        gate_match = re.match(r'(\w+)\s*\(([^)]*)\)\s+(\w+)\[(\d+)\]', line)
        if gate_match:
            gate_name, params, reg_name, index = gate_match.groups()
            return {
                'type': 'gate',
                'name': gate_name,
                'parameters': self.parse_parameters(params),
                'register': reg_name,
                'index': int(index)
            }
        
        # Parse two-qubit gates
        two_qubit_match = re.match(r'(\w+)\s+(\w+)\[(\d+)\],(\w+)\[(\d+)\]', line)
        if two_qubit_match:
            gate_name, reg1, idx1, reg2, idx2 = two_qubit_match.groups()
            return {
                'type': 'two_qubit_gate',
                'name': gate_name,
                'register1': reg1,
                'index1': int(idx1),
                'register2': reg2,
                'index2': int(idx2)
            }
        
        return None
    
    def parse_parameters(self, param_str: str) -> List[float]:
        """Parse gate parameters"""
        if not param_str.strip():
            return []
        
        params = []
        for param in param_str.split(','):
            try:
                params.append(float(param.strip()))
            except ValueError:
                # Handle expressions or variables
                params.append(param.strip())
        
        return params


class QASMConverter:
    """Convert QASM circuits to different quantum simulator formats"""
    
    def __init__(self):
        self.parser = QASMParser()
    
    def qasm_to_qiskit(self, qasm_file: str) -> Optional[QuantumCircuit]:
        """Convert QASM file to Qiskit QuantumCircuit"""
        if not QISKIT_AVAILABLE:
            return None
        
        try:
            return QuantumCircuit.from_qasm_file(qasm_file)
        except Exception as e:
            print(f"Error converting to Qiskit: {e}")
            return None
    
    def qasm_to_cirq(self, qasm_file: str) -> Optional[cirq.Circuit]:
        """Convert QASM file to Cirq Circuit"""
        if not CIRQ_AVAILABLE:
            return None
        
        try:
            # Parse QASM
            parsed_data = self.parser.parse_qasm_file(qasm_file)
            
            # Get number of qubits
            num_qubits = 0
            for reg_info in parsed_data['registers'].values():
                if reg_info['type'] == 'qreg':
                    num_qubits += reg_info['size']
            
            if num_qubits == 0:
                return None
            
            # Create qubits
            qubits = cirq.LineQubit.range(num_qubits)
            circuit = cirq.Circuit()
            
            # Convert instructions
            for instruction in parsed_data['instructions']:
                if instruction['type'] == 'gate':
                    gate = self._create_cirq_gate(instruction)
                    if gate is not None:
                        qubit_idx = instruction['index']
                        if qubit_idx < num_qubits:
                            circuit.append(gate(qubits[qubit_idx]))
                            
                elif instruction['type'] == 'two_qubit_gate':
                    gate = self._create_cirq_two_qubit_gate(instruction)
                    if gate is not None:
                        idx1, idx2 = instruction['index1'], instruction['index2']
                        if idx1 < num_qubits and idx2 < num_qubits:
                            circuit.append(gate(qubits[idx1], qubits[idx2]))
            
            return circuit
            
        except Exception as e:
            print(f"Error converting to Cirq: {e}")
            return None
    
    def qasm_to_quimb(self, qasm_file: str) -> Optional[Dict[str, Any]]:
        """Convert QASM file to Quimb Circuit"""
        if not QUIMB_AVAILABLE:
            return None
        
        try:
            # Parse QASM
            parsed_data = self.parser.parse_qasm_file(qasm_file)
            
            # Get number of qubits
            num_qubits = 0
            for reg_info in parsed_data['registers'].values():
                if reg_info['type'] == 'qreg':
                    num_qubits += reg_info['size']
            
            if num_qubits == 0:
                return None
            
            # Create qubits - use correct Quimb API
            # Note: Quimb doesn't have a simple qubit() function, so we'll create a simplified representation
            qubits = list(range(num_qubits))
            circuit = {
                'qubits': qubits,
                'num_qubits': num_qubits,
                'instructions': []
            }
            
            # Convert instructions (simplified)
            # Note: Full conversion would require more sophisticated mapping
            for instruction in parsed_data['instructions']:
                if instruction['type'] == 'gate':
                    gate = self._create_quimb_gate(instruction)
                    if gate is not None:
                        qubit_idx = instruction['index']
                        if qubit_idx < num_qubits:
                            circuit['instructions'].append(gate)
                            
                elif instruction['type'] == 'two_qubit_gate':
                    gate = self._create_quimb_two_qubit_gate(instruction)
                    if gate is not None:
                        idx1, idx2 = instruction['index1'], instruction['index2']
                        if idx1 < num_qubits and idx2 < num_qubits:
                            circuit['instructions'].append(gate)
            
            return circuit
            
        except Exception as e:
            print(f"Error converting to Quimb: {e}")
            return None
    
    def qasm_to_tensornetwork(self, qasm_file: str) -> Optional[Any]:
        """Convert QASM file to TensorNetwork representation"""
        if not TENSORNETWORK_AVAILABLE:
            return None
        
        try:
            # Parse QASM
            parsed_data = self.parser.parse_qasm_file(qasm_file)
            
            # Get number of qubits
            num_qubits = 0
            for reg_info in parsed_data['registers'].values():
                if reg_info['type'] == 'qreg':
                    num_qubits += reg_info['size']
            
            if num_qubits == 0:
                return None
            
            # Create tensor network representation
            # This is a simplified version - full implementation would be more complex
            tensors = []
            
            # Convert instructions to tensor operations
            for instruction in parsed_data['instructions']:
                if instruction['type'] == 'gate':
                    tensor = self._create_tensornetwork_gate(instruction)
                    if tensor is not None:
                        tensors.append(tensor)
                        
                elif instruction['type'] == 'two_qubit_gate':
                    tensor = self._create_tensornetwork_two_qubit_gate(instruction)
                    if tensor is not None:
                        tensors.append(tensor)
            
            return tensors
            
        except Exception as e:
            print(f"Error converting to TensorNetwork: {e}")
            return None
    
    def _create_cirq_gate(self, instruction: Dict[str, Any]) -> Optional[cirq.Gate]:
        """Create Cirq gate from instruction"""
        gate_name = instruction['name']
        params = instruction['parameters']
        
        if gate_name == 'u3':
            if len(params) >= 3:
                # Use proper Cirq gate construction for u3
                theta, phi, lam = params[0], params[1], params[2]
                return cirq.ops.PhasedXZGate(
                    x_exponent=theta / np.pi,
                    z_exponent=lam / np.pi,
                    axis_phase_exponent=phi / np.pi
                )
        elif gate_name == 'u2':
            if len(params) >= 2:
                # Use proper Cirq gate construction for u2
                phi, lam = params[0], params[1]
                return cirq.ops.PhasedXZGate(
                    x_exponent=0.5,
                    z_exponent=lam / np.pi,
                    axis_phase_exponent=phi / np.pi
                )
        elif gate_name == 'u1':
            if len(params) >= 1:
                return cirq.ops.ZPowGate(exponent=params[0] / np.pi)
        elif gate_name == 'rx':
            if len(params) >= 1:
                return cirq.ops.Rx(rads=params[0])
        elif gate_name == 'ry':
            if len(params) >= 1:
                return cirq.ops.Ry(rads=params[0])
        elif gate_name == 'rz':
            if len(params) >= 1:
                return cirq.ops.Rz(rads=params[0])
        elif gate_name == 'x':
            return cirq.ops.X
        elif gate_name == 'y':
            return cirq.ops.Y
        elif gate_name == 'z':
            return cirq.ops.Z
        elif gate_name == 'h':
            return cirq.ops.H
        elif gate_name == 's':
            return cirq.ops.S
        elif gate_name == 't':
            return cirq.ops.T
        
        return None
    
    def _create_cirq_two_qubit_gate(self, instruction: Dict[str, Any]) -> Optional[cirq.Gate]:
        """Create Cirq two-qubit gate from instruction"""
        gate_name = instruction['name']
        
        if gate_name == 'cx':
            return cirq.ops.CX
        elif gate_name == 'cy':
            return cirq.ops.CY
        elif gate_name == 'cz':
            return cirq.ops.CZ
        elif gate_name == 'swap':
            return cirq.ops.SWAP
        
        return None
    
    def _create_quimb_gate(self, instruction: Dict[str, Any]) -> Optional[Any]:
        """Create Quimb gate from instruction"""
        # Simplified implementation
        gate_name = instruction['name']
        params = instruction['parameters']
        
        # Return gate specification for Quimb
        return {
            'name': gate_name,
            'parameters': params,
            'qubit': instruction['index']
        }
    
    def _create_quimb_two_qubit_gate(self, instruction: Dict[str, Any]) -> Optional[Any]:
        """Create Quimb two-qubit gate from instruction"""
        gate_name = instruction['name']
        
        return {
            'name': gate_name,
            'qubit1': instruction['index1'],
            'qubit2': instruction['index2']
        }
    
    def _create_tensornetwork_gate(self, instruction: Dict[str, Any]) -> Optional[Any]:
        """Create TensorNetwork gate from instruction"""
        # Simplified implementation
        gate_name = instruction['name']
        params = instruction['parameters']
        
        return {
            'name': gate_name,
            'parameters': params,
            'qubit': instruction['index']
        }
    
    def _create_tensornetwork_two_qubit_gate(self, instruction: Dict[str, Any]) -> Optional[Any]:
        """Create TensorNetwork two-qubit gate from instruction"""
        gate_name = instruction['name']
        
        return {
            'name': gate_name,
            'qubit1': instruction['index1'],
            'qubit2': instruction['index2']
        }


def test_converter():
    """Test the QASM converter with a simple circuit"""
    converter = QASMConverter()
    
    # Create a simple test QASM
    test_qasm = """
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""
    
    # Test parsing
    parser = QASMParser()
    parsed = parser.parse_qasm_content(test_qasm)
    print("Parsed QASM:", parsed)
    
    # Test conversions
    if QISKIT_AVAILABLE:
        print("Qiskit conversion available")
    
    if CIRQ_AVAILABLE:
        print("Cirq conversion available")
    
    if QUIMB_AVAILABLE:
        print("Quimb conversion available")
    
    if TENSORNETWORK_AVAILABLE:
        print("TensorNetwork conversion available")


if __name__ == "__main__":
    test_converter() 