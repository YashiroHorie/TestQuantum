OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

// Simple 3-qubit circuit with basic gates
// This creates a simple quantum state for testing

// Apply Hadamard to first qubit
h q[0];

// Apply CNOT between qubits 0 and 1
cx q[0],q[1];

// Apply rotation to qubit 2
rz(1.5707963267948966) q[2];

// Apply CNOT between qubits 1 and 2
cx q[1],q[2];

// Apply another Hadamard to qubit 0
h q[0];

// Apply X gate to qubit 1
x q[1];

// Apply CNOT between qubits 0 and 2
cx q[0],q[2];

// Apply rotation to qubit 1
ry(0.7853981633974483) q[1];

// Apply final CNOT
cx q[2],q[0];

// Measure all qubits
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2]; 