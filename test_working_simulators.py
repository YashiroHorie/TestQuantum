#!/usr/bin/env python3
"""
Working Simulators Test

This script tests QASM files using the working simulators:
- Qiskit MPS (Matrix Product State)
- Qsim (Google's Quantum Simulator)

It provides detailed analysis and comparison of results.
"""

import os
import time
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import quantum libraries
try:
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit.compiler import transpile
    from qiskit.quantum_info import Statevector, state_fidelity
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available")

try:
    import qsimcirq
    import cirq
    QSIMCIRQ_AVAILABLE = True
except ImportError:
    QSIMCIRQ_AVAILABLE = False
    print("Warning: QsimCirq not available")

class WorkingSimulatorTest:
    """Test quantum circuits using working simulators"""
    
    def __init__(self, sample_dir="sample_circuits"):
        self.sample_dir = sample_dir
        self.results = []
        
    def find_qasm_files(self):
        """Find all QASM files in the sample directory"""
        qasm_files = []
        
        if os.path.exists(self.sample_dir):
            for root, dirs, files in os.walk(self.sample_dir):
                for file in files:
                    if file.endswith('.qasm'):
                        qasm_files.append(os.path.join(root, file))
        
        # Sort by difficulty (extract from filename)
        def extract_difficulty(filepath):
            filename = os.path.basename(filepath)
            if 'diff=' in filename:
                try:
                    diff_str = filename.split('diff=')[1].split('_')[0]
                    return float(diff_str)
                except:
                    return float('inf')
            return float('inf')
        
        qasm_files.sort(key=extract_difficulty)
        return qasm_files
    
    def test_qiskit_mps(self, qasm_file):
        """Test using Qiskit MPS simulator"""
        if not QISKIT_AVAILABLE:
            return None, "Qiskit not available"
        
        try:
            start_time = time.time()
            
            # Load QASM circuit
            circuit = QuantumCircuit.from_qasm_file(qasm_file)
            num_qubits = circuit.num_qubits
            
            # Use MPS simulator with explicit shots
            mps_backend = AerSimulator(method='matrix_product_state')
            mps_backend.set_options(max_parallel_threads=4)
            transpiled_circuit = transpile(circuit, mps_backend)
            job = mps_backend.run(transpiled_circuit, shots=10000)
            print("Successfully ran the job")
            
            # Implement proper timeout mechanism
            import threading
            import queue
            
            result_queue = queue.Queue()
            error_queue = queue.Queue()
            
            def get_result():
                try:
                    result = job.result(timeout=100)
                    result_queue.put(result)
                except Exception as e:
                    error_queue.put(e)
            
            # Start result retrieval in a separate thread
            result_thread = threading.Thread(target=get_result)
            result_thread.daemon = True
            result_thread.start()
            
            # Wait for result with timeout
            try:
                result = result_queue.get(timeout=120)  # 120 second timeout
            except queue.Empty:
                # Timeout occurred
                print("Job timed out after 120 seconds")
                return None, "Job timed out after 120 seconds"
            
            # Check for errors
            try:
                error = error_queue.get_nowait()
                return None, f"Job failed: {error}"
            except queue.Empty:
                pass  # No error occurred
            
            # Get counts and reconstruct state vector
            counts = result.get_counts()
            statevector = np.zeros(2**num_qubits, dtype=complex)
            total_shots = sum(counts.values())
            
            for bitstring, count in counts.items():
                index = int(bitstring, 2)
                # Estimate amplitude from counts
                amplitude = np.sqrt(count / total_shots)
                statevector[index] = amplitude
            
            execution_time = time.time() - start_time
            
            # Find peak probability
            probabilities = np.abs(statevector) ** 2
            peak_index = np.argmax(probabilities)
            peak_probability = probabilities[peak_index]
            peak_bitstring = format(peak_index, f'0{num_qubits}b')
            
            return {
                'peak_bitstring': peak_bitstring,
                'peak_probability': peak_probability,
                'statevector': statevector,
                'execution_time': execution_time,
                'counts': counts,
                'total_shots': total_shots
            }, None
            
        except Exception as e:
            return None, str(e)
    
    def test_qiskit_mps_with_timeout(self, qasm_file, timeout_seconds=120):
        """Test using Qiskit MPS simulator with reliable timeout"""
        if not QISKIT_AVAILABLE:
            return None, "Qiskit not available"
        
        try:
            start_time = time.time()
            
            # Load QASM circuit
            circuit = QuantumCircuit.from_qasm_file(qasm_file)
            num_qubits = circuit.num_qubits
            
            # Use MPS simulator with explicit shots
            mps_backend = AerSimulator(method='matrix_product_state')
            mps_backend.set_options(max_parallel_threads=4)
            transpiled_circuit = transpile(circuit, mps_backend)
            job = mps_backend.run(transpiled_circuit, shots=10000)
            print("Successfully ran the job")
            
            # Use threading for reliable timeout
            import threading
            import queue
            
            result_queue = queue.Queue()
            error_queue = queue.Queue()
            
            def get_result():
                try:
                    result = job.result(timeout=timeout_seconds)
                    result_queue.put(('success', result))
                except Exception as e:
                    error_queue.put(('error', str(e)))
            
            # Start result retrieval in a separate thread
            result_thread = threading.Thread(target=get_result)
            result_thread.daemon = True
            result_thread.start()
            
            # Wait for result with timeout
            try:
                status, result_data = result_queue.get(timeout=timeout_seconds + 10)
                if status == 'error':
                    return None, f"Job failed: {result_data}"
                result = result_data
            except queue.Empty:
                # Timeout occurred
                print(f"Job timed out after {timeout_seconds} seconds")
                return None, f"Job timed out after {timeout_seconds} seconds"
            
            # Get counts and reconstruct state vector
            counts = result.get_counts()
            statevector = np.zeros(2**num_qubits, dtype=complex)
            total_shots = sum(counts.values())
            
            for bitstring, count in counts.items():
                index = int(bitstring, 2)
                # Estimate amplitude from counts
                amplitude = np.sqrt(count / total_shots)
                statevector[index] = amplitude
            
            execution_time = time.time() - start_time
            
            # Find peak probability
            probabilities = np.abs(statevector) ** 2
            peak_index = np.argmax(probabilities)
            peak_probability = probabilities[peak_index]
            peak_bitstring = format(peak_index, f'0{num_qubits}b')
            
            return {
                'peak_bitstring': peak_bitstring,
                'peak_probability': peak_probability,
                'statevector': statevector,
                'execution_time': execution_time,
                'counts': counts,
                'total_shots': total_shots
            }, None
            
        except Exception as e:
            return None, str(e)
    
    def test_qiskit_mps_simple(self, qasm_file, timeout_seconds=60):
        """Test using Qiskit MPS simulator with simple timeout"""
        if not QISKIT_AVAILABLE:
            return None, "Qiskit not available"
        
        try:
            start_time = time.time()
            
            # Load QASM circuit
            circuit = QuantumCircuit.from_qasm_file(qasm_file)
            num_qubits = circuit.num_qubits
            
            # Use MPS simulator with explicit shots
            mps_backend = AerSimulator(method='matrix_product_state')
            mps_backend.set_options(max_parallel_shots=16, max_parallel_threads=16, max_parallel_experiments=16)
            transpiled_circuit = transpile(circuit, mps_backend)
            job = mps_backend.run(transpiled_circuit, shots=100000)
            print("Successfully ran the job")
            
            # Simple timeout approach
            result = job.result(timeout=timeout_seconds)
            print(f"Result: {result}")
            # Get counts and reconstruct state vector
            counts = result.get_counts()
            statevector = np.zeros(2**num_qubits, dtype=complex)
            total_shots = sum(counts.values())
            
            for bitstring, count in counts.items():
                index = int(bitstring, 2)
                # Estimate amplitude from counts
                amplitude = np.sqrt(count / total_shots)
                statevector[index] = amplitude
            
            execution_time = time.time() - start_time
            
            # Find peak probability
            probabilities = np.abs(statevector) ** 2
            peak_index = np.argmax(probabilities)
            peak_probability = probabilities[peak_index]
            peak_bitstring = format(peak_index, f'0{num_qubits}b')
            
            return {
                'peak_bitstring': peak_bitstring,
                'peak_probability': peak_probability,
                'statevector': statevector,
                'execution_time': execution_time,
                'counts': counts,
                'total_shots': total_shots
            }, None
            
        except Exception as e:
            return None, str(e)
    
    def test_qsim(self, qasm_file):
        """Test using Qsim simulator"""
        if not QSIMCIRQ_AVAILABLE:
            return None, "QsimCirq not available"
        
        try:
            start_time = time.time()
            
            # Convert QASM to Cirq circuit
            circuit = self.qasm_to_cirq(qasm_file)
            if circuit is None:
                return None, "Failed to convert QASM to Cirq"
            
            # Use Qsim simulator
            simulator = qsimcirq.QSimSimulator()
            result = simulator.simulate(circuit)
            statevector = result.final_state_vector
            
            execution_time = time.time() - start_time
            
            # Find peak probability
            probabilities = np.abs(statevector) ** 2
            peak_index = np.argmax(probabilities)
            peak_probability = probabilities[peak_index]
            peak_bitstring = format(peak_index, f'0{len(circuit.all_qubits())}b')
            
            return {
                'peak_bitstring': peak_bitstring,
                'peak_probability': peak_probability,
                'statevector': statevector,
                'execution_time': execution_time
            }, None
            
        except Exception as e:
            return None, str(e)
    
    def qasm_to_cirq(self, qasm_file):
        """Convert QASM file to Cirq circuit"""
        try:
            # Read QASM file
            with open(qasm_file, 'r') as f:
                qasm_content = f.read()
            
            # Parse QASM to get circuit information
            num_qubits = 0
            instructions = []
            
            for line in qasm_content.split('\n'):
                line = line.strip()
                if line.startswith('qreg q['):
                    num_qubits = int(line.split('[')[1].split(']')[0])
                elif line.startswith('h q['):
                    qubit = int(line.split('[')[1].split(']')[0])
                    instructions.append(('h', qubit))
                elif line.startswith('cx q['):
                    parts = line.split('q[')
                    control = int(parts[1].split(']')[0])
                    target = int(parts[2].split(']')[0])
                    instructions.append(('cx', control, target))
                elif line.startswith('x q['):
                    qubit = int(line.split('[')[1].split(']')[0])
                    instructions.append(('x', qubit))
                elif line.startswith('rz('):
                    qubit = int(line.split('q[')[1].split(']')[0])
                    angle = float(line.split('(')[1].split(')')[0])
                    instructions.append(('rz', qubit, angle))
            
            # Create Cirq circuit
            qubits = cirq.LineQubit.range(num_qubits)
            circuit = cirq.Circuit()
            
            # Add operations
            for instruction in instructions:
                if instruction[0] == 'h':
                    qubit_idx = instruction[1]
                    circuit.append(cirq.H(qubits[qubit_idx]))
                elif instruction[0] == 'x':
                    qubit_idx = instruction[1]
                    circuit.append(cirq.X(qubits[qubit_idx]))
                elif instruction[0] == 'rz':
                    qubit_idx = instruction[1]
                    angle = instruction[2]
                    circuit.append(cirq.rz(angle)(qubits[qubit_idx]))
                elif instruction[0] == 'cx':
                    control_idx = instruction[1]
                    target_idx = instruction[2]
                    circuit.append(cirq.CNOT(qubits[control_idx], qubits[target_idx]))
            
            return circuit
            
        except Exception as e:
            print(f"Error converting QASM to Cirq: {e}")
            return None
    
    def run_single_test(self, qasm_file):
        """Run both simulators on a single QASM file"""
        print(f"\n{'='*80}")
        print(f"Testing: {os.path.basename(qasm_file)}")
        print(f"{'='*80}")
        
        results = {}
        
        # Test Qiskit MPS
        print("\n1. Testing Qiskit MPS...")
        result, error = self.test_qiskit_mps_simple(qasm_file, timeout_seconds=60)
        if result:
            print(f"   ✓ Qiskit MPS: Peak={result['peak_bitstring']}, "
                  f"Prob={result['peak_probability']:.6f}, "
                  f"Time={result['execution_time']:.3f}s")
            results['qiskit_mps'] = result
        else:
            print(f"   ✗ Qiskit MPS failed: {error}")
        
        # Test Qsim
        print("\n2. Testing Qsim...")
        result, error = self.test_qsim(qasm_file)
        if result:
            print(f"   ✓ Qsim: Peak={result['peak_bitstring']}, "
                  f"Prob={result['peak_probability']:.6f}, "
                  f"Time={result['execution_time']:.3f}s")
            results['qsim'] = result
        else:
            print(f"   ✗ Qsim failed: {error}")
        
        # Compare results
        if len(results) >= 2:
            print(f"\n{'='*50}")
            print("COMPARISON RESULTS:")
            print(f"{'='*50}")
            
            # Compare peak bitstrings
            peak_bitstrings = {name: result['peak_bitstring'] for name, result in results.items()}
            print(f"Peak bitstrings: {peak_bitstrings}")
            
            # Compare peak probabilities
            peak_probs = {name: result['peak_probability'] for name, result in results.items()}
            print(f"Peak probabilities: {peak_probs}")
            
            # Compare execution times
            exec_times = {name: result['execution_time'] for name, result in results.items()}
            print(f"Execution times: {exec_times}")
            
            # Calculate fidelity between simulators
            if 'qiskit_mps' in results and 'qsim' in results:
                try:
                    sv1 = results['qiskit_mps']['statevector']
                    sv2 = results['qsim']['statevector']
                    if len(sv1) == len(sv2):
                        # Normalize state vectors
                        sv1_norm = sv1 / np.linalg.norm(sv1)
                        sv2_norm = sv2 / np.linalg.norm(sv2)
                        fidelity = np.abs(np.vdot(sv1_norm, sv2_norm)) ** 2
                        print(f"Fidelity (Qiskit MPS vs Qsim): {fidelity:.6f}")
                except Exception as e:
                    print(f"Could not calculate fidelity: {e}")
        
        return results
    
    def run_all_tests(self, max_files=None):
        """Run tests on all QASM files"""
        print("Working Simulators Quantum Circuit Test")
        print("=" * 60)
        
        # Find QASM files
        qasm_files = self.find_qasm_files()
        
        if not qasm_files:
            print("No QASM files found in sample directory")
            return
        
        print(f"Found {len(qasm_files)} QASM files")
        
        if max_files:
            qasm_files = qasm_files[:max_files]
            print(f"Testing first {len(qasm_files)} files")
        
        # Test each file
        all_results = {}
        for i, qasm_file in enumerate(qasm_files, 1):
            print(f"\nFile {i}/{len(qasm_files)}")
            results = self.run_single_test(qasm_file)
            all_results[qasm_file] = results
        
        # Generate summary report
        self.generate_summary_report(all_results)
    
    def generate_summary_report(self, all_results):
        """Generate a summary report of all tests"""
        print(f"\n{'='*80}")
        print("SUMMARY REPORT")
        print(f"{'='*80}")
        
        # Create summary data
        summary_data = []
        
        for qasm_file, results in all_results.items():
            filename = os.path.basename(qasm_file)
            
            for simulator, result in results.items():
                summary_data.append({
                    'filename': filename,
                    'simulator': simulator,
                    'peak_bitstring': result['peak_bitstring'],
                    'peak_probability': result['peak_probability'],
                    'execution_time': result['execution_time']
                })
        
        # Create DataFrame
        df = pd.DataFrame(summary_data)
        
        if not df.empty:
            print("\nResults Summary:")
            print(df.to_string(index=False))
            
            # Save to CSV
            output_file = "working_simulator_results.csv"
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            # Show statistics
            print(f"\nStatistics:")
            print(f"Total tests: {len(df)}")
            print(f"Successful tests: {len(df)}")
            
            # Average execution times
            if 'execution_time' in df.columns:
                avg_times = df.groupby('simulator')['execution_time'].mean()
                print(f"\nAverage execution times:")
                for simulator, avg_time in avg_times.items():
                    print(f"  {simulator}: {avg_time:.3f}s")

def main():
    """Main function"""
    print("Working Simulators Test")
    print("=" * 60)
    
    # Create test object
    test = WorkingSimulatorTest()
    
    # Check available simulators
    available_simulators = []
    if QISKIT_AVAILABLE:
        available_simulators.append("Qiskit MPS")
    if QSIMCIRQ_AVAILABLE:
        available_simulators.append("Qsim")
    
    print(f"Available simulators: {', '.join(available_simulators)}")
    
    if len(available_simulators) < 2:
        print("Warning: Need at least 2 simulators for comparison")
        print("Continuing with available simulators...")
    
    # Run tests on first 3 files
    test.run_all_tests(max_files=3)

if __name__ == "__main__":
    main() 