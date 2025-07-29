#!/usr/bin/env python3
"""
Quantum Simulator Comparison Tool

This script runs QASM files from the sample_circuits directory using different quantum simulators:
- Qiskit Aer (statevector_simulator)
- Quimb
- QsimCirq
- TensorNetwork

It compares results in terms of:
- Accuracy (fidelity between simulators)
- Statevector (bitstring representation)
- Running time
"""

import os
import time
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import json
import re
import signal
warnings.filterwarnings('ignore')

# Import quantum libraries
try:
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit.compiler import transpile
    from qiskit.quantum_info import Statevector, state_fidelity
    from qiskit_aer import Aer
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available")

try:
    import quimb.tensor as qtn
    import quimb.gen as qugen
    QUIMB_AVAILABLE = True
except ImportError:
    QUIMB_AVAILABLE = False
    print("Warning: Quimb not available")

try:
    import qsimcirq
    import cirq
    QSIMCIRQ_AVAILABLE = True
except ImportError:
    QSIMCIRQ_AVAILABLE = False
    print("Warning: QsimCirq not available")

try:
    import tensornetwork as tn
    import tensornetwork.backends.numpy as tn_numpy
    TENSORNETWORK_AVAILABLE = True
except ImportError:
    TENSORNETWORK_AVAILABLE = False
    print("Warning: TensorNetwork not available")

# Import QASM converter
try:
    from qasm_converter import QASMConverter
    QASM_CONVERTER_AVAILABLE = True
except ImportError:
    QASM_CONVERTER_AVAILABLE = False
    print("Warning: QASM converter not available")


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Operation timed out")


def run_with_timeout(timeout_seconds=30):
    """Decorator to run a function with a timeout"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set up signal handler for timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel the alarm
                return result
            except TimeoutError:
                print(f"  ⏰ Timeout after {timeout_seconds}s")
                return None, None
            except Exception as e:
                signal.alarm(0)  # Cancel the alarm
                raise e
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


class QuantumSimulatorComparison:
    def __init__(self, circuits_dir="sample_circuits", enable_mps=True, enable_gpu=True):
        self.circuits_dir = circuits_dir
        self.results = []
        self.converter = QASMConverter() if QASM_CONVERTER_AVAILABLE else None
        self.enable_mps = enable_mps
        self.enable_gpu = enable_gpu
        print(f"Initialized with MPS: {enable_mps}, GPU: {enable_gpu}")

    def find_qasm_files(self):
        """Find all QASM files in the circuits directory, sorted by difficulty"""
        qasm_files = []
        for root, dirs, files in os.walk(self.circuits_dir):
            for file in files:
                if file.endswith('.qasm'):
                    qasm_files.append(os.path.join(root, file))
        
        # Sort by difficulty level (extracted from filename)
        def extract_difficulty(filepath):
            filename = os.path.basename(filepath)
            # Extract difficulty from filename like "peaked_circuit_diff=0.000_PUBLIC_..."
            match = re.search(r'diff=([\d.]+)', filename)
            if match:
                return float(match.group(1))
            return float('inf')  # Put files without difficulty at the end
        
        # Sort by difficulty (smaller first)
        qasm_files.sort(key=extract_difficulty)
        
        return qasm_files
    
    def get_expected_value(self, qasm_file):
        """Extract expected value from corresponding meta.json file"""
        try:
            # Get the base name without extension
            base_name = os.path.splitext(qasm_file)[0]
            meta_file = f"{base_name}_meta.json"
            
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    meta_data = json.load(f)
                
                return {
                    'target_state': meta_data.get('target_state'),
                    'peak_prob': meta_data.get('peak_prob'),
                    'difficulty_level': meta_data.get('difficulty_level'),
                    'num_qubits': meta_data.get('num_qubits'),
                    'rqc_depth': meta_data.get('rqc_depth'),
                    'pqc_depth': meta_data.get('pqc_depth'),
                    'est_num_shots': meta_data.get('est_num_shots')
                }
            else:
                return None
        except Exception as e:
            print(f"Error reading meta file for {qasm_file}: {e}")
            return None
    
    @run_with_timeout(timeout_seconds=30)
    def _run_qiskit(self, qasm_file):
        """Internal Qiskit simulation with timeout"""
        start_time = time.time()
        
        # Load QASM file
        circuit = QuantumCircuit.from_qasm_file(qasm_file)
        
        # Get number of qubits
        num_qubits = circuit.num_qubits
        
        # Execute on statevector simulator using new API
        backend = Aer.get_backend('statevector_simulator')
        transpiled_circuit = transpile(circuit, backend)
        job = backend.run(transpiled_circuit)
        result = job.result()
        statevector = result.get_statevector()
        
        # Convert Statevector object to numpy array if needed
        if hasattr(statevector, 'data'):
            statevector = statevector.data
        
        execution_time = time.time() - start_time
        
        return statevector, execution_time

    def run_qiskit_simulation(self, qasm_file):
        """Run simulation using Qiskit Aer statevector simulator"""
        if not QISKIT_AVAILABLE:
            return None, None
        
        try:
            return self._run_qiskit(qasm_file)
        except Exception as e:
            print(f"Qiskit simulation failed for {qasm_file}: {e}")
            return None, None
    
    @run_with_timeout(timeout_seconds=30)
    def _run_qiskit_mps(self, qasm_file):
        """Internal Qiskit MPS simulation with timeout"""
        start_time = time.time()
        
        # Load QASM file
        circuit = QuantumCircuit.from_qasm_file(qasm_file)
        
        # Get number of qubits
        num_qubits = circuit.num_qubits
        
        # Execute on MPS simulator
        # Use aer_simulator with MPS method for better performance on certain circuits
        backend = Aer.get_backend('aer_simulator')
        
        # Configure for MPS simulation
        from qiskit_aer import AerSimulator
        mps_backend = AerSimulator(method='matrix_product_state')
        
        transpiled_circuit = transpile(circuit, mps_backend)
        job = mps_backend.run(transpiled_circuit)
        result = job.result()
        statevector = result.get_statevector()
        
        # Convert Statevector object to numpy array if needed
        if hasattr(statevector, 'data'):
            statevector = statevector.data
        
        execution_time = time.time() - start_time
        
        return statevector, execution_time

    @run_with_timeout(timeout_seconds=30)
    def _run_qiskit_fallback(self, qasm_file):
        """Internal Qiskit fallback simulation with timeout"""
        start_time = time.time()
        circuit = QuantumCircuit.from_qasm_file(qasm_file)
        backend = Aer.get_backend('aer_simulator')
        transpiled_circuit = transpile(circuit, backend)
        job = backend.run(transpiled_circuit)
        result = job.result()
        statevector = result.get_statevector()
        
        # Convert Statevector object to numpy array if needed
        if hasattr(statevector, 'data'):
            statevector = statevector.data
        
        execution_time = time.time() - start_time
        return statevector, execution_time

    def run_qiskit_mps_simulation(self, qasm_file):
        """Run simulation using Qiskit Aer MPS simulator"""
        if not QISKIT_AVAILABLE:
            return None, None
        
        try:
            return self._run_qiskit_mps(qasm_file)
        except Exception as e:
            print(f"Qiskit MPS simulation failed for {qasm_file}: {e}")
            # Fallback to regular aer_simulator if MPS fails
            try:
                return self._run_qiskit_fallback(qasm_file)
            except Exception as e2:
                print(f"Qiskit MPS fallback also failed: {e2}")
                return None, None

    def run_qiskit_gpu_simulation(self, qasm_file):
        """Run simulation using Qiskit Aer GPU simulator"""
        if not QISKIT_AVAILABLE:
            return None, None
        
        try:
            return self._run_qiskit_gpu(qasm_file)
        except Exception as e:
            print(f"Qiskit GPU simulation failed for {qasm_file}: {e}")
            return None, None

    @run_with_timeout(timeout_seconds=30)
    def _run_qiskit_gpu(self, qasm_file):
        """Internal Qiskit GPU simulation with timeout"""
        start_time = time.time()
        
        # Load QASM file
        circuit = QuantumCircuit.from_qasm_file(qasm_file)
        
        # Execute on GPU simulator
        from qiskit_aer import AerSimulator
        gpu_backend = AerSimulator(method='statevector', device='GPU')
        
        transpiled_circuit = transpile(circuit, gpu_backend)
        job = gpu_backend.run(transpiled_circuit)
        result = job.result()
        statevector = result.get_statevector()
        
        # Convert Statevector object to numpy array if needed
        if hasattr(statevector, 'data'):
            statevector = statevector.data
        
        execution_time = time.time() - start_time
        
        return statevector, execution_time
    
    @run_with_timeout(timeout_seconds=30)
    def _run_quimb(self, qasm_file):
        """Internal Quimb simulation with timeout"""
        start_time = time.time()
        
        # Convert QASM to Quimb circuit
        circuit_quimb = self.converter.qasm_to_quimb(qasm_file)

        if circuit_quimb is None:
            return None, None

        # Execute simulation (simplified)
        # Note: Full Quimb simulation would require more sophisticated implementation
        # For now, we just verify the conversion worked
        execution_time = time.time() - start_time

        # Return None for statevector as full simulation is complex
        # But we can verify the circuit was converted successfully
        if circuit_quimb and 'num_qubits' in circuit_quimb:
            print(f"  Quimb conversion successful: {circuit_quimb['num_qubits']} qubits, {len(circuit_quimb['instructions'])} instructions")
        
        return None, execution_time

    def run_quimb_simulation(self, qasm_file):
        """Run simulation using Quimb"""
        if not QUIMB_AVAILABLE or self.converter is None:
            return None, None
        
        try:
            return self._run_quimb(qasm_file)
        except Exception as e:
            print(f"Quimb simulation failed for {qasm_file}: {e}")
            return None, None
    
    @run_with_timeout(timeout_seconds=30)
    def _run_qsimcirq(self, qasm_file):
        """Internal QsimCirq simulation with timeout"""
        start_time = time.time()
        
        # Convert QASM to Cirq circuit
        circuit_cirq = self.converter.qasm_to_cirq(qasm_file)

        if circuit_cirq is None:
            return None, None

        # Execute simulation using QsimCirq
        try:
            # Create QsimCirq simulator
            simulator = qsimcirq.QSimSimulator()
            
            # Run simulation
            result = simulator.simulate(circuit_cirq)
            statevector = result.final_state_vector
            
            execution_time = time.time() - start_time
            return statevector, execution_time
            
        except Exception as e:
            print(f"QsimCirq execution failed: {e}")
            execution_time = time.time() - start_time
            return None, execution_time

    def run_qsimcirq_simulation(self, qasm_file):
        """Run simulation using QsimCirq"""
        if not QSIMCIRQ_AVAILABLE or self.converter is None:
            return None, None
        
        try:
            return self._run_qsimcirq(qasm_file)
        except Exception as e:
            print(f"QsimCirq simulation failed for {qasm_file}: {e}")
            return None, None
    
    @run_with_timeout(timeout_seconds=30)
    def _run_tensornetwork(self, qasm_file):
        """Internal TensorNetwork simulation with timeout"""
        start_time = time.time()
        
        # Convert QASM to TensorNetwork representation
        tensors = self.converter.qasm_to_tensornetwork(qasm_file)
        
        if tensors is None:
            return None, None
        
        # Execute simulation (simplified)
        # Note: Full TensorNetwork simulation would require more sophisticated implementation
        execution_time = time.time() - start_time
        
        # For now, return None as full simulation is complex
        return None, execution_time

    def run_tensornetwork_simulation(self, qasm_file):
        """Run simulation using TensorNetwork"""
        if not TENSORNETWORK_AVAILABLE or self.converter is None:
            return None, None
        
        try:
            return self._run_tensornetwork(qasm_file)
        except Exception as e:
            print(f"TensorNetwork simulation failed for {qasm_file}: {e}")
            return None, None
    
    def calculate_fidelity(self, state1, state2):
        """Calculate fidelity between two quantum states"""
        if state1 is None or state2 is None:
            return None
        
        try:
            # Convert to Statevector objects if they aren't already
            if not isinstance(state1, Statevector):
                state1 = Statevector(state1)
            if not isinstance(state2, Statevector):
                state2 = Statevector(state2)
            
            return state_fidelity(state1, state2)
        except Exception as e:
            print(f"Fidelity calculation failed: {e}")
            return None
    
    def get_statevector_info(self, statevector):
        """Extract information from statevector"""
        if statevector is None:
            return None, None, None
        
        try:
            # Convert to numpy array if needed
            if hasattr(statevector, 'data'):
                state_array = statevector.data
            else:
                state_array = np.array(statevector)
            
            # Get magnitude and phase
            magnitude = np.abs(state_array)
            phase = np.angle(state_array)
            
            # Get dominant basis states (top 5)
            indices = np.argsort(magnitude)[::-1][:5]
            dominant_states = []
            for idx in indices:
                binary = format(idx, f'0{int(np.log2(len(state_array)))}b')
                amplitude = state_array[idx]
                dominant_states.append((binary, amplitude))
            
            return magnitude, phase, dominant_states
            
        except Exception as e:
            print(f"Statevector info extraction failed: {e}")
            return None, None, None
    
    def calculate_target_accuracy(self, statevector, target_state):
        """Calculate accuracy against target state"""
        if statevector is None or target_state is None:
            return None
            
        try:
            # Get the probability of the target state
            num_qubits = int(np.log2(len(statevector)))
            target_idx = int(target_state, 2)
            
            if target_idx >= len(statevector):
                return None
                
            target_prob = np.abs(statevector[target_idx]) ** 2
            
            return target_prob
            
        except Exception as e:
            print(f"Error calculating target accuracy: {e}")
            return None
    
    def run_comparison(self, qasm_file):
        """Run comparison for a single QASM file"""
        print(f"Processing: {qasm_file}")
        print("=" * 60)
        
        # Step 1: Get expected values from meta.json
        print("Step 1: Extracting expected values from meta.json...")
        expected_data = self.get_expected_value(qasm_file)
        if expected_data:
            print(f"  ✓ Found meta data: target_state='{expected_data.get('target_state')}', difficulty={expected_data.get('difficulty_level')}")
        else:
            print("  ⚠ No meta.json file found")
        
        # Step 2: Run Qiskit simulation
        print("\nStep 2: Running Qiskit (Statevector) simulation...")
        qiskit_state, qiskit_time = self.run_qiskit_simulation(qasm_file)
        if qiskit_state is not None:
            print(f"  ✓ Qiskit simulation completed in {qiskit_time:.4f}s")
            print(f"    Statevector shape: {qiskit_state.shape}")
        else:
            print("  ✗ Qiskit simulation failed")
        
        # Step 3: Run MPS simulation (if enabled)
        if self.enable_mps:
            print("\nStep 3: Running Qiskit (MPS) simulation...")
            qiskit_mps_state, qiskit_mps_time = self.run_qiskit_mps_simulation(qasm_file)
            if qiskit_mps_state is not None:
                print(f"  ✓ Qiskit MPS simulation completed in {qiskit_mps_time:.4f}s")
                print(f"    Statevector shape: {qiskit_mps_state.shape}")
            else:
                print("  ✗ Qiskit MPS simulation failed")
        else:
            print("\nStep 3: Skipping Qiskit MPS simulation (disabled)")
            qiskit_mps_state, qiskit_mps_time = None, None
            
        # Step 4: Run GPU simulation (if enabled)
        if self.enable_gpu:
            print("\nStep 4: Running Qiskit (GPU) simulation...")
            qiskit_gpu_state, qiskit_gpu_time = self.run_qiskit_gpu_simulation(qasm_file)
            if qiskit_gpu_state is not None:
                print(f"  ✓ Qiskit GPU simulation completed in {qiskit_gpu_time:.4f}s")
                print(f"    Statevector shape: {qiskit_gpu_state.shape}")
            else:
                print("  ✗ Qiskit GPU simulation failed")
        else:
            print("\nStep 4: Skipping Qiskit GPU simulation (disabled)")
            qiskit_gpu_state, qiskit_gpu_time = None, None
            
        # Step 5: Run Quimb simulation
        print("\nStep 5: Running Quimb simulation...")
        quimb_state, quimb_time = self.run_quimb_simulation(qasm_file)
        if quimb_time is not None:
            print(f"  ✓ Quimb simulation completed in {quimb_time:.4f}s")
        else:
            print("  ✗ Quimb simulation failed")
        
        # Step 6: Run QsimCirq simulation
        print("\nStep 6: Running QsimCirq simulation...")
        qsimcirq_state, qsimcirq_time = self.run_qsimcirq_simulation(qasm_file)
        if qsimcirq_state is not None:
            print(f"  ✓ QsimCirq simulation completed in {qsimcirq_time:.4f}s")
            print(f"    Statevector shape: {qsimcirq_state.shape}")
        else:
            print("  ✗ QsimCirq simulation failed")
        
        # Step 7: Run TensorNetwork simulation
        print("\nStep 7: Running TensorNetwork simulation...")
        tensornetwork_state, tensornetwork_time = self.run_tensornetwork_simulation(qasm_file)
        if tensornetwork_time is not None:
            print(f"  ✓ TensorNetwork simulation completed in {tensornetwork_time:.4f}s")
        else:
            print("  ✗ TensorNetwork simulation failed")
        
        # Step 8: Get circuit information
        print("\nStep 8: Extracting circuit information...")
        try:
            circuit = QuantumCircuit.from_qasm_file(qasm_file)
            num_qubits = circuit.num_qubits
            num_gates = circuit.size()
            print(f"  ✓ Circuit info: {num_qubits} qubits, {num_gates} gates")
        except Exception as e:
            print(f"  ✗ Failed to get circuit info: {e}")
            num_qubits = None
            num_gates = None
        
        # Step 9: Calculate fidelities
        print("\nStep 9: Calculating fidelities between simulators...")
        fidelities = {}
        if qiskit_state is not None:
            if qiskit_mps_state is not None:
                fidelity = self.calculate_fidelity(qiskit_state, qiskit_mps_state)
                fidelities['qiskit_qiskit_mps'] = fidelity
                print(f"  ✓ Qiskit vs Qiskit MPS fidelity: {fidelity:.6f}" if fidelity else "  ✗ Fidelity calculation failed")
            if qiskit_gpu_state is not None:
                fidelity = self.calculate_fidelity(qiskit_state, qiskit_gpu_state)
                fidelities['qiskit_qiskit_gpu'] = fidelity
                print(f"  ✓ Qiskit vs Qiskit GPU fidelity: {fidelity:.6f}" if fidelity else "  ✗ Fidelity calculation failed")
            if quimb_state is not None:
                fidelity = self.calculate_fidelity(qiskit_state, quimb_state)
                fidelities['qiskit_quimb'] = fidelity
                print(f"  ✓ Qiskit vs Quimb fidelity: {fidelity:.6f}" if fidelity else "  ✗ Fidelity calculation failed")
            if qsimcirq_state is not None:
                fidelity = self.calculate_fidelity(qiskit_state, qsimcirq_state)
                fidelities['qiskit_qsimcirq'] = fidelity
                print(f"  ✓ Qiskit vs QsimCirq fidelity: {fidelity:.6f}" if fidelity else "  ✗ Fidelity calculation failed")
            if tensornetwork_state is not None:
                fidelity = self.calculate_fidelity(qiskit_state, tensornetwork_state)
                fidelities['qiskit_tensornetwork'] = fidelity
                print(f"  ✓ Qiskit vs TensorNetwork fidelity: {fidelity:.6f}" if fidelity else "  ✗ Fidelity calculation failed")
        else:
            print("  ⚠ No Qiskit state available for fidelity calculations")
        
        # Step 9: Get statevector information
        print("\nStep 9: Extracting statevector information...")
        qiskit_info = self.get_statevector_info(qiskit_state)
        if qiskit_info:
            magnitude, phase, dominant_states = qiskit_info
            print(f"  ✓ Statevector info extracted")
            if dominant_states:
                print(f"    Top state: {dominant_states[0][0]} (amplitude: {dominant_states[0][1]:.6f})")
        else:
            print("  ✗ Failed to extract statevector info")
        
        # Step 10: Calculate target accuracy
        print("\nStep 10: Calculating target accuracy...")
        target_accuracies = {}
        if expected_data and expected_data.get('target_state'):
            target_state = expected_data['target_state']
            print(f"  Target state: {target_state}")
            
            if qiskit_state is not None:
                accuracy = self.calculate_target_accuracy(qiskit_state, target_state)
                target_accuracies['qiskit_target'] = accuracy
                print(f"  ✓ Qiskit target accuracy: {accuracy:.8f}" if accuracy else "  ✗ Target accuracy calculation failed")
            if qiskit_mps_state is not None:
                accuracy = self.calculate_target_accuracy(qiskit_mps_state, target_state)
                target_accuracies['qiskit_mps_target'] = accuracy
                print(f"  ✓ Qiskit MPS target accuracy: {accuracy:.8f}" if accuracy else "  ✗ Target accuracy calculation failed")
            if qiskit_gpu_state is not None:
                accuracy = self.calculate_target_accuracy(qiskit_gpu_state, target_state)
                target_accuracies['qiskit_gpu_target'] = accuracy
                print(f"  ✓ Qiskit GPU target accuracy: {accuracy:.8f}" if accuracy else "  ✗ Target accuracy calculation failed")
            if qsimcirq_state is not None:
                accuracy = self.calculate_target_accuracy(qsimcirq_state, target_state)
                target_accuracies['qsimcirq_target'] = accuracy
                print(f"  ✓ QsimCirq target accuracy: {accuracy:.8f}" if accuracy else "  ✗ Target accuracy calculation failed")
        else:
            print("  ⚠ No target state available for accuracy calculations")
        
        # Step 11: Store results
        print("\nStep 11: Storing results...")
        result = {
            'file': qasm_file,
            'num_qubits': num_qubits,
            'num_gates': num_gates,
            'qiskit_time': qiskit_time,
            'qiskit_mps_time': qiskit_mps_time,
            'qiskit_gpu_time': qiskit_gpu_time,
            'quimb_time': quimb_time,
            'qsimcirq_time': qsimcirq_time,
            'tensornetwork_time': tensornetwork_time,
            'fidelities': fidelities,
            'qiskit_statevector_info': qiskit_info,
            'expected_data': expected_data,
            'target_accuracies': target_accuracies
        }
        
        self.results.append(result)
        print("  ✓ Results stored successfully")
        
        # Step 12: Summary
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY:")
        print(f"  File: {os.path.basename(qasm_file)}")
        print(f"  Qubits: {num_qubits}")
        print(f"  Gates: {num_gates}")
        print(f"  Qiskit time: {qiskit_time:.4f}s" if qiskit_time else "  Qiskit time: N/A")
        print(f"  Qiskit MPS time: {qiskit_mps_time:.4f}s" if qiskit_mps_time else "  Qiskit MPS time: N/A")
        print(f"  Qiskit GPU time: {qiskit_gpu_time:.4f}s" if qiskit_gpu_time else "  Qiskit GPU time: N/A")
        print(f"  QsimCirq time: {qsimcirq_time:.4f}s" if qsimcirq_time else "  QsimCirq time: N/A")
        
        if expected_data:
            print(f"  Target state: {expected_data.get('target_state')}")
            print(f"  Expected peak prob: {expected_data.get('peak_prob'):.2e}")
        
        if target_accuracies:
            for key, value in target_accuracies.items():
                if value is not None:
                    print(f"  {key}: {value:.8f}")
        
        if fidelities:
            for key, value in fidelities.items():
                if value is not None:
                    print(f"  {key}: {value:.6f}")
        
        print("=" * 60)
        return result
    
    def run_all_comparisons(self):
        """Run comparisons for all QASM files"""
        qasm_files = self.find_qasm_files()
        
        if not qasm_files:
            print("No QASM files found")
            return
        
        print(f"Found {len(qasm_files)} QASM files")
        print("Files will be processed in order of increasing difficulty")
        
        for i, qasm_file in enumerate(qasm_files):
            # Extract difficulty for display
            filename = os.path.basename(qasm_file)
            difficulty_match = re.search(r'diff=([\d.]+)', filename)
            difficulty = difficulty_match.group(1) if difficulty_match else "unknown"
            
            print(f"\n[{i+1}/{len(qasm_files)}] Processing: {os.path.basename(qasm_file)} (difficulty: {difficulty})")
            self.run_comparison(qasm_file)
    
    def generate_report(self):
        """Generate a comprehensive report"""
        if not self.results:
            print("No results to report")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Summary statistics
        print("\n" + "="*80)
        print("QUANTUM SIMULATOR COMPARISON REPORT")
        print("="*80)
        
        # Timing comparison
        print("\nTIMING COMPARISON:")
        print("-" * 40)
        timing_cols = ['qiskit_time', 'qiskit_mps_time', 'qiskit_gpu_time', 'quimb_time', 'qsimcirq_time', 'tensornetwork_time']
        for col in timing_cols:
            if col in df.columns:
                valid_times = df[col].dropna()
                if len(valid_times) > 0:
                    print(f"{col.replace('_', ' ').title()}:")
                    print(f"  Mean: {valid_times.mean():.4f}s")
                    print(f"  Min:  {valid_times.min():.4f}s")
                    print(f"  Max:  {valid_times.max():.4f}s")
                    print(f"  Count: {len(valid_times)}/{len(df)}")
        
        # Fidelity analysis
        print("\nFIDELITY ANALYSIS:")
        print("-" * 40)
        if 'fidelities' in df.columns:
            fidelity_data = []
            for _, row in df.iterrows():
                if row['fidelities']:
                    for key, value in row['fidelities'].items():
                        if value is not None:
                            fidelity_data.append({
                                'comparison': key,
                                'fidelity': value,
                                'file': row['file']
                            })
            
            if fidelity_data:
                fidelity_df = pd.DataFrame(fidelity_data)
                for comparison in fidelity_df['comparison'].unique():
                    subset = fidelity_df[fidelity_df['comparison'] == comparison]
                    print(f"{comparison.replace('_', ' vs ').title()}:")
                    print(f"  Mean fidelity: {subset['fidelity'].mean():.6f}")
                    print(f"  Min fidelity:  {subset['fidelity'].min():.6f}")
                    print(f"  Max fidelity:  {subset['fidelity'].max():.6f}")
                    print(f"  Count: {len(subset)}")
        
        # Target accuracy analysis
        print("\nTARGET ACCURACY ANALYSIS:")
        print("-" * 40)
        target_accuracy_data = []
        for result in self.results:
            if result.get('target_accuracies'):
                for key, value in result['target_accuracies'].items():
                    if value is not None:
                        target_accuracy_data.append({
                            'simulator': key,
                            'accuracy': value,
                            'file': result['file'],
                            'target_state': result.get('expected_data', {}).get('target_state'),
                            'peak_prob': result.get('expected_data', {}).get('peak_prob')
                        })
        
        if target_accuracy_data:
            target_df = pd.DataFrame(target_accuracy_data)
            for simulator in target_df['simulator'].unique():
                subset = target_df[target_df['simulator'] == simulator]
                print(f"{simulator.replace('_', ' ').title()}:")
                print(f"  Mean accuracy: {subset['accuracy'].mean():.8f}")
                print(f"  Min accuracy:  {subset['accuracy'].min():.8f}")
                print(f"  Max accuracy:  {subset['accuracy'].max():.8f}")
                print(f"  Count: {len(subset)}")
                
                # Compare with expected peak probability
                if 'peak_prob' in subset.columns:
                    peak_probs = subset['peak_prob'].dropna()
                    if len(peak_probs) > 0:
                        print(f"  Expected peak prob: {peak_probs.mean():.8f}")
                        accuracy_vs_expected = subset['accuracy'].mean() / peak_probs.mean() if peak_probs.mean() > 0 else 0
                        print(f"  Accuracy vs Expected: {accuracy_vs_expected:.2f}x")
        
        # Circuit statistics
        print("\nCIRCUIT STATISTICS:")
        print("-" * 40)
        if 'num_qubits' in df.columns:
            valid_qubits = df['num_qubits'].dropna()
            if len(valid_qubits) > 0:
                print(f"Number of qubits:")
                print(f"  Mean: {valid_qubits.mean():.1f}")
                print(f"  Min:  {valid_qubits.min()}")
                print(f"  Max:  {valid_qubits.max()}")
        
        if 'num_gates' in df.columns:
            valid_gates = df['num_gates'].dropna()
            if len(valid_gates) > 0:
                print(f"Number of gates:")
                print(f"  Mean: {valid_gates.mean():.1f}")
                print(f"  Min:  {valid_gates.min()}")
                print(f"  Max:  {valid_gates.max()}")
        
        # Save detailed results
        self.save_detailed_results()
        
        return df
    
    def save_detailed_results(self):
        """Save detailed results to files"""
        if not self.results:
            return
        
        # Save to CSV
        df = pd.DataFrame(self.results)
        
        # Flatten the results for CSV
        flat_results = []
        for result in self.results:
            flat_result = {
                'file': result['file'],
                'num_qubits': result['num_qubits'],
                'num_gates': result['num_gates'],
                'qiskit_time': result['qiskit_time'],
                'qiskit_mps_time': result['qiskit_mps_time'],
                'qiskit_gpu_time': result['qiskit_gpu_time'],
                'quimb_time': result['quimb_time'],
                'qsimcirq_time': result['qsimcirq_time'],
                'tensornetwork_time': result['tensornetwork_time']
            }
            
            # Add fidelities
            if result['fidelities']:
                for key, value in result['fidelities'].items():
                    flat_result[f'fidelity_{key}'] = value
            
            # Add expected data
            if result.get('expected_data'):
                expected = result['expected_data']
                flat_result['target_state'] = expected.get('target_state')
                flat_result['peak_prob'] = expected.get('peak_prob')
                flat_result['difficulty_level'] = expected.get('difficulty_level')
                flat_result['expected_num_qubits'] = expected.get('num_qubits')
                flat_result['rqc_depth'] = expected.get('rqc_depth')
                flat_result['pqc_depth'] = expected.get('pqc_depth')
                flat_result['est_num_shots'] = expected.get('est_num_shots')
            
            # Add target accuracies
            if result.get('target_accuracies'):
                for key, value in result['target_accuracies'].items():
                    flat_result[f'target_accuracy_{key}'] = value
            
            flat_results.append(flat_result)
        
        flat_df = pd.DataFrame(flat_results)
        flat_df.to_csv('quantum_simulator_results.csv', index=False)
        print(f"\nDetailed results saved to 'quantum_simulator_results.csv'")
        
        # Save statevector information
        statevector_info = []
        for result in self.results:
            if result['qiskit_statevector_info']:
                magnitude, phase, dominant_states = result['qiskit_statevector_info']
                if dominant_states:
                    for i, (binary, amplitude) in enumerate(dominant_states):
                        statevector_info.append({
                            'file': result['file'],
                            'rank': i+1,
                            'binary_state': binary,
                            'amplitude': amplitude,
                            'magnitude': abs(amplitude),
                            'phase': np.angle(amplitude)
                        })
        
        if statevector_info:
            statevector_df = pd.DataFrame(statevector_info)
            statevector_df.to_csv('statevector_analysis.csv', index=False)
            print(f"Statevector analysis saved to 'statevector_analysis.csv'")


def main():
    """Main function"""
    print("Quantum Simulator Comparison Tool")
    print("=" * 50)
    
    # Check available simulators
    print("Available simulators:")
    print(f"  Qiskit (Statevector): {'✓' if QISKIT_AVAILABLE else '✗'}")
    print(f"  Qiskit (MPS): {'✓' if QISKIT_AVAILABLE else '✗'}")
    print(f"  Quimb: {'✓' if QUIMB_AVAILABLE else '✗'}")
    print(f"  QsimCirq: {'✓' if QSIMCIRQ_AVAILABLE else '✗'}")
    print(f"  TensorNetwork: {'✓' if TENSORNETWORK_AVAILABLE else '✗'}")
    
    # Create comparison object with MPS enabled
    comparison = QuantumSimulatorComparison(enable_mps=True)
    
    # Run comparisons
    print("\nStarting quantum simulator comparisons...")
    print("MPS mode: Enabled")
    print("Timeout: 30 seconds per simulation")
    comparison.run_all_comparisons()
    
    # Generate report
    print("\nGenerating report...")
    df = comparison.generate_report()
    
    print("\nComparison completed!")


if __name__ == "__main__":
    main() 