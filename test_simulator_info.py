#!/usr/bin/env python3
"""
Simulator Information and Installation Guide

This script provides information about available quantum simulators
and installation instructions for missing ones.
"""

import sys

def check_simulator_availability():
    """Check which simulators are available and provide installation info"""
    print("Quantum Simulator Availability Check")
    print("=" * 50)
    
    simulators = {}
    
    # Check Qiskit
    try:
        import qiskit
        from qiskit_aer import AerSimulator
        simulators['Qiskit MPS'] = "✓ Available"
        print("✓ Qiskit MPS: Available")
    except ImportError as e:
        simulators['Qiskit MPS'] = f"✗ Not available: {e}"
        print(f"✗ Qiskit MPS: Not available - {e}")
    
    # Check Qsim
    try:
        import qsimcirq
        import cirq
        simulators['Qsim'] = "✓ Available"
        print("✓ Qsim: Available")
    except ImportError as e:
        simulators['Qsim'] = f"✗ Not available: {e}"
        print(f"✗ Qsim: Not available - {e}")
    
    # Check Quimb
    try:
        import quimb.tensor as qtn
        simulators['Quimb'] = "✓ Available"
        print("✓ Quimb: Available")
    except ImportError as e:
        simulators['Quimb'] = f"✗ Not available: {e}"
        print(f"✗ Quimb: Not available - {e}")
    
    # Check Cotengra
    try:
        import cotengra
        simulators['Cotengra'] = "✓ Available"
        print("✓ Cotengra: Available")
    except ImportError as e:
        simulators['Cotengra'] = f"✗ Not available: {e}"
        print(f"✗ Cotengra: Not available - {e}")
    
    # Check ExaTN
    try:
        import exatn
        simulators['ExaTN'] = "✓ Available"
        print("✓ ExaTN: Available")
    except ImportError as e:
        simulators['ExaTN'] = f"✗ Not available: {e}"
        print(f"✗ ExaTN: Not available - {e}")
    
    return simulators

def print_installation_guide():
    """Print installation instructions for missing simulators"""
    print("\n" + "=" * 50)
    print("INSTALLATION GUIDE")
    print("=" * 50)
    
    print("\n1. Qiskit MPS (Matrix Product State):")
    print("   pip install qiskit qiskit-aer")
    print("   Note: MPS method is included in qiskit-aer")
    
    print("\n2. Qsim (Google's Quantum Simulator):")
    print("   pip install qsimcirq cirq")
    print("   Note: Requires Python 3.7+ and compatible with most platforms")
    
    print("\n3. Quimb (Tensor Networks):")
    print("   pip install quimb")
    print("   Note: Core tensor network library")
    
    print("\n4. Cotengra (Tensor Network Optimization):")
    print("   pip install cotengra")
    print("   Note: Tensor network contraction optimization")
    
    print("\n5. ExaTN (Exascale Tensor Networks):")
    print("   Note: ExaTN requires special installation:")
    print("   - Requires C++ compiler and MPI")
    print("   - May need to build from source")
    print("   - Check: https://github.com/ORNL-QCI/exatn")
    print("   - Alternative: Use other tensor network libraries")

def print_working_combinations():
    """Print working simulator combinations"""
    print("\n" + "=" * 50)
    print("WORKING SIMULATOR COMBINATIONS")
    print("=" * 50)
    
    print("\n✅ Recommended (Fully Working):")
    print("   - Qiskit MPS + Qsim")
    print("   - Provides good comparison between different approaches")
    print("   - Both support full state vector simulation")
    
    print("\n⚠️  Limited (Partial Support):")
    print("   - Quimb + Cotengra (tensor network approach)")
    print("   - Good for research but complex setup")
    print("   - May have limitations with certain gates")
    
    print("\n❌ Not Available:")
    print("   - ExaTN (requires special installation)")
    print("   - May be available in HPC environments")

def print_test_recommendations():
    """Print recommendations for testing"""
    print("\n" + "=" * 50)
    print("TESTING RECOMMENDATIONS")
    print("=" * 50)
    
    print("\n1. For Quick Testing:")
    print("   python test_working_simulators.py")
    print("   - Tests Qiskit MPS vs Qsim")
    print("   - Provides meaningful comparisons")
    
    print("\n2. For Full Testing:")
    print("   python test_multiple_simulators.py")
    print("   - Tests all available simulators")
    print("   - Handles missing simulators gracefully")
    
    print("\n3. For Setup Verification:")
    print("   python test_simulator_setup.py")
    print("   - Verifies all simulators work correctly")
    print("   - Tests basic functionality")

def main():
    """Main function"""
    print("Quantum Simulator Information and Setup Guide")
    print("=" * 60)
    
    # Check availability
    simulators = check_simulator_availability()
    
    # Count available simulators
    available_count = sum(1 for status in simulators.values() if "✓" in status)
    total_count = len(simulators)
    
    print(f"\nSummary: {available_count}/{total_count} simulators available")
    
    # Print installation guide
    print_installation_guide()
    
    # Print working combinations
    print_working_combinations()
    
    # Print test recommendations
    print_test_recommendations()
    
    # Final recommendations
    print("\n" + "=" * 50)
    print("FINAL RECOMMENDATIONS")
    print("=" * 50)
    
    if available_count >= 2:
        print("✅ You have enough simulators for meaningful comparison!")
        print("   Run: python test_working_simulators.py")
    elif available_count == 1:
        print("⚠️  You have one simulator available.")
        print("   Consider installing additional simulators for comparison.")
    else:
        print("❌ No simulators available.")
        print("   Please install at least one simulator from the guide above.")
    
    print(f"\nAvailable simulators: {available_count}")
    print("Ready for testing: {'Yes' if available_count >= 1 else 'No'}")

if __name__ == "__main__":
    main() 