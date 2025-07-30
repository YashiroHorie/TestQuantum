#!/usr/bin/env python3
"""
HPC Environment and ExaTN Check

This script checks for ExaTN availability in HPC environments
and provides detailed installation guidance.
"""

import os
import subprocess
import sys

def check_hpc_environment():
    """Check if we're in an HPC environment"""
    print("HPC Environment Check")
    print("=" * 40)
    
    hpc_indicators = []
    
    # Check for common HPC environment variables
    hpc_vars = ['SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID', 'SGE_TASK_ID']
    for var in hpc_vars:
        if os.environ.get(var):
            hpc_indicators.append(f"✓ {var} is set")
            print(f"✓ {var} is set")
    
    # Check for module command
    try:
        result = subprocess.run(['which', 'module'], capture_output=True, text=True)
        if result.returncode == 0:
            hpc_indicators.append("✓ module command available")
            print("✓ module command available")
        else:
            print("✗ module command not found")
    except:
        print("✗ module command not found")
    
    # Check for common HPC directories
    hpc_dirs = ['/usr/local', '/opt', '/sw', '/apps']
    for dir_path in hpc_dirs:
        if os.path.exists(dir_path):
            hpc_indicators.append(f"✓ {dir_path} exists")
            print(f"✓ {dir_path} exists")
    
    return len(hpc_indicators) > 0

def check_exatn_availability():
    """Check for ExaTN availability in various ways"""
    print("\nExaTN Availability Check")
    print("=" * 40)
    
    exatn_found = False
    
    # Method 1: Try to import exatn
    try:
        import exatn
        print("✓ ExaTN Python package is available")
        exatn_found = True
    except ImportError:
        print("✗ ExaTN Python package not found")
    
    # Method 2: Check for exatn executable
    try:
        result = subprocess.run(['which', 'exatn'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ ExaTN executable found: {result.stdout.strip()}")
            exatn_found = True
        else:
            print("✗ ExaTN executable not found")
    except:
        print("✗ Could not check for ExaTN executable")
    
    # Method 3: Check common installation directories
    exatn_dirs = [
        '/usr/local/exatn',
        '/opt/exatn',
        '/sw/exatn',
        '/apps/exatn',
        os.path.expanduser('~/exatn')
    ]
    
    for dir_path in exatn_dirs:
        if os.path.exists(dir_path):
            print(f"✓ ExaTN directory found: {dir_path}")
            exatn_found = True
    
    # Method 4: Check for module availability
    try:
        result = subprocess.run(['module', 'avail', 'exatn'], 
                              capture_output=True, text=True, timeout=10)
        if 'exatn' in result.stdout.lower():
            print("✓ ExaTN module available")
            exatn_found = True
        else:
            print("✗ ExaTN module not found")
    except:
        print("✗ Could not check module availability")
    
    return exatn_found

def check_compiler_availability():
    """Check for required compilers and dependencies"""
    print("\nCompiler and Dependency Check")
    print("=" * 40)
    
    compilers = {
        'gcc': 'GNU C++ Compiler',
        'g++': 'GNU C++ Compiler',
        'clang': 'Clang C++ Compiler',
        'clang++': 'Clang C++ Compiler',
        'mpicc': 'MPI C Compiler',
        'mpicxx': 'MPI C++ Compiler'
    }
    
    available_compilers = []
    
    for compiler, description in compilers.items():
        try:
            result = subprocess.run(['which', compiler], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {compiler} ({description}) found")
                available_compilers.append(compiler)
            else:
                print(f"✗ {compiler} ({description}) not found")
        except:
            print(f"✗ Could not check for {compiler}")
    
    return available_compilers

def check_mpi_availability():
    """Check for MPI availability"""
    print("\nMPI Availability Check")
    print("=" * 40)
    
    mpi_implementations = ['mpirun', 'mpiexec', 'srun']
    mpi_found = False
    
    for mpi_cmd in mpi_implementations:
        try:
            result = subprocess.run(['which', mpi_cmd], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {mpi_cmd} found")
                mpi_found = True
            else:
                print(f"✗ {mpi_cmd} not found")
        except:
            print(f"✗ Could not check for {mpi_cmd}")
    
    return mpi_found

def provide_installation_guidance():
    """Provide detailed installation guidance"""
    print("\n" + "=" * 60)
    print("EXATN INSTALLATION GUIDANCE")
    print("=" * 60)
    
    print("\n1. HPC Environment Installation:")
    print("   If you're on an HPC cluster:")
    print("   - Contact your system administrator")
    print("   - Check available modules: module avail")
    print("   - Look for exatn in module list")
    print("   - Common module names: exatn, tensor-networks, quantum")
    
    print("\n2. Building from Source (Advanced):")
    print("   Prerequisites:")
    print("   - C++17 compatible compiler (gcc >= 7, clang >= 5)")
    print("   - MPI implementation (OpenMPI, MPICH)")
    print("   - BLAS/LAPACK libraries")
    print("   - CMake >= 3.12")
    
    print("\n   Build Steps:")
    print("   git clone https://github.com/ORNL-QCI/exatn.git")
    print("   cd exatn")
    print("   mkdir build && cd build")
    print("   cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install")
    print("   make -j$(nproc)")
    print("   make install")
    
    print("\n3. Alternative Tensor Network Libraries:")
    print("   For similar functionality with easier installation:")
    print("   - Quimb: pip install quimb")
    print("   - TensorNetwork: pip install tensornetwork")
    print("   - ITensor: https://itensor.org/")
    print("   - These work well for most quantum simulation tasks")
    
    print("\n4. Python Integration:")
    print("   If ExaTN is installed but Python bindings are missing:")
    print("   - Check for exatn-python package")
    print("   - May need to set PYTHONPATH to include ExaTN Python modules")
    print("   - Alternative: Use ExaTN C++ API directly")

def main():
    """Main function"""
    print("HPC Environment and ExaTN Availability Check")
    print("=" * 60)
    
    # Check HPC environment
    is_hpc = check_hpc_environment()
    
    # Check ExaTN availability
    exatn_available = check_exatn_availability()
    
    # Check compilers
    compilers = check_compiler_availability()
    
    # Check MPI
    mpi_available = check_mpi_availability()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"HPC Environment: {'Yes' if is_hpc else 'No'}")
    print(f"ExaTN Available: {'Yes' if exatn_available else 'No'}")
    print(f"Compilers Available: {len(compilers)}")
    print(f"MPI Available: {'Yes' if mpi_available else 'No'}")
    
    if exatn_available:
        print("\n✅ ExaTN is available! You can use it in your quantum simulations.")
        print("   Try importing it in Python: import exatn")
    else:
        print("\n❌ ExaTN is not available.")
        if is_hpc:
            print("   Since you're in an HPC environment, contact your system administrator.")
        else:
            print("   Consider using alternative tensor network libraries.")
    
    # Provide guidance
    provide_installation_guidance()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if exatn_available:
        print("✅ Use ExaTN for high-performance tensor network simulations")
    elif is_hpc:
        print("⚠️  Contact HPC system administrator to install ExaTN")
    else:
        print("✅ Use alternative libraries: Quimb, TensorNetwork, or ITensor")
    
    print("✅ Continue with other simulators (Qiskit MPS, Qsim) for now")

if __name__ == "__main__":
    main() 