#!/usr/bin/env python3
"""
Test Difficulty Sorting

This script tests that QASM files are sorted by difficulty level
in ascending order (smallest difficulty first).
"""

import os
import re
from quantum_simulator_comparison import QuantumSimulatorComparison

def test_difficulty_sorting():
    """Test that files are sorted by difficulty"""
    print("Testing Difficulty-Based File Sorting")
    print("=" * 50)
    
    # Create comparison object
    comparison = QuantumSimulatorComparison()
    
    # Get sorted files
    qasm_files = comparison.find_qasm_files()
    
    if not qasm_files:
        print("No QASM files found")
        return
    
    print(f"Found {len(qasm_files)} QASM files")
    print("\nFiles sorted by difficulty (ascending):")
    print("-" * 50)
    
    # Extract and display difficulty for each file
    difficulties = []
    for i, qasm_file in enumerate(qasm_files):
        filename = os.path.basename(qasm_file)
        difficulty_match = re.search(r'diff=([\d.]+)', filename)
        difficulty = float(difficulty_match.group(1)) if difficulty_match else float('inf')
        difficulties.append(difficulty)
        
        print(f"{i+1:2d}. {filename} (difficulty: {difficulty})")
    
    # Verify sorting
    print("\n" + "=" * 50)
    print("VERIFICATION:")
    
    # Check if difficulties are in ascending order
    is_sorted = all(difficulties[i] <= difficulties[i+1] for i in range(len(difficulties)-1))
    
    if is_sorted:
        print("✓ Files are correctly sorted by difficulty (ascending)")
    else:
        print("✗ Files are NOT correctly sorted by difficulty")
        
        # Find the first violation
        for i in range(len(difficulties)-1):
            if difficulties[i] > difficulties[i+1]:
                print(f"  Violation at position {i+1}: {difficulties[i]} > {difficulties[i+1]}")
                break
    
    # Show difficulty statistics
    valid_difficulties = [d for d in difficulties if d != float('inf')]
    if valid_difficulties:
        print(f"\nDifficulty Statistics:")
        print(f"  Min difficulty: {min(valid_difficulties)}")
        print(f"  Max difficulty: {max(valid_difficulties)}")
        print(f"  Number of files with difficulty: {len(valid_difficulties)}")
        print(f"  Number of files without difficulty: {len(difficulties) - len(valid_difficulties)}")
    
    print("\nDifficulty sorting test completed!")

def test_first_few_files():
    """Test the first few files to show they're the easiest"""
    print("\n" + "=" * 50)
    print("Testing First Few Files (Should be Easiest)")
    print("=" * 50)
    
    comparison = QuantumSimulatorComparison()
    qasm_files = comparison.find_qasm_files()
    
    if not qasm_files:
        print("No QASM files found")
        return
    
    print("First 5 files (should be lowest difficulty):")
    print("-" * 40)
    
    for i, qasm_file in enumerate(qasm_files[:5]):
        filename = os.path.basename(qasm_file)
        difficulty_match = re.search(r'diff=([\d.]+)', filename)
        difficulty = difficulty_match.group(1) if difficulty_match else "unknown"
        
        print(f"{i+1}. {filename}")
        print(f"   Difficulty: {difficulty}")
        
        # Show some file info
        file_size = os.path.getsize(qasm_file)
        print(f"   Size: {file_size:,} bytes")
        print()

if __name__ == "__main__":
    test_difficulty_sorting()
    test_first_few_files() 