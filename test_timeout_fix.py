#!/usr/bin/env python3
"""
Test script to verify the threading-based timeout works correctly
"""

import time
from quantum_simulator_comparison import run_with_timeout, TimeoutError

def slow_function():
    """A function that takes a long time"""
    print("    Starting slow function...")
    time.sleep(35)  # This should timeout after 30 seconds
    print("    Slow function completed (this should not print)")
    return "success"

def fast_function():
    """A function that completes quickly"""
    print("    Starting fast function...")
    time.sleep(2)
    print("    Fast function completed")
    return "success"

@run_with_timeout(timeout_seconds=30)
def test_slow():
    return slow_function()

@run_with_timeout(timeout_seconds=30)
def test_fast():
    return fast_function()

def main():
    print("Testing Threading-Based Timeout")
    print("=" * 40)
    
    print("\n1. Testing fast function (should complete):")
    start_time = time.time()
    result = test_fast()
    end_time = time.time()
    print(f"   Result: {result}")
    print(f"   Time taken: {end_time - start_time:.2f}s")
    
    print("\n2. Testing slow function (should timeout):")
    start_time = time.time()
    result = test_slow()
    end_time = time.time()
    print(f"   Result: {result}")
    print(f"   Time taken: {end_time - start_time:.2f}s")
    
    print("\nTimeout test completed!")
    print("If the slow function timed out correctly, you should see:")
    print("- '⏰ Timeout after 30s' message")
    print("- Result should be (None, None)")
    print("- Total time should be around 30 seconds")

if __name__ == "__main__":
    main() 