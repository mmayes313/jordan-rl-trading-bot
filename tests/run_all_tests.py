#!/usr/bin/env python3
"""
Run all tests for the Jordan RL Trading Bot.
"""

import subprocess
import sys
import os

def run_setup():
    """Run the setup script to ensure dependencies are installed."""
    print("� Running setup to ensure dependencies are installed...")
    try:
        result = subprocess.run([sys.executable, 'setup.py'],
                               check=True, capture_output=False, timeout=600)  # 10 minute timeout
        return True
    except subprocess.TimeoutExpired:
        print("⏰ Setup timed out after 10 minutes")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Setup failed: {e}")
        return False

def run_tests():
    """Run all test suites."""
    print("🚀 Running Full Integration Tests...")

    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    # Run setup first
    if not run_setup():
        print("❌ Cannot run tests without proper setup")
        return False
    print()

    # Run pytest with diagnostic logging
    print("🧪 Running unit tests...")
    print("📋 Including Jordan's Diagnostic Test Suite...")

    # Run diagnostic test first with detailed logging
    diag_result = subprocess.run([
        sys.executable, '-m', 'pytest',
        'tests/test_diagnostic.py',
        '-v', '--log-cli-level=INFO'
    ], capture_output=True, text=True)

    print("=== DIAGNOSTIC TEST RESULTS ===")
    print(diag_result.stdout)
    if diag_result.stderr:
        print("Diagnostic Errors:", diag_result.stderr)

    # Run all other tests
    print("\n=== ALL TESTS ===")
    result = subprocess.run([
        sys.executable, '-m', 'pytest',
        'tests/', '-v', '--ignore=tests/test_diagnostic.py'
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

    # Check overall success
    diag_success = diag_result.returncode == 0
    other_success = result.returncode == 0

    if diag_success and other_success:
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        if not diag_success:
            print("   - Diagnostic test failed")
        if not other_success:
            print("   - Other tests failed")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
