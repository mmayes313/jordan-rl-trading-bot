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

    # Run pytest
    print("🧪 Running unit tests...")
    result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'],
                           capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

    if result.returncode == 0:
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
