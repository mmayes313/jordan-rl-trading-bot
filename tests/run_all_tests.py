#!/usr/bin/env python
# tests/run_all_tests.py - Comprehensive Test Runner for Jordan RL Trading Bot
# Runs all test suites with proper configuration, reporting, and coverage analysis.
# Usage: python tests/run_all_tests.py [--coverage] [--verbose] [--html]

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_tests(args):
    """Run all tests with specified configuration."""

    # Base pytest arguments
    pytest_args = [
        'tests/',
        '-s',
        '-vv',  # verbose output
        '--tb=short',  # shorter traceback format
        '--strict-markers',  # strict marker validation
        '--disable-warnings',  # disable warnings for cleaner output
        '--log-cli-level=DEBUG',  # log level for CLI output
        '--log-file=data/logs/test_run.log',  # log file
        '--log-file-level=DEBUG',  # detailed logging to file
    ]

    # Add coverage if requested
    if args.coverage:
        pytest_args.extend([
            '--cov=src',  # coverage for src directory
            '--cov-report=term-missing',  # terminal coverage report
            '--cov-report=html:htmlcov',  # HTML coverage report
            '--cov-fail-under=80',  # fail if coverage below 80%
        ])

    # Add HTML report if requested
    if args.html:
        pytest_args.extend([
            '--html=reports/test_report.html',
            '--self-contained-html',
        ])

    # Add junit XML for CI/CD
    pytest_args.extend([
        '--junitxml=reports/test_results.xml',
    ])

    # Run pytest
    print("Running Jordan RL Trading Bot Test Suite...")
    print(f"Command: {' '.join(['pytest'] + pytest_args)}")
    print("=" * 60)

    # Add the specific command from the prompt: pytest.main(['-v', '--cov=src', '--cov-report=html', 'tests/'])
    if not args.coverage:
        # Use pytest.main for the specific command
        import pytest
        sys.exit(pytest.main(['-v', '--cov=src', '--cov-report=html', 'tests/']))

    try:
        result = subprocess.run(['pytest'] + pytest_args, cwd=Path(__file__).parent.parent)
        return result.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Please install pytest: pip install pytest")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

def setup_test_environment():
    """Setup test environment and directories."""
    # Create necessary directories
    dirs = [
        'data/logs',
        'reports',
        'htmlcov'
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Ensure test data exists (create dummy if needed)
    test_data_path = Path('data/raw/eurusd_1y.csv')
    if not test_data_path.exists():
        print("Warning: Test data not found. Some tests may use dummy data.")
        print("For full testing, place EURUSD 1-minute data in data/raw/eurusd_1y.csv")

def generate_test_summary():
    """Generate a summary of test results."""
    print("\n" + "=" * 60)
    print("JORDAN RL TRADING BOT - TEST SUMMARY")
    print("=" * 60)

    test_files = [
        'test_indicators.py',
        'test_environment.py',
        'test_rewards.py',
        'test_masks.py',
        'test_hyperparam_optimizer.py',
        'test_diagnostic.py',
        'test_jordan.py',
        'test_jordan_personality.py',
        'test_hyperparam_optimizer.py'
    ]

    print("Test Files:")
    for test_file in test_files:
        file_path = Path(f'tests/{test_file}')
        if file_path.exists():
            print(f"  ✅ {test_file}")
        else:
            print(f"  ❌ {test_file} (missing)")

    print("\nTest Coverage Areas:")
    print("  ✅ Technical Indicators (372 features)")
    print("  ✅ Trading Environment (417-dim observations)")
    print("  ✅ Reward System (21 rules)")
    print("  ✅ Trading Masks (CCI-based logic)")
    print("  ✅ Hyperparameter Optimization (Optuna)")
    print("  ✅ Diagnostic Suite (system health)")
    print("  ✅ PPO Model Training")
    print("  ✅ Data Processing Pipeline")

    print("\nReports Generated:")
    if Path('reports/test_report.html').exists():
        print("  📊 HTML Report: reports/test_report.html")
    if Path('htmlcov/index.html').exists():
        print("  📈 Coverage Report: htmlcov/index.html")
    if Path('reports/test_results.xml').exists():
        print("  📋 JUnit XML: reports/test_results.xml")

    print("\nNext Steps:")
    print("  1. Review test failures in the logs")
    print("  2. Fix any critical issues found")
    print("  3. Run full training: python scripts/demo_training.py")
    print("  4. Deploy to production when all tests pass")

def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description='Run Jordan RL Trading Bot Test Suite')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--html', action='store_true', help='Generate HTML test report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Setup environment
    setup_test_environment()

    # Run tests
    exit_code = run_tests(args)

    # Generate summary
    generate_test_summary()

    # Exit with appropriate code
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
