#!/usr/bin/env python3
"""Run all test scripts for the RAG benchmark platform."""

import os
import sys
import logging
import subprocess
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test script configurations
TEST_SCRIPTS = [
    {
        "name": "Query Service Test",
        "script": "test_query_service.py",
        "options": ["enhancement", "strategies", "full"]
    },
    {
        "name": "Orchestration Service Test",
        "script": "test_orchestration.py",
        "options": ["strategies", "comparison", "complex", "custom"]
    },
    {
        "name": "Evaluation Service Test",
        "script": "test_evaluation.py",
        "options": ["individual", "multiple", "real"]
    }
]


def run_test(script_path, option=None):
    """Run a test script with an optional specific test."""
    cmd = [sys.executable, script_path]
    if option:
        cmd.append(option)
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        duration = time.time() - start_time
        
        logger.info(f"Completed in {duration:.2f}s with return code: {result.returncode}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Test failed with return code: {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False


def run_all_tests(specific_test=None):
    """Run all test scripts or a specific test if specified."""
    start_time = time.time()
    results = {}
    
    logger.info(f"Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for test_config in TEST_SCRIPTS:
        test_name = test_config["name"]
        script = test_config["script"]
        
        if specific_test and specific_test not in script:
            continue
        
        logger.info(f"Running test suite: {test_name}")
        
        if specific_test and "." in specific_test:
            # Run a specific subtest
            test_parts = specific_test.split(".")
            if test_parts[0] in script:
                option = test_parts[1] if len(test_parts) > 1 else None
                success = run_test(script, option)
                results[specific_test] = success
        else:
            # Run all options or the whole script
            options = test_config.get("options", [])
            if options and not specific_test:
                sub_results = {}
                for option in options:
                    sub_results[option] = run_test(script, option)
                results[test_name] = sub_results
            else:
                results[test_name] = run_test(script)
    
    # Print summary
    total_duration = time.time() - start_time
    logger.info(f"All tests completed in {total_duration:.2f}s")
    
    logger.info("Test Results Summary:")
    all_passed = True
    
    for test_name, result in results.items():
        if isinstance(result, dict):
            sub_results = []
            for subtest, success in result.items():
                sub_results.append(f"{subtest}: {'✅' if success else '❌'}")
                all_passed = all_passed and success
            logger.info(f"{test_name}: {', '.join(sub_results)}")
        else:
            logger.info(f"{test_name}: {'✅' if result else '❌'}")
            all_passed = all_passed and result
    
    return all_passed


if __name__ == "__main__":
    """Run the test script."""
    specific_test = sys.argv[1] if len(sys.argv) > 1 else None
    
    if specific_test:
        logger.info(f"Running specific test: {specific_test}")
    else:
        logger.info("Running all tests")
    
    success = run_all_tests(specific_test)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 