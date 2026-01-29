#!/usr/bin/env python3
"""
Test script for get_scientific_name() function from download_genomes_multi.py
"""

import subprocess
import sys
import os
from loguru import logger

# Add the script directory to path to import the function
sys.path.insert(0, os.path.dirname(__file__))

# Import the function from download_genomes_multi
from download_genomes_multi import get_scientific_name

def test_get_scientific_name():
    """Test get_scientific_name() with various taxids"""
    
    # Configure logger for testing
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
    
    # Test cases: (taxid, expected_name_or_partial)
    test_cases = [
        ("9606", "Homo sapiens"),      # Human
        ("10090", "Mus musculus"),      # Mouse
        ("7227", "Drosophila"),        # Fruit fly
        ("6239", "Caenorhabditis"),    # C. elegans
        ("3702", "Arabidopsis"),       # Thale cress
    ]
    
    print("=" * 60)
    print("Testing get_scientific_name() function")
    print("=" * 60)
    
    results = []
    for taxid, expected in test_cases:
        print(f"\nTesting taxid: {taxid} (expected to contain: {expected})")
        try:
            sci_name = get_scientific_name(taxid, retries=3)
            if sci_name:
                print(f"  ✓ SUCCESS: {sci_name}")
                if expected.lower() in sci_name.lower():
                    print(f"  ✓ Name contains expected text")
                else:
                    print(f"  ⚠ Warning: Name doesn't contain expected text")
                results.append((taxid, True, sci_name))
            else:
                print(f"  ✗ FAILED: Function returned None")
                results.append((taxid, False, None))
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            results.append((taxid, False, str(e)))
    
    # Test with invalid taxid
    print(f"\nTesting with invalid taxid: 99999999")
    try:
        sci_name = get_scientific_name("99999999", retries=3)
        if sci_name:
            print(f"  ⚠ Unexpected: Got result {sci_name} for invalid taxid")
        else:
            print(f"  ✓ Expected: Function returned None for invalid taxid")
    except Exception as e:
        print(f"  ✓ Expected error for invalid taxid: {type(e).__name__}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    print(f"Successful: {successful}/{total}")
    
    for taxid, success, result in results:
        status = "✓" if success else "✗"
        print(f"  {status} {taxid}: {result}")

if __name__ == "__main__":
    test_get_scientific_name()


