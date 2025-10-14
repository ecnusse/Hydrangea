#!/usr/bin/env python3
"""
Comfrey Framework Usage Example
Demonstrates the three-stage error detection and repair workflow from the paper.

Section 4: Method Implementation
- Stage 1: Format Error Resolution
- Stage 2: Syntax Error Resolution  
- Stage 3: Repetition Error Resolution
"""

import sys
import os
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.comfrey_core import ComfreyFramework
from src.config import ComfreyConfig

def example_llm_function(text: str) -> str:
    """
    Example LLM function that might produce errors.
    This simulates an AI component that could have format, syntax, or repetition issues.
    """
    # Simulate LLM output with potential issues
    if "code" in text.lower():
        return """
def example_function():
    print("Hello World"
    return "result"
        """
    elif "json" in text.lower():
        return """
{
    "name": "John",
    "age": 30,
    "city": "New York"
    "country": "USA"
}
        """
    elif "repetition" in text.lower():
        return """
This is a test. This is a test. This is a test.
The same information is repeated multiple times.
The same information is repeated multiple times.
        """
    else:
        return "Normal output without issues."

def example_rag_function(query: str) -> str:
    """
    Example RAG function that might produce context construction issues.
    """
    # Simulate RAG output with irrelevant content
    return """
Document 1: Machine learning algorithms and their applications in data science.
Document 2: The history of ancient Rome and its emperors.
Document 3: Neural networks and deep learning techniques.
Document 4: Cooking recipes for Italian pasta dishes.
Document 5: Python programming best practices and code optimization.
        """

def main():
    """Demonstrate Comfrey framework usage according to paper methodology."""
    
    print("=== Comfrey Framework Demo ===\n")
    print("Implementing the three-stage workflow from paper Section 4:\n")
    
    # Initialize Comfrey with lightweight configuration for low overhead
    config = ComfreyConfig.create_lightweight_config()
    comfrey = ComfreyFramework(config)
    
    # Extract requirements from codebase (Section 4.2)
    print("1. Extracting requirements from codebase...")
    try:
        requirements = comfrey.extract_requirements_from_codebase(
            target_directory=".",
            entry_functions=["main", "example_llm_function", "example_rag_function"]
        )
        print(f"   SUCCESS Extracted {len(requirements)} requirement categories")
    except Exception as e:
        print(f"   WARNING Requirements extraction failed: {e}")
    
    print("\n2. Testing three-stage error detection and repair:\n")
    
    # Test 1: Format Error (Template discrepancy)
    print("Test 1: Format Error - Template Discrepancy")
    print("Input: 'code'")
    result1 = comfrey(example_llm_function)("code")
    print(f"Output: {result1[:100]}...")
    print(f"Statistics: {comfrey.get_statistics()}\n")
    
    # Test 2: Syntax Error (Syntax-parser misalignment)  
    print("Test 2: Syntax Error - Syntax-Parser Misalignment")
    print("Input: 'json'")
    result2 = comfrey(example_llm_function)("json")
    print(f"Output: {result2[:100]}...")
    print(f"Statistics: {comfrey.get_statistics()}\n")
    
    # Test 3: Repetition Error (Redundant semantics)
    print("Test 3: Repetition Error - Redundant Semantics")
    print("Input: 'repetition'")
    result3 = comfrey(example_llm_function)("repetition")
    print(f"Output: {result3[:100]}...")
    print(f"Statistics: {comfrey.get_statistics()}\n")
    
    # Test 4: Context Construction Error (Incorrect context construction)
    print("Test 4: Context Construction Error - Incorrect Context Construction")
    print("Input: 'machine learning'")
    result4 = comfrey(example_rag_function)("machine learning")
    print(f"Output: {result4[:100]}...")
    print(f"Statistics: {comfrey.get_statistics()}\n")
    
    # Final statistics
    print("=== Final Statistics ===")
    stats = comfrey.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Key Features Implemented ===")
    print("SUCCESS Three-stage workflow: Format → Syntax → Repetition")
    print("SUCCESS Low-overhead design with rule-based techniques first")
    print("SUCCESS Two-stage similarity detection (TF-IDF + embeddings)")
    print("SUCCESS Unified similarity threshold τ=0.7")
    print("SUCCESS Compiler/parser syntax validation")
    print("SUCCESS Unicode script detection for language consistency")
    print("SUCCESS History-based redundant software behavior detection")
    print("SUCCESS Requirement extraction from software expectations")

if __name__ == "__main__":
    main() 