"""
Comfrey: Run-time Prevention of LLM Integration Failures

A framework for detecting and repairing format, syntax, and repetition errors 
in LLM outputs at runtime. Architecture designed for 
LLM-specific problems.

Usage:
    from comfrey import ComfreyFramework, ComfreyConfig
    
    # Create framework with default config
    comfrey = ComfreyFramework()
    
    # Use as decorator
    @comfrey
    def my_llm_function(prompt):
        return llm_call(prompt)
"""

from .comfrey_core import ComfreyFramework
from .config import ComfreyConfig
from .types import ErrorType, DetectionResult, RepairResult
from .syntax_detector import SyntaxDetector
from .format_detector import FormatDetector

__version__ = "1.0.0"
__author__ = "Comfrey Team"
__email__ = "comfrey@example.com"

__all__ = [
    "ComfreyFramework",
    "ComfreyConfig", 
    "ErrorType",
    "DetectionResult",
    "RepairResult",
    "SyntaxDetector",
    "FormatDetector"
]
