"""
Type definitions for Comfrey framework.
Separated to avoid circular imports.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional

class ErrorType(Enum):
    """Types of errors that Comfrey can detect and repair"""
    FORMAT_TEMPLATE_DISCREPANCY = "format_template_discrepancy"
    FORMAT_DATA_SEGMENTATION = "format_data_segmentation"  
    FORMAT_CONTEXT_CONSTRUCTION = "format_context_construction"
    SYNTAX_PARSER_MISALIGNMENT = "syntax_parser_misalignment"
    SYNTAX_LEXICAL_INCONSISTENCY = "syntax_lexical_inconsistency"
    REPETITION_SOFTWARE_BEHAVIOR = "repetition_software_behavior"
    REPETITION_SEMANTICS = "repetition_semantics"

@dataclass
class DetectionResult:
    """Result of error detection"""
    error_type: ErrorType
    detected: bool
    severity: float
    details: Dict[str, Any]
    suggested_repair: Optional[str] = None

@dataclass
class RepairResult:
    """Result of error repair"""
    success: bool
    original_output: str
    repaired_output: str
    repair_actions: List[str]
    metadata: Dict[str, Any] 