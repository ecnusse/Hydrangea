"""
Type definitions for lightweight Comfrey subset used by Hydrangea.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional


class ErrorType(Enum):
    """Types of errors that the detector can report"""
    FORMAT_TEMPLATE_DISCREPANCY = "format_template_discrepancy"
    FORMAT_DATA_SEGMENTATION = "format_data_segmentation"
    FORMAT_CONTEXT_CONSTRUCTION = "format_context_construction"


@dataclass
class DetectionResult:
    """Result of error detection"""
    error_type: ErrorType
    detected: bool
    severity: float
    details: Dict[str, Any]
    suggested_repair: Optional[str] = None


