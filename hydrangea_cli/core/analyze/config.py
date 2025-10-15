"""
Lightweight configuration used by Hydrangea.
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class Config:
    """Minimal configuration subset for the format detector"""

    # General
    enable_format_detection: bool = True

    # Format detection
    element_threshold: int = 3
    enable_fsa_validation: bool = True
    enable_code_fenced_templates: bool = True

    # Context/construction and similarity
    similarity_threshold: float = 0.7
    enable_embedding_similarity: bool = False

    # Performance
    enable_early_termination: bool = True
    max_processing_time_ms: int = 500

    # Logging
    log_level: str = "INFO"
    enable_detailed_logging: bool = False

    @classmethod
    def create_lightweight_config(cls) -> 'Config':
        return cls(
            enable_embedding_similarity=False,
            enable_detailed_logging=False,
            max_processing_time_ms=500,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}


