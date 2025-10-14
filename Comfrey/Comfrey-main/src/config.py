"""
Configuration module for Comfrey framework.
Configuration settings for LLM output validation and repair.
Implements low-overhead design principles from the paper.
All default values are set exactly as specified in the paper.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class ComfreyConfig:
    """Configuration class for Comfrey framework"""
    
    # General settings
    enable_format_detection: bool = True
    enable_syntax_detection: bool = True
    enable_repetition_detection: bool = True
    
    # History and caching - low overhead design
    history_window_size: int = 10  # N from paper - configurable and set to 10 by default
    enable_repair_caching: bool = True
    
    # Format detection settings - from paper Section 4.3.1
    element_threshold: int = 3  # τ_element from paper - threshold for missing/extraneous elements (3 by default)
    enable_fsa_validation: bool = True  # Finite state automata validation
    enable_code_fenced_templates: bool = True  # Support for code-fenced templates
    
    # Syntax detection settings - from paper Section 4.4
    enable_ast_analysis: bool = True  # AST refinement with minimal edit distance
    enable_bytecode_analysis: bool = True
    max_repair_iterations: int = 3  # As specified in paper - retries up to 3 times
    repair_iteration_limit: int = 3  # As specified in paper - retries up to 3 times
    
    # Repetition detection settings - from paper Section 4.5
    # Unified similarity threshold τ=0.7 as specified in paper for both context construction and semantic redundancy
    similarity_threshold: float = 0.7  # τ from paper - unified threshold for all similarity detection
    enable_embedding_similarity: bool = True  # Only used when necessary for low overhead design
    embedding_model_size: str = "0.6B"  # As mentioned in paper for low overhead - only when necessary
    
    # Context construction settings - from paper Section 4.3.3
    # Two-stage similarity detection: TF-IDF first, then sentence embeddings for low-scoring pairs
    enable_lightweight_coherence: bool = True  # Two-stage similarity detection as described in paper
    bottom_quartile_threshold: float = 0.25  # Bottom quartile threshold for second-stage examination
    
    # Semantic redundancy settings - from paper Section 4.5.2
    # Note: Paper mentions τ_internal=0.8 as default in comments, but uses unified τ=0.7
    internal_redundancy_threshold: float = 0.7  # τ_internal - using unified threshold τ=0.7
    contextual_redundancy_threshold: float = 0.4  # τ_contextual from paper comments (set to 0.4 as default)
    
    # Coherence settings - from paper comments
    coherence_alpha: float = 0.7  # α from paper comments (default: 0.7) emphasizes topical consistency
    coherence_beta: float = 0.6  # β from paper comments (default: 0.6) for query similarity
    coherence_gamma: float = 0.4  # γ from paper comments (default: 0.4) for density
    
    # Content overlap threshold - from paper comments
    content_overlap_threshold: float = 0.7  # 70% from paper comments - flagging segment pairs with overlap exceeding 70%
    
    # Performance settings - low overhead design
    enable_early_termination: bool = True  # Iteration-aware termination mechanism
    max_processing_time_ms: int = 1000
    
    # Logging and debugging
    log_level: str = "INFO"
    enable_detailed_logging: bool = False
    save_repair_history: bool = True
    
    # Lexical features settings - from paper Section 4.4.2
    enable_unicode_detection: bool = True  # Unicode script detection
    enable_ngram_analysis: bool = True  # n-gram frequency analysis
    enable_language_standard_check: bool = True  # Dictionary-based language standard check
    
    # Redundant software behavior settings - from paper Section 4.5.1
    # History queue size N=10 as specified in paper
    history_queue_size: int = 10  # N from paper - configurable and set to 10 by default
    deterministic_function_patterns: List[str] = None  # Will be initialized in __post_init__
    non_deterministic_function_patterns: List[str] = None  # Will be initialized in __post_init__
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Initialize function patterns for redundant software behavior detection
        # Based on LLM internal knowledge and heuristics as mentioned in paper
        if self.deterministic_function_patterns is None:
            self.deterministic_function_patterns = [
                'read', 'parse', 'fetch', 'get', 'load', 'extract', 'validate',
                'search', 'query', 'find', 'lookup', 'retrieve'
            ]
        
        if self.non_deterministic_function_patterns is None:
            self.non_deterministic_function_patterns = [
                'random', 'time', 'date', 'uuid', 'generate', 'create', 'write',
                'update', 'delete', 'modify', 'save', 'store'
            ]
        
        # Validate thresholds
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.bottom_quartile_threshold <= 1.0):
            raise ValueError("bottom_quartile_threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.internal_redundancy_threshold <= 1.0):
            raise ValueError("internal_redundancy_threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.contextual_redundancy_threshold <= 1.0):
            raise ValueError("contextual_redundancy_threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.content_overlap_threshold <= 1.0):
            raise ValueError("content_overlap_threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.coherence_alpha <= 1.0):
            raise ValueError("coherence_alpha must be between 0.0 and 1.0")
        
        if not (0.0 <= self.coherence_beta <= 1.0):
            raise ValueError("coherence_beta must be between 0.0 and 1.0")
        
        if not (0.0 <= self.coherence_gamma <= 1.0):
            raise ValueError("coherence_gamma must be between 0.0 and 1.0")
    
    @classmethod
    def create_lightweight_config(cls) -> 'ComfreyConfig':
        """Create a lightweight configuration for minimal overhead as described in paper"""
        return cls(
            enable_embedding_similarity=False,  # Avoid computation-intensive techniques
            enable_detailed_logging=False,
            history_window_size=5,
            max_repair_iterations=3,
            enable_lightweight_coherence=True,  # Use rule-based techniques first
            max_processing_time_ms=500  # Faster processing
        )
    
    @classmethod  
    def create_comprehensive_config(cls) -> 'ComfreyConfig':
        """Create a comprehensive configuration for maximum detection capability"""
        return cls(
            enable_format_detection=True,
            enable_syntax_detection=True,
            enable_repetition_detection=True,
            enable_embedding_similarity=True,  # Use when necessary
            enable_detailed_logging=True,
            history_window_size=20,
            max_repair_iterations=5,
            save_repair_history=True
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ComfreyConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict) 