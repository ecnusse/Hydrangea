"""
Requirement Extraction Module for Comfrey Framework

This module implements the "Obtaining the Requirements" functionality from the Comfrey paper,
based on SmartGear's control_flow_analysis architecture.

Two types of requirements are handled:
1. Software expectations: Extracted from code through static analysis
2. Application scenarios: Human-defined requirements based on user expectations

Main Components:
- RequirementExtractor: Main extraction engine using static analysis
- DataFlowAnalyzer: Tracks LLM outputs through data flow analysis
- PatternAnalyzer: Recognizes patterns in code to extract requirements
- ScenarioRequirementManager: Manages human-defined scenario requirements
"""

from .requirement_extractor import RequirementExtractor, RequirementExtractionResult, ExtractedRequirement, RequirementType
from .data_flow_analyzer import DataFlowAnalyzer, DataFlowPath, LLMOutputConsumer
from .pattern_analyzer import PatternAnalyzer, PatternType, RecognizedPattern, RequirementSpec
from .scenario_requirements import ScenarioRequirementManager, ScenarioRequirement, ScenarioDimension

__all__ = [
    # Main classes
    'RequirementExtractor',
    'DataFlowAnalyzer', 
    'PatternAnalyzer',
    'ScenarioRequirementManager',
    
    # Data classes
    'RequirementExtractionResult',
    'ExtractedRequirement',
    'DataFlowPath',
    'LLMOutputConsumer',
    'RecognizedPattern',
    'RequirementSpec',
    'ScenarioRequirement',
    
    # Enums
    'RequirementType',
    'PatternType',
    'ScenarioDimension',
] 