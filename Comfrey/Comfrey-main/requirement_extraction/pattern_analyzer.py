
import ast
import re
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Pattern
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

@dataclass
class RecognizedPattern:
    """Represents a recognized pattern in code"""
    pattern_type: 'PatternType'
    confidence: float
    location: str
    source_code: str
    extracted_details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RequirementSpec:
    """Represents a specific requirement specification"""
    requirement_type: str
    specification: Dict[str, Any]
    source_patterns: List[RecognizedPattern] = field(default_factory=list)
    confidence: float = 0.0

class PatternType(Enum):
    """Types of patterns that can be recognized"""
    JSON_SCHEMA = "json_schema"
    XML_SCHEMA = "xml_schema"
    POSITIONAL_TEMPLATE = "positional_template"
    TEXT_SEGMENTATION = "text_segmentation"
    COMPILATION_TARGET = "compilation_target"
    API_SIGNATURE = "api_signature"
    CONTEXT_ASSEMBLY = "context_assembly"
    ITERATION_PATTERN = "iteration_pattern"

@dataclass
class RecognizedPattern:
    """Represents a recognized pattern in code"""
    pattern_type: PatternType
    confidence: float
    location: str
    source_code: str
    extracted_details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RequirementSpec:
    """Specific requirement specification extracted from patterns"""
    requirement_type: str
    specification: Dict[str, Any]
    source_patterns: List[RecognizedPattern] = field(default_factory=list)
    confidence: float = 1.0

class PatternAnalyzer:

    def __init__(self):
        self.pattern_matchers = self._initialize_pattern_matchers()
        self.requirement_extractors = self._initialize_requirement_extractors()
        
    def analyze_consumption_points(self, 
                                 consumption_points: List[Dict[str, Any]]) -> List[RequirementSpec]:
        """
        Analyze consumption points to extract specific requirements.
        
        Args:
            consumption_points: List of LLM output consumption points from data flow analysis
            
        Returns:
            List of extracted requirement specifications
        """
        logger.info(f"Analyzing {len(consumption_points)} consumption points")
        
        requirement_specs = []
        
        for point in consumption_points:
            # Recognize patterns in the consumption point
            patterns = self._recognize_patterns(point)
            
            # Extract requirements from recognized patterns
            for pattern in patterns:
                specs = self._extract_requirements_from_pattern(pattern, point)
                requirement_specs.extend(specs)
        
        # Consolidate and deduplicate requirements
        consolidated_specs = self._consolidate_requirements(requirement_specs)
        
        logger.info(f"Extracted {len(consolidated_specs)} requirement specifications")
        return consolidated_specs
    
    def _recognize_patterns(self, consumption_point: Dict[str, Any]) -> List[RecognizedPattern]:

        patterns = []
        operation = consumption_point.get('operation', '')
        location = consumption_point.get('location', '')
        
        # Apply each pattern matcher
        for pattern_type, matcher in self.pattern_matchers.items():
            if matcher(operation, consumption_point):
                pattern = RecognizedPattern(
                    pattern_type=pattern_type,
                    confidence=self._calculate_pattern_confidence(pattern_type, operation),
                    location=location,
                    source_code=operation,
                    extracted_details=self._extract_pattern_details(pattern_type, operation, consumption_point)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_requirements_from_pattern(self, 
                                         pattern: RecognizedPattern,
                                         consumption_point: Dict[str, Any]) -> List[RequirementSpec]:
        """
        Extract specific requirements from a recognized pattern.
        """
        extractor = self.requirement_extractors.get(pattern.pattern_type)
        if not extractor:
            return []
        
        return extractor(pattern, consumption_point)
    
    def _consolidate_requirements(self, requirement_specs: List[RequirementSpec]) -> List[RequirementSpec]:
        """
        Consolidate and deduplicate similar requirements.
        """
        # Group requirements by type
        grouped_requirements = {}
        for spec in requirement_specs:
            req_type = spec.requirement_type
            if req_type not in grouped_requirements:
                grouped_requirements[req_type] = []
            grouped_requirements[req_type].append(spec)
        
        # Consolidate within each group
        consolidated = []
        for req_type, specs in grouped_requirements.items():
            if len(specs) == 1:
                consolidated.append(specs[0])
            else:
                # Merge similar specifications
                merged_spec = self._merge_similar_specs(specs)
                consolidated.append(merged_spec)
        
        return consolidated
    
    def _initialize_pattern_matchers(self) -> Dict[PatternType, callable]:

        return {
            PatternType.JSON_SCHEMA: self._match_json_pattern,
            PatternType.XML_SCHEMA: self._match_xml_pattern,
            PatternType.POSITIONAL_TEMPLATE: self._match_positional_template,
            PatternType.TEXT_SEGMENTATION: self._match_text_segmentation,
            PatternType.COMPILATION_TARGET: self._match_compilation_pattern,
            PatternType.API_SIGNATURE: self._match_api_signature,
            PatternType.CONTEXT_ASSEMBLY: self._match_context_assembly,
            PatternType.ITERATION_PATTERN: self._match_iteration_pattern
        }
    
    def _initialize_requirement_extractors(self) -> Dict[PatternType, callable]:
        """
        Initialize requirement extractors for each pattern type.
        """
        return {
            PatternType.JSON_SCHEMA: self._extract_json_requirements,
            PatternType.XML_SCHEMA: self._extract_xml_requirements,
            PatternType.POSITIONAL_TEMPLATE: self._extract_template_requirements,
            PatternType.TEXT_SEGMENTATION: self._extract_segmentation_requirements,
            PatternType.COMPILATION_TARGET: self._extract_compilation_requirements,
            PatternType.API_SIGNATURE: self._extract_api_requirements,
            PatternType.CONTEXT_ASSEMBLY: self._extract_context_requirements,
            PatternType.ITERATION_PATTERN: self._extract_iteration_requirements
        }
    
    def _match_json_pattern(self, operation: str, context: Dict[str, Any]) -> bool:
        """Match JSON processing patterns"""
        json_patterns = [
            r'json\.loads\s*\(',
            r'\.json\s*\(\)',
            r'json\.dumps\s*\(',
            r'json\.load\s*\(',
            r'loads\s*\(',
            r'JSONDecodeError',
            r'\.get\s*\(',  # Common JSON access pattern
            r'\[[\'"]\w+[\'\"]\]'  # Dictionary key access
        ]
        
        return any(re.search(pattern, operation, re.IGNORECASE) for pattern in json_patterns)
    
    def _match_xml_pattern(self, operation: str, context: Dict[str, Any]) -> bool:
        """Match XML processing patterns"""
        xml_patterns = [
            r'xml\.etree',
            r'BeautifulSoup',
            r'lxml',
            r'\.find\s*\(',
            r'\.findall\s*\(',
            r'\.tag\b',
            r'\.text\b',
            r'\.attrib\b',
            r'<\w+.*?>',  # XML tag pattern
            r'ElementTree'
        ]
        
        return any(re.search(pattern, operation, re.IGNORECASE) for pattern in xml_patterns)
    
    def _match_positional_template(self, operation: str, context: Dict[str, Any]) -> bool:
        """Match positional template patterns"""
        template_patterns = [
            r'\.format\s*\(',
            r'%\s*\(',
            r'f["\'].*\{.*\}.*["\']',  # f-string
            r'["\'].*\{.*\}.*["\']',   # format string
            r'\.replace\s*\(',
            r'\.substitute\s*\(',
            r'Template\s*\(',
            r'%\w+',  # % formatting
            r'\{\w+\}'  # {} formatting
        ]
        
        return any(re.search(pattern, operation, re.IGNORECASE) for pattern in template_patterns)
    
    def _match_text_segmentation(self, operation: str, context: Dict[str, Any]) -> bool:
        """Match text segmentation patterns"""
        segmentation_patterns = [
            r'\.split\s*\(',
            r'\.splitlines\s*\(',
            r'\.partition\s*\(',
            r'\.chunk\s*\(',
            r'textwrap\.',
            r'\.join\s*\(',
            r'len\s*\(',
            r'[:]\s*\d+',  # Slicing
            r'chunk_size',
            r'max_length',
            r'segment',
            r'boundary'
        ]
        
        return any(re.search(pattern, operation, re.IGNORECASE) for pattern in segmentation_patterns)
    
    def _match_compilation_pattern(self, operation: str, context: Dict[str, Any]) -> bool:
        """Match compilation/parsing patterns"""
        compilation_patterns = [
            r'compile\s*\(',
            r'ast\.parse\s*\(',
            r'exec\s*\(',
            r'eval\s*\(',
            r'subprocess\.',
            r'os\.system\s*\(',
            r'\.py\b',
            r'SyntaxError',
            r'ParseError',
            r'tokenize\.'
        ]
        
        return any(re.search(pattern, operation, re.IGNORECASE) for pattern in compilation_patterns)
    
    def _match_api_signature(self, operation: str, context: Dict[str, Any]) -> bool:
        """Match API signature patterns"""
        api_patterns = [
            r'def\s+\w+\s*\(',
            r'class\s+\w+',
            r'@\w+',  # Decorators
            r'\.method\s*\(',
            r'\.call\s*\(',
            r'signature',
            r'inspect\.',
            r'getattr\s*\(',
            r'hasattr\s*\(',
            r'callable\s*\('
        ]
        
        return any(re.search(pattern, operation, re.IGNORECASE) for pattern in api_patterns)
    
    def _match_context_assembly(self, operation: str, context: Dict[str, Any]) -> bool:
        """Match context assembly patterns"""
        context_patterns = [
            r'context',
            r'retrieve',
            r'embed',
            r'similarity',
            r'coherence',
            r'relevant',
            r'concatenate',
            r'merge',
            r'combine',
            r'assemble',
            r'\.append\s*\(',
            r'\.extend\s*\(',
            r'\+\s*'  # String concatenation
        ]
        
        return any(re.search(pattern, operation, re.IGNORECASE) for pattern in context_patterns)
    
    def _match_iteration_pattern(self, operation: str, context: Dict[str, Any]) -> bool:
        """Match iteration patterns"""
        iteration_patterns = [
            r'for\s+\w+\s+in',
            r'while\s+',
            r'enumerate\s*\(',
            r'\.items\s*\(\)',
            r'\.values\s*\(\)',
            r'\.keys\s*\(\)',
            r'range\s*\(',
            r'zip\s*\(',
            r'map\s*\(',
            r'filter\s*\(',
            r'list\s*\(',
            r'\.next\s*\(\)'
        ]
        
        return any(re.search(pattern, operation, re.IGNORECASE) for pattern in iteration_patterns)
    
    # Requirement extractors
    def _extract_json_requirements(self, pattern: RecognizedPattern, context: Dict[str, Any]) -> List[RequirementSpec]:
        """Extract JSON schema requirements"""
        requirements = []
        
        # Extract JSON schema hints from the code
        schema_hints = self._analyze_json_usage(pattern.source_code, context)
        
        json_requirement = RequirementSpec(
            requirement_type="json_template",
            specification={
                "format": "json",
                "schema": schema_hints.get("schema", {}),
                "required_keys": schema_hints.get("required_keys", []),
                "optional_keys": schema_hints.get("optional_keys", []),
                "value_types": schema_hints.get("value_types", {}),
                "validation_rules": schema_hints.get("validation_rules", [])
            },
            source_patterns=[pattern],
            confidence=pattern.confidence
        )
        requirements.append(json_requirement)
        
        return requirements
    
    def _extract_xml_requirements(self, pattern: RecognizedPattern, context: Dict[str, Any]) -> List[RequirementSpec]:
        """Extract XML schema requirements"""
        requirements = []
        
        # Extract XML schema hints
        schema_hints = self._analyze_xml_usage(pattern.source_code, context)
        
        xml_requirement = RequirementSpec(
            requirement_type="xml_template",
            specification={
                "format": "xml",
                "root_element": schema_hints.get("root_element", "root"),
                "required_elements": schema_hints.get("required_elements", []),
                "attributes": schema_hints.get("attributes", {}),
                "namespace": schema_hints.get("namespace"),
                "validation_rules": schema_hints.get("validation_rules", [])
            },
            source_patterns=[pattern],
            confidence=pattern.confidence
        )
        requirements.append(xml_requirement)
        
        return requirements
    
    def _extract_template_requirements(self, pattern: RecognizedPattern, context: Dict[str, Any]) -> List[RequirementSpec]:
        """Extract positional template requirements"""
        requirements = []
        
        # Extract template structure
        template_info = self._analyze_template_usage(pattern.source_code, context)
        
        template_requirement = RequirementSpec(
            requirement_type="positional_template",
            specification={
                "template_type": "positional",
                "identifiers": template_info.get("identifiers", []),
                "placeholders": template_info.get("placeholders", []),
                "format_style": template_info.get("format_style", "unknown"),
                "required_order": template_info.get("required_order", False),
                "validation_rules": template_info.get("validation_rules", [])
            },
            source_patterns=[pattern],
            confidence=pattern.confidence
        )
        requirements.append(template_requirement)
        
        return requirements
    
    def _extract_segmentation_requirements(self, pattern: RecognizedPattern, context: Dict[str, Any]) -> List[RequirementSpec]:
        """Extract text segmentation requirements"""
        requirements = []
        
        # Extract segmentation parameters
        seg_info = self._analyze_segmentation_usage(pattern.source_code, context)
        
        segmentation_requirement = RequirementSpec(
            requirement_type="text_segmentation",
            specification={
                "boundary_markers": seg_info.get("boundary_markers", []),
                "chunk_size_min": seg_info.get("chunk_size_min", 1),
                "chunk_size_max": seg_info.get("chunk_size_max", 1000),
                "overlap_size": seg_info.get("overlap_size", 0),
                "preserve_words": seg_info.get("preserve_words", True),
                "preserve_sentences": seg_info.get("preserve_sentences", True),
                "validation_rules": seg_info.get("validation_rules", [])
            },
            source_patterns=[pattern],
            confidence=pattern.confidence
        )
        requirements.append(segmentation_requirement)
        
        return requirements
    
    def _extract_compilation_requirements(self, pattern: RecognizedPattern, context: Dict[str, Any]) -> List[RequirementSpec]:
        """Extract compilation/syntax requirements"""
        requirements = []
        
        # Extract compilation parameters
        comp_info = self._analyze_compilation_usage(pattern.source_code, context)
        
        compilation_requirement = RequirementSpec(
            requirement_type="syntax_compliance",
            specification={
                "target_language": comp_info.get("target_language", "python"),
                "syntax_level": comp_info.get("syntax_level", "statement"),
                "allowed_constructs": comp_info.get("allowed_constructs", []),
                "forbidden_constructs": comp_info.get("forbidden_constructs", []),
                "validation_strict": comp_info.get("validation_strict", True),
                "validation_rules": comp_info.get("validation_rules", [])
            },
            source_patterns=[pattern],
            confidence=pattern.confidence
        )
        requirements.append(compilation_requirement)
        
        return requirements
    
    def _extract_api_requirements(self, pattern: RecognizedPattern, context: Dict[str, Any]) -> List[RequirementSpec]:
        """Extract API signature requirements"""
        requirements = []
        
        # Extract API signature information
        api_info = self._analyze_api_usage(pattern.source_code, context)
        
        api_requirement = RequirementSpec(
            requirement_type="api_signature",
            specification={
                "function_name": api_info.get("function_name", "unknown"),
                "parameter_types": api_info.get("parameter_types", {}),
                "return_type": api_info.get("return_type", "any"),
                "required_parameters": api_info.get("required_parameters", []),
                "optional_parameters": api_info.get("optional_parameters", []),
                "validation_rules": api_info.get("validation_rules", [])
            },
            source_patterns=[pattern],
            confidence=pattern.confidence
        )
        requirements.append(api_requirement)
        
        return requirements
    
    def _extract_context_requirements(self, pattern: RecognizedPattern, context: Dict[str, Any]) -> List[RequirementSpec]:
        """Extract context assembly requirements"""
        requirements = []
        
        # Extract context assembly parameters
        ctx_info = self._analyze_context_usage(pattern.source_code, context)
        
        context_requirement = RequirementSpec(
            requirement_type="context_coherence",
            specification={
                "coherence_threshold": ctx_info.get("coherence_threshold", 0.7),
                "similarity_metric": ctx_info.get("similarity_metric", "cosine"),
                "context_window": ctx_info.get("context_window", 5),
                "max_context_length": ctx_info.get("max_context_length", 2000),
                "relevance_threshold": ctx_info.get("relevance_threshold", 0.5),
                "validation_rules": ctx_info.get("validation_rules", [])
            },
            source_patterns=[pattern],
            confidence=pattern.confidence
        )
        requirements.append(context_requirement)
        
        return requirements
    
    def _extract_iteration_requirements(self, pattern: RecognizedPattern, context: Dict[str, Any]) -> List[RequirementSpec]:
        """Extract iteration pattern requirements"""
        requirements = []
        
        # Extract iteration parameters
        iter_info = self._analyze_iteration_usage(pattern.source_code, context)
        
        iteration_requirement = RequirementSpec(
            requirement_type="iteration_behavior",
            specification={
                "iteration_type": iter_info.get("iteration_type", "sequential"),
                "expected_item_type": iter_info.get("expected_item_type", "any"),
                "min_items": iter_info.get("min_items", 0),
                "max_items": iter_info.get("max_items", float('inf')),
                "break_conditions": iter_info.get("break_conditions", []),
                "validation_rules": iter_info.get("validation_rules", [])
            },
            source_patterns=[pattern],
            confidence=pattern.confidence
        )
        requirements.append(iteration_requirement)
        
        return requirements
    
    # Helper methods for pattern analysis
    def _calculate_pattern_confidence(self, pattern_type: PatternType, operation: str) -> float:
        """Calculate confidence score for pattern recognition"""
        base_confidence = 0.8
        
        # Boost confidence for explicit patterns
        if pattern_type == PatternType.JSON_SCHEMA and 'json' in operation.lower():
            base_confidence = 0.95
        elif pattern_type == PatternType.XML_SCHEMA and ('xml' in operation.lower() or 'beautifulsoup' in operation.lower()):
            base_confidence = 0.95
        elif pattern_type == PatternType.COMPILATION_TARGET and ('compile' in operation.lower() or 'ast.parse' in operation.lower()):
            base_confidence = 0.9
        
        # Reduce confidence for ambiguous patterns
        if len(operation) < 10:
            base_confidence *= 0.8
        
        return min(base_confidence, 1.0)
    
    def _extract_pattern_details(self, pattern_type: PatternType, operation: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed information about a recognized pattern"""
        details = {}
        
        if pattern_type == PatternType.JSON_SCHEMA:
            details['json_methods'] = re.findall(r'json\.\w+', operation)
            details['key_accesses'] = re.findall(r'\[[\'"]\w+[\'\"]\]', operation)
        elif pattern_type == PatternType.XML_SCHEMA:
            details['xml_methods'] = re.findall(r'\.(?:find|findall|tag|text|attrib)', operation)
        elif pattern_type == PatternType.TEXT_SEGMENTATION:
            details['split_methods'] = re.findall(r'\.(?:split|splitlines|partition)', operation)
            details['size_constraints'] = re.findall(r'\d+', operation)
        
        return details
    
    # Analysis methods for extracting specific information
    def _analyze_json_usage(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze JSON usage to extract schema requirements"""
        hints = {
            "schema": {},
            "required_keys": [],
            "optional_keys": [],
            "value_types": {},
            "validation_rules": []
        }
        
        # Extract key accesses
        key_accesses = re.findall(r'\[[\'"]([\w_]+)[\'\"]\]', code)
        hints["required_keys"] = list(set(key_accesses))
        
        # Extract get() calls (optional keys)
        get_calls = re.findall(r'\.get\s*\(\s*[\'"]([\w_]+)[\'\"]\s*[,\)]', code)
        hints["optional_keys"] = list(set(get_calls))
        
        return hints
    
    def _analyze_xml_usage(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze XML usage to extract schema requirements"""
        hints = {
            "root_element": "root",
            "required_elements": [],
            "attributes": {},
            "namespace": None,
            "validation_rules": []
        }
        
        # Extract find/findall calls
        find_calls = re.findall(r'\.find(?:all)?\s*\(\s*[\'"]([\w_]+)[\'\"]\s*\)', code)
        hints["required_elements"] = list(set(find_calls))
        
        return hints
    
    def _analyze_template_usage(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze template usage to extract template requirements"""
        hints = {
            "identifiers": [],
            "placeholders": [],
            "format_style": "unknown",
            "required_order": False,
            "validation_rules": []
        }
        
        # Extract format placeholders
        format_placeholders = re.findall(r'\{(\w+)\}', code)
        hints["placeholders"] = list(set(format_placeholders))
        
        # Determine format style
        if '.format(' in code:
            hints["format_style"] = "str.format"
        elif 'f"' in code or "f'" in code:
            hints["format_style"] = "f-string"
        elif '%' in code:
            hints["format_style"] = "percent"
        
        return hints
    
    def _analyze_segmentation_usage(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze segmentation usage to extract segmentation requirements"""
        hints = {
            "boundary_markers": [],
            "chunk_size_min": 1,
            "chunk_size_max": 1000,
            "overlap_size": 0,
            "preserve_words": True,
            "preserve_sentences": True,
            "validation_rules": []
        }
        
        # Extract split delimiters
        split_delims = re.findall(r'\.split\s*\(\s*[\'"](.*?)[\'\"]\s*\)', code)
        hints["boundary_markers"] = list(set(split_delims))
        
        # Extract size constraints
        size_numbers = re.findall(r'\d+', code)
        if size_numbers:
            hints["chunk_size_max"] = max(int(n) for n in size_numbers)
        
        return hints
    
    def _analyze_compilation_usage(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze compilation usage to extract syntax requirements"""
        hints = {
            "target_language": "python",
            "syntax_level": "statement",
            "allowed_constructs": [],
            "forbidden_constructs": [],
            "validation_strict": True,
            "validation_rules": []
        }
        
        # Determine target language
        if 'compile(' in code:
            hints["target_language"] = "python"
        elif 'ast.parse' in code:
            hints["target_language"] = "python"
            hints["syntax_level"] = "expression"
        
        return hints
    
    def _analyze_api_usage(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze API usage to extract signature requirements"""
        hints = {
            "function_name": "unknown",
            "parameter_types": {},
            "return_type": "any",
            "required_parameters": [],
            "optional_parameters": [],
            "validation_rules": []
        }
        
        # Extract function name
        func_match = re.search(r'def\s+(\w+)\s*\(', code)
        if func_match:
            hints["function_name"] = func_match.group(1)
        
        return hints
    
    def _analyze_context_usage(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context usage to extract coherence requirements"""
        hints = {
            "coherence_threshold": 0.7,
            "similarity_metric": "cosine",
            "context_window": 5,
            "max_context_length": 2000,
            "relevance_threshold": 0.5,
            "validation_rules": []
        }
        
        # Extract threshold values
        threshold_matches = re.findall(r'(\d+\.?\d*)', code)
        if threshold_matches:
            hints["coherence_threshold"] = float(threshold_matches[0])
        
        return hints
    
    def _analyze_iteration_usage(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze iteration usage to extract iteration requirements"""
        hints = {
            "iteration_type": "sequential",
            "expected_item_type": "any",
            "min_items": 0,
            "max_items": float('inf'),
            "break_conditions": [],
            "validation_rules": []
        }
        
        # Determine iteration type
        if 'for' in code.lower():
            hints["iteration_type"] = "for_loop"
        elif 'while' in code.lower():
            hints["iteration_type"] = "while_loop"
        elif 'enumerate' in code.lower():
            hints["iteration_type"] = "enumerated"
        
        return hints
    
    def _merge_similar_specs(self, specs: List[RequirementSpec]) -> RequirementSpec:
        """Merge similar requirement specifications"""
        if not specs:
            return None
        
        # Use the first spec as base
        merged = specs[0]
        
        # Merge source patterns
        all_patterns = []
        for spec in specs:
            all_patterns.extend(spec.source_patterns)
        
        # Calculate average confidence
        avg_confidence = sum(spec.confidence for spec in specs) / len(specs)
        
        # Merge specifications (simple approach - could be more sophisticated)
        merged_specification = merged.specification.copy()
        for spec in specs[1:]:
            for key, value in spec.specification.items():
                if key not in merged_specification:
                    merged_specification[key] = value
                elif isinstance(value, list) and isinstance(merged_specification[key], list):
                    merged_specification[key] = list(set(merged_specification[key] + value))
        
        return RequirementSpec(
            requirement_type=merged.requirement_type,
            specification=merged_specification,
            source_patterns=all_patterns,
            confidence=avg_confidence
        ) 