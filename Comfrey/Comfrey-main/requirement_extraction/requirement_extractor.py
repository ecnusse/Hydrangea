
import os
import sys
import ast
import json
import logging
import collections
import jedi
import networkx as nx
import beniget
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

# Try to import pyan3
try:
    import pyan
    PYAN_AVAILABLE = True
except ImportError:
    PYAN_AVAILABLE = False
    logging.warning("pyan3 not available, using fallback call graph analysis")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.types import ErrorType
from src.config import ComfreyConfig

logger = logging.getLogger(__name__)

class RequirementType(Enum):
    """Types of requirements that can be extracted"""
    OUTPUT_TEMPLATE = "output_template"
    DATA_CHUNK_SPEC = "data_chunk_specification"
    CONTEXT_CONSTRUCTION = "context_construction"
    PROGRAMMING_LANGUAGE = "programming_language"
    API_SIGNATURE = "api_signature"
    SCENARIO_BASED = "scenario_based"

@dataclass
class ExtractedRequirement:
    """Represents a single extracted requirement"""
    type: RequirementType
    source_location: str  # file:line
    pattern: str
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RequirementExtractionResult:
    """Result of requirement extraction process"""
    requirements: List[ExtractedRequirement] = field(default_factory=list)
    call_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    llm_output_consumers: List[str] = field(default_factory=list)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)

class RequirementExtractor:

    def __init__(self, config: ComfreyConfig):
        self.config = config
        # Use NetworkX to build directed graph, providing more powerful graph analysis capabilities
        self.call_graph = nx.DiGraph()
        self.data_flow_graph = nx.DiGraph()
        self.llm_output_consumers = []
        self.extracted_requirements = []
        
        # Pattern matchers for different requirement types
        self.template_patterns = self._initialize_template_patterns()
        self.syntax_patterns = self._initialize_syntax_patterns()
        self.segmentation_patterns = self._initialize_segmentation_patterns()
        
    def extract_requirements_from_codebase(self, 
                                         target_directory: str,
                                         entry_functions: List[str] = None) -> RequirementExtractionResult:
        """
        Extract requirements from entire codebase using static analysis.
        
        Implements the algorithm described in Comfrey paper Section 3.2:
        1. Traverse function call graph
        2. Conduct data flow analysis
        3. Identify LLM output consumption points
        4. Extract requirements through pattern analysis
        """
        logger.info(f"Starting requirement extraction from {target_directory}")
        
        # Step 1: Build function call graph
        self._build_call_graph(target_directory, entry_functions)
        
        # Step 2: Conduct data flow analysis
        self._analyze_data_flow()
        
        # Step 3: Identify LLM output consumers
        self._identify_llm_output_consumers()
        
        # Step 4: Extract requirements from consumer code
        self._extract_requirements_from_consumers()
        
        # Step 5: Note - scenario requirements are handled separately
        # They are human-defined, not extracted from code
        self._add_scenario_requirements()
        
        result = RequirementExtractionResult(
            requirements=self.extracted_requirements,
            call_graph=self.call_graph,
            llm_output_consumers=self.llm_output_consumers,
            analysis_metadata={
                "total_functions_analyzed": len(self.call_graph),
                "llm_consumers_found": len(self.llm_output_consumers),
                "requirements_extracted": len(self.extracted_requirements)
            }
        )
        
        logger.info(f"Extraction complete: {len(self.extracted_requirements)} requirements found")
        return result
    
    def _build_call_graph(self, target_directory: str, entry_functions: List[str] = None):
        """
        Build function call graph using professional tools.
        Prioritize Pyan3, fallback to AST analysis.
        """
        logger.debug("Building call graph with professional tools")
        
        # Try to use Pyan3 to build high-quality call graph
        if PYAN_AVAILABLE:
            success = self._build_call_graph_with_pyan3(target_directory)
            if success:
                logger.info("Successfully built call graph with Pyan3")
                return
        
        # Fallback to enhanced AST analysis
        logger.info("Using enhanced AST analysis for call graph")
        self._build_call_graph_with_ast(target_directory)
    
    def _build_call_graph_with_pyan3(self, target_directory: str) -> bool:
        """Use Pyan3 to build professional call graph"""
        try:
            # Collect all Python files
            python_files = []
            for root, dirs, files in os.walk(target_directory):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
            
            if not python_files:
                return False
            
            # Use Pyan3 analysis
            analyzer = pyan.CallGraphVisitor(python_files)
            analyzer.visit_all()
            
            # Convert to NetworkX graph
            for edge in analyzer.get_edges():
                self.call_graph.add_edge(edge.source, edge.target)
            
            # Extract LLM related information
            self._extract_llm_info_from_pyan3(analyzer)
            
            logger.info(f"Pyan3 analysis complete: {len(self.call_graph.nodes())} nodes, {len(self.call_graph.edges())} edges")
            return True
            
        except Exception as e:
            logger.error(f"Pyan3 analysis failed: {e}")
            return False
    
    def _build_call_graph_with_ast(self, target_directory: str):
        """Use enhanced AST analysis to build call graph"""
        import os
        python_files = []
        for root, dirs, files in os.walk(target_directory):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=file_path)
                visitor = CallGraphVisitor(file_path, self)
                visitor.visit(tree)
                
                # Add to NetworkX graph
                for func_name, calls in visitor.call_graph.items():
                    for call in calls:
                        self.call_graph.add_edge(func_name, call)
                
                # Use Beniget to analyze def-use chains
                self._analyze_def_use_chains_for_file(file_path, tree)
                
            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")
                continue
    
    def _extract_llm_info_from_pyan3(self, analyzer):
        """Extract LLM related information from Pyan3 analyzer"""
        # This needs to be implemented based on Pyan3's specific API
        # Temporarily use simple pattern matching
        llm_patterns = ['openai', 'gpt', 'claude', 'llm', 'generate']
        
        for node in analyzer.get_nodes():
            if any(pattern in node.name.lower() for pattern in llm_patterns):
                # Mark as LLM related node
                self.llm_output_consumers.append(node.name)
    
    def _analyze_def_use_chains_for_file(self, file_path: str, tree: ast.AST):
        """Use Beniget to analyze def-use chains for a single file"""
        try:
            duc = beniget.DefUseChains()
            duc.visit(tree)
            
            # Analyze usage chains of LLM output variables
            for var_name in self.llm_output_consumers:
                if var_name in duc.chains:
                    chain = duc.chains[var_name]
                    # Analyze variable usage patterns
                    usage_patterns = self._analyze_variable_usage_patterns(chain)
                    logger.debug(f"Variable {var_name} usage patterns: {usage_patterns}")
                    
        except Exception as e:
            logger.debug(f"Beniget analysis failed {file_path}: {e}")
    
    def _analyze_variable_usage_patterns(self, chain) -> Dict[str, Any]:
        """Analyze variable usage patterns"""
        return {
            'usage_count': len(chain.users) if hasattr(chain, 'users') else 0,
            'definition_count': len(chain.assigners) if hasattr(chain, 'assigners') else 0,
            'chain_length': len(chain.nodes) if hasattr(chain, 'nodes') else 0
        }
    
    def _analyze_data_flow(self):
        """
        Analyze data flow to track LLM outputs through the call graph.
        Enhanced to better identify LLM output consumption points.
        """
        logger.debug("Analyzing data flow")
        
        # Identify LLM output variables
        llm_output_vars = set()
        
        for func_name in self.call_graph.nodes():
            # Check for direct LLM API calls
            callees = list(self.call_graph.successors(func_name))
            for callee in callees:
                if self._is_llm_api_call(callee):
                    # If it's in variable assignment form (function::variable)
                    if '::' in func_name:
                        var_name = func_name.split('::')[1]
                        llm_output_vars.add(var_name)
                        logger.debug(f"Added LLM output variable: {var_name}")
                    else:
                        # Try to infer variable name from function name
                        var_name = self._infer_output_variable(func_name, callee)
                        if var_name:
                            llm_output_vars.add(var_name)
                            logger.debug(f"Inferred LLM output variable: {var_name}")
        
        # Build data flow graph
        self._build_data_flow_graph(llm_output_vars)
        
        logger.debug(f"Found {len(llm_output_vars)} LLM output variables")
    
    def _identify_llm_output_consumers(self):
        """
        Identify all code locations where LLM outputs are consumed.
        Enhanced to find real consumption points in the code.
        """
        logger.debug("Identifying LLM output consumers")
        
        # Traverse all functions to find LLM output usage
        for func_name in self.call_graph.nodes():
            # Skip LLM output variable records
            if '::' in func_name:
                continue
                
            # Check if function has signs of LLM output processing
            callees = list(self.call_graph.successors(func_name))
            for callee in callees:
                if self._is_llm_output_processing(callee):
                    consumer_info = {
                        'function': func_name,
                        'location': f"{func_name}",
                        'operation': callee,
                        'context': {'type': 'llm_output_processing'}
                    }
                    self.llm_output_consumers.append(consumer_info)
                    logger.debug(f"Found LLM output consumer: {func_name} -> {callee}")
    
    def _is_llm_output_processing(self, operation: str) -> bool:
        """Check if operation is LLM output processing"""
        processing_patterns = [
            # JSON/XML processing
            'json.loads', 'json.dumps', '.json()', 'json.load', 'json.dump',
            'xml.etree', 'BeautifulSoup', 'xmltodict', 'yaml.load', 'yaml.dump',
            
            # String processing
            '.split(', '.strip(', '.replace(', '.format(', '.join(', '.lower(', '.upper(',
            '.startswith(', '.endswith(', '.find(', '.index(', 'regex', 're.search', 're.match',
            
            # Parsing and compilation
            'ast.parse', 'compile(', 'exec(', 'eval(', 'subprocess', 'os.system',
            
            # File operations
            'open(', '.write(', '.read(', '.save(', '.load(',
            
            # Data processing
            'pandas', '.DataFrame', '.to_csv', '.to_json', '.to_dict',
            
            # Template processing
            'jinja2', '.render(', 'Template', '.substitute(',
            
            # Validation and checking
            'validate', 'check', 'verify', 'assert', 'isinstance', 'hasattr',
            
            # Output and display
            'print(', 'log', 'display', 'show', 'render'
        ]
        
        operation_lower = operation.lower()
        return any(pattern in operation_lower for pattern in processing_patterns)
    
    def _extract_requirements_from_consumers(self):
        """
        Extract specific requirements from consumer code snippets.
        Enhanced to extract more types of requirements from real consumption points.
        """
        logger.debug("Extracting requirements from consumer code")
        
        for consumer in self.llm_output_consumers:
            operation = consumer.get('operation', '')
            
            # Extract different types of requirements
            self._extract_template_requirements(consumer)
            self._extract_segmentation_requirements(consumer)
            self._extract_context_requirements(consumer)
            self._extract_syntax_requirements(consumer)
            self._extract_validation_requirements(consumer)
            self._extract_format_requirements(consumer)
    
    def _extract_template_requirements(self, consumer: Dict[str, Any]):
        """
        Extract output template requirements from parsing code.
        Identifies positional templates and structured-data templates.
        """
        operation = consumer.get('operation', '')
        location = consumer.get('location', '')
        
        # Check for JSON parsing
        if 'json.loads' in operation or '.json()' in operation:
            requirement = ExtractedRequirement(
                type=RequirementType.OUTPUT_TEMPLATE,
                source_location=location,
                pattern='json_schema',
                confidence=0.9,
                details={
                    'template_type': 'structured_data',
                    'format': 'json',
                    'schema_hints': self._extract_json_schema_hints(consumer)
                }
            )
            self.extracted_requirements.append(requirement)
        
        # Check for XML parsing
        elif 'xml.etree' in operation or 'BeautifulSoup' in operation:
            requirement = ExtractedRequirement(
                type=RequirementType.OUTPUT_TEMPLATE,
                source_location=location,
                pattern='xml_schema',
                confidence=0.9,
                details={
                    'template_type': 'structured_data',
                    'format': 'xml',
                    'schema_hints': self._extract_xml_schema_hints(consumer)
                }
            )
            self.extracted_requirements.append(requirement)
        
        # Check for positional templates
        elif self._has_positional_template_pattern(operation):
            requirement = ExtractedRequirement(
                type=RequirementType.OUTPUT_TEMPLATE,
                source_location=location,
                pattern='positional_template',
                confidence=0.7,
                details={
                    'template_type': 'positional',
                    'identifiers': self._extract_positional_identifiers(consumer)
                }
            )
            self.extracted_requirements.append(requirement)
    
    def _extract_segmentation_requirements(self, consumer: Dict[str, Any]):
        """
        Extract data chunk specifications from segmentation operations.
        """
        operation = consumer.get('operation', '')
        location = consumer.get('location', '')
        
        # Check for text splitting operations
        if any(pattern in operation for pattern in ['.split(', '.splitlines(', 'textwrap.', 'chunk']):
            requirement = ExtractedRequirement(
                type=RequirementType.DATA_CHUNK_SPEC,
                source_location=location,
                pattern='text_segmentation',
                confidence=0.8,
                details={
                    'boundary_markers': self._extract_boundary_markers(consumer),
                    'chunk_size_constraints': self._extract_chunk_constraints(consumer),
                    'completeness_criteria': self._extract_completeness_criteria(consumer)
                }
            )
            self.extracted_requirements.append(requirement)
    
    def _extract_context_requirements(self, consumer: Dict[str, Any]):
        """
        Extract context construction rules from context assembly operations.
        """
        operation = consumer.get('operation', '')
        location = consumer.get('location', '')
        
        # Check for context construction patterns
        if any(pattern in operation for pattern in ['context', 'retrieve', 'embed', 'similarity']):
            requirement = ExtractedRequirement(
                type=RequirementType.CONTEXT_CONSTRUCTION,
                source_location=location,
                pattern='semantic_coherence',
                confidence=0.7,
                details={
                    'coherence_threshold': self._extract_coherence_threshold(consumer),
                    'similarity_metric': self._extract_similarity_metric(consumer),
                    'context_window': self._extract_context_window(consumer)
                }
            )
            self.extracted_requirements.append(requirement)
    
    def _extract_syntax_requirements(self, consumer: Dict[str, Any]):
        """
        Extract programming language specifications and API signatures.
        """
        operation = consumer.get('operation', '')
        location = consumer.get('location', '')
        
        # Check for compilation/parsing operations
        if any(pattern in operation for pattern in ['compile(', 'ast.parse', 'exec(', 'eval(']):
            requirement = ExtractedRequirement(
                type=RequirementType.PROGRAMMING_LANGUAGE,
                source_location=location,
                pattern='python_syntax',
                confidence=0.9,
                details={
                    'language': 'python',
                    'syntax_level': self._extract_syntax_level(consumer),
                    'api_signatures': self._extract_api_signatures(consumer)
                }
            )
            self.extracted_requirements.append(requirement)
    
    def _extract_validation_requirements(self, consumer: Dict[str, Any]):
        """Extract requirements from validation operations"""
        operation = consumer.get('operation', '')
        location = consumer.get('location', '')
        
        if any(pattern in operation.lower() for pattern in ['validate', 'check', 'verify', 'assert', 'isinstance']):
            requirement = ExtractedRequirement(
                type=RequirementType.OUTPUT_TEMPLATE,
                source_location=location,
                pattern='validation_constraint',
                confidence=0.8,
                details={
                    'validation_type': 'data_validation',
                    'constraint_pattern': operation,
                    'requirement_source': 'validation_code'
                }
            )
            self.extracted_requirements.append(requirement)
            logger.debug(f"Extracted validation requirement from {location}")
    
    def _extract_format_requirements(self, consumer: Dict[str, Any]):
        """Extract requirements from formatting operations"""
        operation = consumer.get('operation', '')
        location = consumer.get('location', '')
        
        # Check various format processing
        if any(pattern in operation.lower() for pattern in ['json', 'xml', 'yaml', 'csv']):
            format_type = 'json' if 'json' in operation.lower() else \
                         'xml' if 'xml' in operation.lower() else \
                         'yaml' if 'yaml' in operation.lower() else \
                         'csv' if 'csv' in operation.lower() else 'structured'
            
            requirement = ExtractedRequirement(
                type=RequirementType.OUTPUT_TEMPLATE,
                source_location=location,
                pattern=f'{format_type}_format',
                confidence=0.9,
                details={
                    'format_type': format_type,
                    'processing_operation': operation,
                    'requirement_source': 'format_processing'
                }
            )
            self.extracted_requirements.append(requirement)
            logger.debug(f"Extracted format requirement: {format_type} from {location}")
        
        # Check string processing requirements
        elif any(pattern in operation.lower() for pattern in ['.split(', '.strip(', '.replace(', '.format(']):
            requirement = ExtractedRequirement(
                type=RequirementType.DATA_CHUNK_SPEC,
                source_location=location,
                pattern='string_processing',
                confidence=0.7,
                details={
                    'processing_type': 'string_manipulation',
                    'operation': operation,
                    'requirement_source': 'string_processing'
                }
            )
            self.extracted_requirements.append(requirement)
            logger.debug(f"Extracted string processing requirement from {location}")
        
        # Check code execution requirements
        elif any(pattern in operation.lower() for pattern in ['exec(', 'eval(', 'compile(', 'ast.parse']):
            requirement = ExtractedRequirement(
                type=RequirementType.PROGRAMMING_LANGUAGE,
                source_location=location,
                pattern='code_execution',
                confidence=0.9,
                details={
                    'execution_type': 'code_compilation',
                    'operation': operation,
                    'requirement_source': 'code_execution'
                }
            )
            self.extracted_requirements.append(requirement)
            logger.debug(f"Extracted code execution requirement from {location}")
    
    def _add_scenario_requirements(self):
        """
        NOTE: This method is intentionally left empty.
        
        Scenario-based requirements are NOT extracted from code - they are 
        human-defined requirements from application scenarios, managed separately
        by the ScenarioRequirementManager class.
        
        The 7 scenario requirements (intact textual elements, content relevance, etc.)
        are characterized manually based on user expectations and application scenarios,
        not extracted through static analysis.
        """
        logger.debug("Scenario requirements are handled separately by ScenarioRequirementManager")
        pass
    
    # Helper methods for pattern recognition
    def _initialize_template_patterns(self) -> List[str]:
        """Initialize template recognition patterns"""
        return [
            r'json\.loads\(',
            r'\.json\(\)',
            r'xml\.etree',
            r'BeautifulSoup',
            r'yaml\.load',
            r'csv\.reader',
            r'\.format\(',
            r'%\s*\(',
            r'f["\'].*{.*}.*["\']'
        ]
    
    def _initialize_syntax_patterns(self) -> List[str]:
        """Initialize syntax recognition patterns"""
        return [
            r'compile\(',
            r'ast\.parse',
            r'exec\(',
            r'eval\(',
            r'import\s+ast',
            r'tokenize\.',
            r'\.py$',
            r'subprocess\.run'
        ]
    
    def _initialize_segmentation_patterns(self) -> List[str]:
        """Initialize segmentation recognition patterns"""
        return [
            r'\.split\(',
            r'\.splitlines\(',
            r'textwrap\.',
            r'chunk',
            r'segment',
            r'\.join\(',
            r'len\(',
            r'[:]\s*\d+'
        ]
    
    def _is_llm_api_call(self, function_call: str) -> bool:
        """Check if a function call is an LLM API call"""
        # Based on common patterns in real App code, extend LLM API detection
        llm_indicators = [
            # OpenAI related
            'openai', 'gpt', 'chatgpt', 'ChatCompletion', 'Completion',
            'ChatOpenAI', 'AzureChatOpenAI', 'openai_call', 'llm_call',
            
            # Anthropic related  
            'claude', 'anthropic', 'ChatAnthropic', 'Anthropic',
            'anthropic_client', 'anthropic.messages',
            
            # Google related
            'gemini', 'genai', 'GenerativeModel', 'google.generativeai',
            
            # General LLM services
            'llm', 'language_model', 'chat_model', 'generate', 'complete',
            'BaseChatModel', 'LLMClient', 'LLMEndpoint', 'create_llm',
            
            # Other LLM services
            'huggingface', 'together', 'cohere', 'replicate', 'palm',
            'llama', 'mistral', 'qianfan', 'tongyi', 'baichuan',
            
            # API call patterns
            'chat', 'completion', 'embedding', 'model', 'api', 'client',
            'messages.create', 'chat.completions', 'completions.create',
            
            # Specific function names
            'api_request', 'webhook_request', 'send_message', 'ask_question',
            'get_response', 'generate_text', 'chat_completion'
        ]
        
        function_call_lower = function_call.lower()
        return any(indicator in function_call_lower for indicator in llm_indicators)
    
    def _is_core_functionality_branch(self, flow: Dict[str, Any]) -> bool:
        """
        Check if a data flow represents core functionality (not error handling).
        Focus on main execution paths.
        """
        context = flow.get('context', {})
        
        # Skip error handling branches
        if any(keyword in str(context).lower() for keyword in ['except', 'error', 'fail', 'catch']):
            return False
        
        # Skip logging and debugging branches
        if any(keyword in str(context).lower() for keyword in ['log', 'debug', 'print', 'warn']):
            return False
            
        return True
    
    def _extract_return_variable(self, func_name: str, callee: str) -> Optional[str]:
        """Extract variable name that receives LLM output"""
        # Simplified implementation - would need more sophisticated parsing
        if '=' in callee:
            parts = callee.split('=')
            if len(parts) >= 2:
                return parts[0].strip()
        return None
    
    def _build_data_flow_graph(self, llm_output_vars: Set[str]):
        """Build data flow graph tracking LLM output variables"""
        # Simplified implementation - would need more sophisticated analysis
        for var in llm_output_vars:
            # Add node for each LLM output variable
            self.data_flow_graph.add_node(var, type='llm_output')
            # Add usage edge
            self.data_flow_graph.add_edge(var, f"{var}_usage", operation='usage')
    
    # Pattern extraction helper methods
    def _extract_json_schema_hints(self, consumer: Dict[str, Any]) -> Dict[str, Any]:
        """Extract JSON schema hints from consumer code"""
        return {'type': 'object', 'properties': {}}
    
    def _extract_xml_schema_hints(self, consumer: Dict[str, Any]) -> Dict[str, Any]:
        """Extract XML schema hints from consumer code"""
        return {'root_element': 'unknown', 'namespace': None}
    
    def _has_positional_template_pattern(self, operation: str) -> bool:
        """Check if operation indicates positional template usage"""
        return any(pattern in operation for pattern in ['.format(', '%', 'f"', "f'"])
    
    def _extract_positional_identifiers(self, consumer: Dict[str, Any]) -> List[str]:
        """Extract positional template identifiers"""
        return []  # Simplified implementation
    
    def _extract_boundary_markers(self, consumer: Dict[str, Any]) -> List[str]:
        """Extract text boundary markers from segmentation code"""
        return ['.', '!', '?', '\n', '\n\n']
    
    def _extract_chunk_constraints(self, consumer: Dict[str, Any]) -> Dict[str, Any]:
        """Extract chunk size constraints"""
        return {'max_size': 1000, 'min_size': 10}
    
    def _extract_completeness_criteria(self, consumer: Dict[str, Any]) -> Dict[str, Any]:
        """Extract completeness criteria for text segments"""
        return {'word_completeness': True, 'sentence_integrity': True}
    
    def _extract_coherence_threshold(self, consumer: Dict[str, Any]) -> float:
        """Extract coherence threshold from context construction code"""
        return 0.7  # Default threshold
    
    def _extract_similarity_metric(self, consumer: Dict[str, Any]) -> str:
        """Extract similarity metric used in context construction"""
        return 'cosine'  # Default metric
    
    def _extract_context_window(self, consumer: Dict[str, Any]) -> int:
        """Extract context window size"""
        return 5  # Default window
    
    def _extract_syntax_level(self, consumer: Dict[str, Any]) -> str:
        """Extract required syntax level"""
        return 'statement'  # Default level
    
    def _extract_api_signatures(self, consumer: Dict[str, Any]) -> List[str]:
        """Extract API signatures from compilation code"""
        return []  # Simplified implementation

    def _infer_output_variable(self, func_name: str, llm_call: str) -> Optional[str]:
        """Infer output variable name from function name and LLM call"""
        common_var_names = [
            'response', 'result', 'output', 'answer', 'content', 'text',
            'completion', 'generation', 'reply', 'message', 'data'
        ]
        
        # Infer variable name based on LLM call type
        if 'completion' in llm_call.lower():
            return 'completion'
        elif 'chat' in llm_call.lower():
            return 'response'
        elif 'generate' in llm_call.lower():
            return 'generation'
        else:
            return 'response'  # Default return value


class CallGraphVisitor(ast.NodeVisitor):
    """
    AST visitor to build function call graph.
    Enhanced to better detect LLM API calls and track variable assignments.
    """
    
    def __init__(self, file_path: str, extractor: 'RequirementExtractor'):
        self.file_path = file_path
        self.call_graph = collections.defaultdict(list)
        self.current_function = None
        self.extractor = extractor
    
    def visit_FunctionDef(self, node):
        """Visit function definition"""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_Call(self, node):
        """Visit function calls to detect LLM API calls"""
        if self.current_function:
            call_name = self._get_call_name(node)
            if call_name:
                # Record function call
                self.call_graph[self.current_function].append(call_name)
                
                # Check if it's an LLM API call
                if self.extractor._is_llm_api_call(call_name):
                    logger.debug(f"Found LLM API call: {call_name} in {self.current_function}")
        
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        """Visit assignments to track LLM output variables"""
        if self.current_function and isinstance(node.value, ast.Call):
            call_name = self._get_call_name(node.value)
            if call_name and self.extractor._is_llm_api_call(call_name):
                # Extract variable names
                for target in node.targets:
                    var_names = self._extract_variable_names(target)
                    for var_name in var_names:
                        logger.debug(f"Found LLM output variable: {var_name} = {call_name}")
                        # Record LLM output variable
                        self.call_graph[f"{self.current_function}::{var_name}"] = [call_name]
        
        self.generic_visit(node)
    
    def _get_call_name(self, node) -> Optional[str]:
        """Extract function call name from AST node"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return f"{self._get_attr_name(node.func.value)}.{node.func.attr}"
        return None
    
    def _get_attr_name(self, node) -> str:
        """Extract attribute name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_attr_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_call_name(node) or "unknown"
        return "unknown"
    
    def _extract_variable_names(self, node) -> Set[str]:
        """Extract variable names from assignment target"""
        var_names = set()
        
        if isinstance(node, ast.Name):
            var_names.add(node.id)
        elif isinstance(node, ast.Tuple) or isinstance(node, ast.List):
            for elt in node.elts:
                var_names.update(self._extract_variable_names(elt))
        elif isinstance(node, ast.Starred):
            var_names.update(self._extract_variable_names(node.value))
        
        return var_names 