import ast
import logging
import collections
import networkx as nx
import beniget
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

@dataclass
class LLMOutputConsumer:
    """Represents a consumption point for LLM output"""
    function_name: str
    location: str
    operation: str
    context: Dict[str, Any] = field(default_factory=dict)

class DataFlowType(Enum):
    """Types of data flow operations"""
    ASSIGNMENT = "assignment"
    FUNCTION_CALL = "function_call"
    ATTRIBUTE_ACCESS = "attribute_access"
    SUBSCRIPT_ACCESS = "subscript_access"
    COMPARISON = "comparison"
    ITERATION = "iteration"
    CONDITIONAL = "conditional"
    RETURN = "return"

@dataclass
class DataFlowNode:
    """Represents a node in the data flow graph"""
    node_id: str
    node_type: DataFlowType
    location: str  # file:line:column
    operation: str
    input_vars: List[str] = field(default_factory=list)
    output_vars: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    is_core_functionality: bool = True

@dataclass
class DataFlowPath:
    """Represents a complete data flow path"""
    path_id: str
    nodes: List[DataFlowNode] = field(default_factory=list)
    start_variable: str = ""
    end_operations: List[str] = field(default_factory=list)
    confidence: float = 1.0

class DataFlowAnalyzer:
    
    def __init__(self, call_graph: nx.DiGraph):
        self.call_graph = call_graph
        # Use NetworkX to build data flow graph, providing more powerful graph analysis capabilities
        self.data_flow_graph = nx.DiGraph()
        # Use dictionary to store data flow nodes for each variable
        self.data_flow_nodes = collections.defaultdict(list)
        self.variable_dependencies = collections.defaultdict(set)
        self.llm_output_variables = set()
        self.consumption_points = []
        # Use Beniget for more precise def-use chain analysis
        self.duc_analyzer = beniget.DefUseChains()
        
    def analyze_data_flow(self, 
                         target_directory: str,
                         llm_api_patterns: List[str] = None) -> Dict[str, List[DataFlowPath]]:
        """
        Analyze data flow for LLM outputs through the codebase.
        Enhanced with better LLM API pattern detection.
        """
        logger.info("Starting data flow analysis")
        
        # Use improved LLM API patterns
        if llm_api_patterns is None:
            llm_api_patterns = self._get_enhanced_llm_patterns()
        
        # Step 1: Identify LLM output variables
        self._identify_llm_output_variables(target_directory, llm_api_patterns)
        
        # Step 2: Build data flow graph
        self._build_data_flow_graph(target_directory)
        
        # Step 3: Trace data flow paths
        data_flow_paths = self._trace_data_flow_paths()
        
        # Step 4: Identify consumption points
        self._identify_consumption_points(data_flow_paths)
        
        logger.info(f"Data flow analysis complete: {len(data_flow_paths)} paths found")
        return data_flow_paths
    
    def _get_enhanced_llm_patterns(self) -> List[str]:
        """Get enhanced LLM API patterns based on real App code analysis"""
        return [
            # OpenAI related
            'openai', 'gpt', 'chatgpt', 'ChatCompletion', 'Completion',
            'ChatOpenAI', 'AzureChatOpenAI', 'openai_call', 'llm_call',
            'openai.chat.completions', 'openai.completions',
            
            # Anthropic related  
            'claude', 'anthropic', 'ChatAnthropic', 'Anthropic',
            'anthropic_client', 'anthropic.messages', 'messages.create',
            
            # Google related
            'gemini', 'genai', 'GenerativeModel', 'google.generativeai',
            'chat_session', 'send_message',
            
            # General LLM services
            'llm', 'language_model', 'chat_model', 'generate', 'complete',
            'BaseChatModel', 'LLMClient', 'LLMEndpoint', 'create_llm',
            
            # Other LLM services
            'huggingface', 'together', 'cohere', 'replicate', 'palm',
            'llama', 'mistral', 'qianfan', 'tongyi', 'baichuan',
            'TogetherLLM', 'HuggingFaceLLM', 'QianfanChatEndpoint',
            
            # API call patterns
            'chat', 'completion', 'embedding', 'model', 'api', 'client',
            'messages.create', 'chat.completions', 'completions.create',
            
            # Specific function names
            'api_request', 'webhook_request', 'send_message', 'ask_question',
            'get_response', 'generate_text', 'chat_completion', 'ask',
            'get_llm', 'create_chat_model', 'llm_generate'
        ]
    
    def _identify_llm_output_variables(self, target_directory: str, llm_api_patterns: List[str]):

        logger.debug("Identifying LLM output variables")
        
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
                visitor = LLMOutputVariableVisitor(file_path, llm_api_patterns)
                visitor.visit(tree)
                
                # Collect LLM output variables
                self.llm_output_variables.update(visitor.llm_output_vars)
                
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
                continue
    
    def _build_data_flow_graph(self, target_directory: str):

        logger.debug("Building data flow graph")
        
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
                visitor = DataFlowGraphVisitor(file_path, self.llm_output_variables)
                visitor.visit(tree)
                
                # Merge data flow nodes
                for var, nodes in visitor.data_flow_nodes.items():
                    self.data_flow_nodes[var].extend(nodes)
                
                # Update variable dependencies
                for var, deps in visitor.variable_dependencies.items():
                    self.variable_dependencies[var].update(deps)
                    
            except Exception as e:
                logger.warning(f"Failed to build data flow for {file_path}: {e}")
                continue
    
    def _trace_data_flow_paths(self) -> Dict[str, List[DataFlowPath]]:
        """
        Trace complete data flow paths from LLM outputs to consumption points.
        Uses constraint-based path exploration.
        """
        logger.debug("Tracing data flow paths")
        
        data_flow_paths = {}
        
        for llm_var in self.llm_output_variables:
            paths = self._trace_paths_for_variable(llm_var)
            if paths:
                data_flow_paths[llm_var] = paths
        
        return data_flow_paths
    
    def _trace_paths_for_variable(self, start_var: str) -> List[DataFlowPath]:
        """
        Trace all possible data flow paths for a specific variable.
        Implements symbolic path exploration.
        """
        paths = []
        visited = set()
        
        def explore_path(current_var: str, current_path: List[DataFlowNode], path_id: str):
            if current_var in visited:
                return
            
            visited.add(current_var)
            
            # Get data flow nodes for current variable
            nodes = self.data_flow_nodes.get(current_var, [])
            
            for node in nodes:
                new_path = current_path + [node]
                
                # Check if this is a consumption point
                if self._is_consumption_point(node):
                    path = DataFlowPath(
                        path_id=f"{path_id}_{len(paths)}",
                        nodes=new_path,
                        start_variable=start_var,
                        end_operations=[node.operation],
                        confidence=self._calculate_path_confidence(new_path)
                    )
                    paths.append(path)
                
                # Continue exploring through output variables
                for output_var in node.output_vars:
                    if output_var != current_var:  # Avoid cycles
                        explore_path(output_var, new_path, path_id)
        
        explore_path(start_var, [], f"path_{start_var}")
        return paths
    
    def _identify_consumption_points(self, data_flow_paths: Dict[str, List[DataFlowPath]]):
        """
        Identify points where LLM outputs are consumed by downstream operations.
        Focus on core functionality, ignore error handling.
        """
        logger.debug("Identifying consumption points")
        
        for var, paths in data_flow_paths.items():
            for path in paths:
                for node in path.nodes:
                    if self._is_consumption_point(node) and node.is_core_functionality:
                        consumption_info = {
                            'variable': var,
                            'location': node.location,
                            'operation': node.operation,
                            'operation_type': node.node_type.value,
                            'context': node.context,
                            'confidence': path.confidence
                        }
                        self.consumption_points.append(consumption_info)
    
    def _is_consumption_point(self, node: DataFlowNode) -> bool:

        consumption_patterns = [
            # Template processing
            'json.loads', '.json()', 'yaml.load', 'xml.etree', 'BeautifulSoup',
            # String processing
            '.split(', '.strip(', '.replace(', '.format(', '.join(',
            # Parsing and compilation
            'ast.parse', 'compile(', 'exec(', 'eval(',
            # File operations
            'open(', '.write(', '.read(',
            # Comparison operations
            '==', '!=', 'in', 'not in',
            # Iteration operations
            'for', 'while', 'enumerate',
            # Function calls with LLM output as parameter
            'print(', 'len(', 'str(', 'int(', 'float('
        ]
        
        operation = node.operation.lower()
        return any(pattern in operation for pattern in consumption_patterns)
    
    def _calculate_path_confidence(self, path: List[DataFlowNode]) -> float:
        """
        Calculate confidence score for a data flow path.
        Based on path complexity and operation types.
        """
        if not path:
            return 0.0
        
        base_confidence = 1.0
        
        # Reduce confidence for complex paths
        complexity_penalty = min(0.1 * len(path), 0.5)
        
        # Reduce confidence for non-core functionality
        non_core_penalty = sum(0.1 for node in path if not node.is_core_functionality)
        
        # Reduce confidence for uncertain operations
        uncertain_penalty = sum(0.05 for node in path if 'unknown' in node.operation)
        
        final_confidence = base_confidence - complexity_penalty - non_core_penalty - uncertain_penalty
        return max(final_confidence, 0.1)  # Minimum confidence
    
    def get_consumption_points(self) -> List[Dict[str, Any]]:
        """Get all identified consumption points"""
        return self.consumption_points
    
    def get_variable_dependencies(self) -> Dict[str, Set[str]]:
        """Get variable dependency graph"""
        return dict(self.variable_dependencies)


class LLMOutputVariableVisitor(ast.NodeVisitor):

    
    def __init__(self, file_path: str, llm_api_patterns: List[str]):
        self.file_path = file_path
        self.llm_api_patterns = llm_api_patterns
        self.llm_output_vars = set()
        self.current_function = None
    
    def visit_FunctionDef(self, node):
        """Visit function definition"""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_Assign(self, node):
        """Visit assignment statements to identify LLM output variables"""
        if isinstance(node.value, ast.Call):
            call_name = self._get_call_name(node.value)
            if call_name and self._is_llm_api_call(call_name):
                # Extract variable names from assignment targets
                for target in node.targets:
                    var_names = self._extract_variable_names(target)
                    self.llm_output_vars.update(var_names)
        
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
        return "unknown"
    
    def _is_llm_api_call(self, call_name: str) -> bool:
        """Check if a function call is an LLM API call"""
        return any(pattern in call_name.lower() for pattern in self.llm_api_patterns)
    
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


class DataFlowGraphVisitor(ast.NodeVisitor):

    def __init__(self, file_path: str, llm_output_variables: Set[str]):
        self.file_path = file_path
        self.llm_output_variables = llm_output_variables
        # Track all variables that are derived from LLM outputs
        self.llm_derived_variables = set(llm_output_variables)
        self.data_flow_nodes = collections.defaultdict(list)
        self.variable_dependencies = collections.defaultdict(set)
        self.current_function = None
        self.current_line = 0
    
    def visit_FunctionDef(self, node):
        """Visit function definition"""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_Assign(self, node):
        """Visit assignment statements"""
        self.current_line = node.lineno
        
        # Extract input variables from the value
        input_vars = self._extract_used_variables(node.value)
        
        # Extract output variables from targets
        output_vars = set()
        for target in node.targets:
            output_vars.update(self._extract_variable_names(target))
        
        # Check if this assignment involves LLM output variables or derived variables
        if any(var in self.llm_derived_variables for var in input_vars.union(output_vars)):
            data_flow_node = DataFlowNode(
                node_id=f"{self.file_path}:{node.lineno}:{node.col_offset}",
                node_type=DataFlowType.ASSIGNMENT,
                location=f"{self.file_path}:{node.lineno}:{node.col_offset}",
                operation=ast.unparse(node) if hasattr(ast, 'unparse') else str(node),
                input_vars=list(input_vars),
                output_vars=list(output_vars),
                context={'function': self.current_function},
                is_core_functionality=self._is_core_functionality(node)
            )
            
            # Add to data flow graph for all involved variables
            for var in input_vars.union(output_vars):
                self.data_flow_nodes[var].append(data_flow_node)
            
            # Update variable dependencies
            for output_var in output_vars:
                self.variable_dependencies[output_var].update(input_vars)
                
            # If any input variable is LLM-derived, mark output variables as LLM-derived too
            if any(var in self.llm_derived_variables for var in input_vars):
                self.llm_derived_variables.update(output_vars)
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Visit function calls"""
        self.current_line = node.lineno
        
        # Extract input variables from arguments
        input_vars = set()
        for arg in node.args:
            input_vars.update(self._extract_used_variables(arg))
        for keyword in node.keywords:
            input_vars.update(self._extract_used_variables(keyword.value))
        
        # Check if this call involves LLM output variables or derived variables
        if any(var in self.llm_derived_variables for var in input_vars):
            call_name = self._get_call_name(node)
            
            data_flow_node = DataFlowNode(
                node_id=f"{self.file_path}:{node.lineno}:{node.col_offset}",
                node_type=DataFlowType.FUNCTION_CALL,
                location=f"{self.file_path}:{node.lineno}:{node.col_offset}",
                operation=call_name or "unknown_call",
                input_vars=list(input_vars),
                output_vars=[],  # Function calls don't directly assign to variables
                context={'function': self.current_function, 'call_name': call_name},
                is_core_functionality=self._is_core_functionality(node)
            )
            
            # Add to data flow graph
            for var in input_vars:
                self.data_flow_nodes[var].append(data_flow_node)
        
        self.generic_visit(node)
    
    def visit_Compare(self, node):
        """Visit comparison operations"""
        self.current_line = node.lineno
        
        # Extract variables from comparison
        input_vars = self._extract_used_variables(node.left)
        for comparator in node.comparators:
            input_vars.update(self._extract_used_variables(comparator))
        
        # Check if this comparison involves LLM output variables or derived variables
        if any(var in self.llm_derived_variables for var in input_vars):
            data_flow_node = DataFlowNode(
                node_id=f"{self.file_path}:{node.lineno}:{node.col_offset}",
                node_type=DataFlowType.COMPARISON,
                location=f"{self.file_path}:{node.lineno}:{node.col_offset}",
                operation=ast.unparse(node) if hasattr(ast, 'unparse') else str(node),
                input_vars=list(input_vars),
                output_vars=[],
                context={'function': self.current_function},
                is_core_functionality=self._is_core_functionality(node)
            )
            
            # Add to data flow graph
            for var in input_vars:
                self.data_flow_nodes[var].append(data_flow_node)
        
        self.generic_visit(node)
    
    def _extract_used_variables(self, node) -> Set[str]:
        """Extract all variables used in an AST node"""
        var_names = set()
        
        if isinstance(node, ast.Name):
            var_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            var_names.update(self._extract_used_variables(node.value))
        elif isinstance(node, ast.Subscript):
            var_names.update(self._extract_used_variables(node.value))
            var_names.update(self._extract_used_variables(node.slice))
        elif isinstance(node, ast.Call):
            var_names.update(self._extract_used_variables(node.func))
            for arg in node.args:
                var_names.update(self._extract_used_variables(arg))
            for keyword in node.keywords:
                var_names.update(self._extract_used_variables(keyword.value))
        elif isinstance(node, ast.BinOp):
            var_names.update(self._extract_used_variables(node.left))
            var_names.update(self._extract_used_variables(node.right))
        elif isinstance(node, ast.UnaryOp):
            var_names.update(self._extract_used_variables(node.operand))
        elif isinstance(node, ast.Compare):
            var_names.update(self._extract_used_variables(node.left))
            for comparator in node.comparators:
                var_names.update(self._extract_used_variables(comparator))
        elif isinstance(node, (ast.List, ast.Tuple)):
            for elt in node.elts:
                var_names.update(self._extract_used_variables(elt))
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if key:  # key can be None for **kwargs
                    var_names.update(self._extract_used_variables(key))
            for value in node.values:
                var_names.update(self._extract_used_variables(value))
        
        return var_names
    
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
        return "unknown"
    
    def _is_core_functionality(self, node) -> bool:
        """
        Check if an AST node represents core functionality.
        Exclude error handling, logging, and debugging code.
        """
        # Get the source code context if possible
        try:
            if hasattr(ast, 'unparse'):
                source = ast.unparse(node).lower()
            else:
                source = str(node).lower()
            
            # Skip error handling
            if any(keyword in source for keyword in ['except', 'error', 'fail', 'catch', 'raise']):
                return False
            
            # Skip logging and debugging
            if any(keyword in source for keyword in ['log', 'debug', 'print', 'warn', 'info']):
                return False
            
            # Skip test-related code
            if any(keyword in source for keyword in ['test', 'assert', 'mock']):
                return False
                
            return True
        except:
            return True  # Default to core functionality if we can't determine 