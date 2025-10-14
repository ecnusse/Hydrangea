"""
Syntax Detector for Comfrey framework.
Implements syntax error detection according to paper specifications.

Section 4.4: Resolving Syntax Errors
- Syntax-parser misalignment (Section 4.4.1)
- Inconsistent lexical features (Section 4.4.2)
"""

import ast
import dis
import re
import token
import tokenize
import io
import unicodedata
from typing import Dict, List, Any, Optional, Tuple
import logging

from .types import DetectionResult, ErrorType
from .config import ComfreyConfig

logger = logging.getLogger(__name__)

class SyntaxDetector:
    """
    Detects syntax-related errors in AI outputs using AST and bytecode analysis.
    
    Implements the detection algorithms described in Comfrey paper Section 4.4.
    """
    
    def __init__(self, config: ComfreyConfig):
        self.config = config
        self.session_language_patterns = {}  # Track language patterns across session
        self.session_spelling_standard = None  # Track spelling standard (US/UK)
        
    def detect_parser_misalignment(self, output: Any, func_name: str) -> DetectionResult:
        """
        Detect syntax-parser misalignment according to paper Section 4.4.1.
        
        Implements low overhead design by proactively reusing the syntax checking module
        of the compiler/parser to validate AI component outputs before they undergo actual processing.
        Focuses on syntax requirements in branch edges for function's core functionality,
        ignoring error-handling and fall-through edges.
        """
        try:
            output_str = str(output)
            violations = []
            severity = 0.0
            
            # Check if output looks like code
            if not self._looks_like_code(output_str):
                return DetectionResult(
                    error_type=ErrorType.SYNTAX_PARSER_MISALIGNMENT,
                    detected=False,
                    severity=0.0,
                    details={"reason": "Output does not appear to be code"}
                )
            
            # Use compiler/parser syntax validation as described in paper
            try:
                # Try to compile the code to check syntax
                compile(output_str, '<string>', 'exec')
                # If compilation succeeds, no syntax errors
                return DetectionResult(
                    error_type=ErrorType.SYNTAX_PARSER_MISALIGNMENT,
                    detected=False,
                    severity=0.0,
                    details={"reason": "Code compiles successfully"}
                )
            except SyntaxError as e:
                # Syntax error detected
                violations.append({
                    'type': 'syntax_error',
                    'message': str(e),
                    'line': e.lineno,
                    'offset': e.offset,
                    'text': e.text
                })
                severity = 0.8  # High severity for syntax errors
            
            # Additional AST-based validation for specific error types
            if self.config.enable_ast_analysis:
                ast_issues = self._detect_ast_violations(output_str)
                if ast_issues:
                    violations.extend(ast_issues)
                    severity = min(severity + 0.2, 1.0)
            
            detected = len(violations) > 0
            
            return DetectionResult(
                error_type=ErrorType.SYNTAX_PARSER_MISALIGNMENT,
                detected=detected,
                severity=severity,
                details={
                    "violations": violations,
                    "total_issues": len(violations),
                    "output_length": len(output_str),
                    "detection_method": "compiler_validation"
                },
                suggested_repair="ast_refinement" if detected else None
            )
            
        except Exception as e:
            logger.error(f"Error in syntax detection: {str(e)}")
            return DetectionResult(
                error_type=ErrorType.SYNTAX_PARSER_MISALIGNMENT,
                detected=False,
                severity=0.0,
                details={"error": str(e)}
            )
    
    def detect_lexical_inconsistency(self, output: Any, func_name: str) -> DetectionResult:
        """
        Detect inconsistent lexical features according to paper Section 4.4.2.
        
        Focuses on linguistic uniformity of three features throughout a session:
        1. Language usage (Unicode script detection + n-gram frequency analysis)
        2. Language standard (American/British English)
        3. Text structure (subheadings, lists, structural elements)
        """
        try:
            output_str = str(output)
            violations = []
            severity = 0.0
            
            # 1. Language usage detection using Unicode script detection
            if self.config.enable_unicode_detection:
                language_issues = self._detect_language_usage_inconsistency(output_str)
                if language_issues:
                    violations.extend(language_issues)
                    severity += 0.4
            
            # 2. Language standard detection using dictionary validation
            if self.config.enable_language_standard_check:
                spelling_issues = self._detect_spelling_standard_inconsistency(output_str)
                if spelling_issues:
                    violations.extend(spelling_issues)
                    severity += 0.3
            
            # 3. Text structure consistency detection
            structure_issues = self._detect_text_structure_inconsistency(output_str)
            if structure_issues:
                violations.extend(structure_issues)
                severity += 0.3
            
            detected = len(violations) > 0
            severity = min(severity, 1.0)
            
            return DetectionResult(
                error_type=ErrorType.SYNTAX_LEXICAL_INCONSISTENCY,
                detected=detected,
                severity=severity,
                details={
                    "violations": violations,
                    "total_issues": len(violations),
                    "session_spelling_standard": self.session_spelling_standard,
                    "detected_languages": self._get_detected_languages(output_str),
                    "detection_method": "unicode_ngram_analysis"
                },
                suggested_repair="lexical_standardization" if detected else None
            )
            
        except Exception as e:
            logger.error(f"Error in lexical detection: {str(e)}")
            return DetectionResult(
                error_type=ErrorType.SYNTAX_LEXICAL_INCONSISTENCY,
                detected=False,
                severity=0.0,
                details={"error": str(e)}
            )
    
    def _looks_like_code(self, text: str) -> bool:
        """Check if text appears to be code"""
        code_indicators = [
            r'\bdef\s+\w+\s*\(',  # Function definitions
            r'\bclass\s+\w+',     # Class definitions
            r'\bimport\s+\w+',    # Import statements
            r'\bfrom\s+\w+\s+import',  # From imports
            r'[\{\[\(].*[\}\]\)]',     # Brackets
            r'[=+\-*/]',              # Operators
            r';\s*$',                 # Semicolons
            r'^\s*#',                 # Comments
        ]
        
        for pattern in code_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False
    
    def _detect_ast_violations(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect AST violations for specific error types mentioned in paper.
        
        Focuses on errors common in LLM scenarios:
        - Bracket and string literal mismatches
        - Trailing commas
        - Invalid operators
        - Incomplete expressions
        """
        violations = []
        
        try:
            # Try to parse with AST
            tree = ast.parse(text)
            
            # AST visitor to detect specific violations
            class ASTVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.violations = []
                
                def visit_FunctionDef(self, node):
                    # Check for functions without return statements
                    has_return = False
                    for child in ast.walk(node):
                        if isinstance(child, ast.Return):
                            has_return = True
                            break
                    
                    if not has_return and len(node.body) > 0:
                        self.violations.append({
                            'type': 'missing_return',
                            'line': node.lineno,
                            'function': node.name
                        })
                
                def visit_Name(self, node):
                    # Check for undefined variable usage (basic check)
                    if node.id not in ['True', 'False', 'None', 'self']:
                        # This is a simplified check - in practice would need scope analysis
                        pass
            
            visitor = ASTVisitor()
            visitor.visit(tree)
            violations.extend(visitor.violations)
            
        except SyntaxError as e:
            # AST parsing failed - this is already caught by compiler validation
            pass
        
        return violations
    
    def _detect_language_usage_inconsistency(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect language usage inconsistency using Unicode script detection.
        
        Implements the approach from paper Section 4.4.2:
        - Uses Unicode script detection to identify character ranges (Latin, CJK, Arabic)
        - Uses n-gram frequency analysis against language models to detect language switches
        - Deactivates language usage examination in translations, linguistic discussion, and other typical multi-lingual scenarios
        """
        violations = []
        
        # Unicode script detection
        scripts = {}
        for char in text:
            if char.isalpha():
                try:
                    script = unicodedata.script(char)
                except AttributeError:
                    # Fallback for older Python versions
                    script = unicodedata.category(char)
                scripts[script] = scripts.get(script, 0) + 1
        
        # Check for mixed scripts (excluding common mixed scenarios)
        if len(scripts) > 1:
            # Check if this is a legitimate mixed scenario
            legitimate_mixed = self._is_legitimate_mixed_language(text)
            if not legitimate_mixed:
                violations.append({
                    'type': 'mixed_language_usage',
                    'scripts': list(scripts.keys()),
                    'script_counts': scripts,
                    'text_sample': text[:100]
                })
        
        # N-gram frequency analysis for language detection
        if self.config.enable_ngram_analysis:
            ngram_issues = self._detect_ngram_language_inconsistency(text)
            violations.extend(ngram_issues)
        
        return violations
    
    def _detect_spelling_standard_inconsistency(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect spelling standard inconsistency using dictionary validation.
        
        Implements the approach from paper Section 4.4.2:
        - Refers to a dictionary to examine whether phrases belong to the same standard variety (American/British English)
        - Establishes the standard using the lexical features of the first user text input
        """
        violations = []
        
        # Common US/UK spelling differences
        us_spellings = {
            'color', 'center', 'theater', 'traveled', 'counseling', 'program',
            'analyze', 'organize', 'realize', 'recognize', 'apologize'
        }
        
        uk_spellings = {
            'colour', 'centre', 'theatre', 'travelled', 'counselling', 'programme',
            'analyse', 'organise', 'realise', 'recognise', 'apologise'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        us_words = [word for word in words if word in us_spellings]
        uk_words = [word for word in words if word in uk_spellings]
        
        # Check for mixed spelling standards in the same text
        if us_words and uk_words:
            violations.append({
                'type': 'spelling_standard_inconsistency',
                'us_words': us_words,
                'uk_words': uk_words,
                'mixed_standards': True
            })
        else:
            # Determine current spelling standard
            if us_words:
                current_standard = 'US'
                inconsistent_words = uk_words
            elif uk_words:
                current_standard = 'UK'
                inconsistent_words = us_words
            else:
                current_standard = None
                inconsistent_words = []
            
            # Check against session standard
            if self.session_spelling_standard is None:
                self.session_spelling_standard = current_standard
            elif current_standard and current_standard != self.session_spelling_standard:
                violations.append({
                    'type': 'spelling_standard_inconsistency',
                    'session_standard': self.session_spelling_standard,
                    'current_standard': current_standard,
                    'inconsistent_words': inconsistent_words[:5]  # Limit to first 5
                })
        
        return violations
    
    def _detect_text_structure_inconsistency(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect text structure inconsistency.
        
        Implements the approach from paper Section 4.4.2:
        - Examines the existence of subheadings, lists, and other structural elements
        - Reports an error when any of the standard is violated
        """
        violations = []
        
        lines = text.split('\n')
        
        # Check for inconsistent heading patterns
        heading_patterns = []
        for line in lines:
            if line.strip().startswith('#'):
                heading_patterns.append(line.strip())
        
        if len(heading_patterns) > 1:
            # Check for inconsistent heading levels
            heading_levels = [len(line) - len(line.lstrip('#')) for line in heading_patterns]
            if len(set(heading_levels)) > 3:  # Too many different heading levels
                violations.append({
                    'type': 'inconsistent_heading_structure',
                    'heading_levels': heading_levels,
                    'heading_patterns': heading_patterns[:3]  # Limit to first 3
                })
        
        # Check for inconsistent list formatting
        list_patterns = []
        for line in lines:
            if re.match(r'^\s*[-*+]\s+', line):  # Bullet lists
                list_patterns.append('bullet')
            elif re.match(r'^\s*\d+\.\s+', line):  # Numbered lists
                list_patterns.append('numbered')
        
        if list_patterns:
            bullet_count = list_patterns.count('bullet')
            numbered_count = list_patterns.count('numbered')
            
            if bullet_count > 0 and numbered_count > 0:
                violations.append({
                    'type': 'mixed_list_formatting',
                    'bullet_lists': bullet_count,
                    'numbered_lists': numbered_count
                })
        
        return violations
    
    def _is_legitimate_mixed_language(self, text: str) -> bool:
        """
        Check if mixed language usage is legitimate.
        
        Recognizable patterns include:
        - Translation indicators
        - Code blocks
        - Structured presentation formats
        """
        # Translation indicators
        translation_patterns = [
            r'\b(translation|version|original|translated)\b',
            r'\b(中文|English|日本語|한국어)\b'
        ]
        
        for pattern in translation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Code blocks
        if re.search(r'```[\w]*\n.*\n```', text, re.DOTALL):
            return True
        
        # Structured formats
        if re.search(r':\s*[^\n]+\n.*:', text):  # Colon-separated format
            return True
        
        return False
    
    def _detect_ngram_language_inconsistency(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect language inconsistency using n-gram frequency analysis.
        
        Simplified implementation - in practice would use language models.
        """
        violations = []
        
        # Simple n-gram analysis for common language patterns
        words = text.lower().split()
        
        # Check for language-specific patterns
        english_patterns = ['the', 'and', 'or', 'but', 'in', 'on', 'at']
        chinese_patterns = ['的', '是', '在', '有', '和', '或']
        
        english_count = sum(1 for word in words if word in english_patterns)
        chinese_count = sum(1 for word in words if any(char in chinese_patterns for char in word))
        
        if english_count > 0 and chinese_count > 0:
            # Check if this is legitimate mixed usage
            if not self._is_legitimate_mixed_language(text):
                violations.append({
                    'type': 'ngram_language_inconsistency',
                    'english_patterns': english_count,
                    'chinese_patterns': chinese_count
                })
        
        return violations
    
    def _get_detected_languages(self, text: str) -> List[str]:
        """Get detected languages in the text"""
        languages = []
        
        # Unicode script detection
        scripts = set()
        for char in text:
            if char.isalpha():
                try:
                    script = unicodedata.script(char)
                except AttributeError:
                    # Fallback for older Python versions
                    script = unicodedata.category(char)
                scripts.add(script)
        
        # Map scripts to languages
        script_to_lang = {
            'Latn': 'English',
            'Hani': 'Chinese',
            'Hira': 'Japanese',
            'Kana': 'Japanese',
            'Hang': 'Korean',
            'Arab': 'Arabic',
            'Cyrl': 'Russian'
        }
        
        for script in scripts:
            if script in script_to_lang:
                languages.append(script_to_lang[script])
        
        return list(set(languages)) 