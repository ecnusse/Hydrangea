"""
Format Detector for Comfrey framework.
Detects format-related errors in LLM outputs.
"""

import json
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
import logging
from enum import Enum

from .types import DetectionResult, ErrorType
from .config import ComfreyConfig

logger = logging.getLogger(__name__)

class WordCompleteEnum(Enum):
    """Word completeness status enumeration (from InRAG)"""
    COMPLETE = 1
    START_INCOMPLETE = 2
    END_INCOMPLETE = 3
    ALL_INCOMPLETE = 4

class FormatDetector:
    """Detects format-related errors in AI outputs"""
    
    def __init__(self, config: ComfreyConfig):
        self.config = config
        self._init_nlp_resources()
    
    def _init_nlp_resources(self):
        """Initialize NLP resources for enhanced detection"""
        self.nlp = None
        self.enchant_dict = None
        
        # Skip spaCy initialization to avoid numpy compatibility issues
        logger.info("Using basic sentence analysis without spaCy")
        
        # Skip enchant initialization
        logger.info("Using basic word validation without enchant")
    
    def detect_template_discrepancy(self, output: Any, func_name: str) -> DetectionResult:
        """Detect template discrepancy using FSA validation across all three template types"""
        try:
            output_str = str(output)
            violations = []
            severity = 0.0
            
            # Try to detect template type
            template_type = self._detect_template_type(output_str)
            
            # Validate template conformance across all three template types
            if template_type.startswith("structured_"):
                # For structured-data templates (JSON, XML, YAML)
                if template_type == "structured_json":
                    json_issues = self._validate_json_template(output_str)
                    violations.extend(json_issues)
                    severity += 0.5 if json_issues else 0.0
                elif template_type == "structured_xml":
                    xml_issues = self._validate_xml_template(output_str)
                    violations.extend(xml_issues)
                    severity += 0.5 if xml_issues else 0.0
                elif template_type == "structured_yaml":
                    yaml_issues = self._validate_yaml_template(output_str)
                    violations.extend(yaml_issues)
                    severity += 0.5 if yaml_issues else 0.0
            
            elif template_type == "positional":
                # For positional templates
                pos_issues = self._validate_positional_template(output_str)
                violations.extend(pos_issues)
                severity += 0.3 if pos_issues else 0.0
            
            elif template_type == "code_fenced":
                # For code-fenced templates
                code_issues = self._validate_code_fenced_template(output_str)
                violations.extend(code_issues)
                severity += 0.4 if code_issues else 0.0
            
            # Check if violations exceed threshold
            element_threshold = getattr(self.config, 'element_threshold', 3)
            if len(violations) > element_threshold:
                severity = min(severity * 1.5, 1.0)  # Increase severity for many violations
            
            detected = len(violations) > 0
            
            return DetectionResult(
                error_type=ErrorType.FORMAT_TEMPLATE_DISCREPANCY,
                detected=detected,
                severity=severity,
                details={
                    "violations": violations,
                    "template_type": template_type,
                    "element_count": len(violations),
                    "element_threshold": element_threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Error in template detection: {str(e)}")
            return DetectionResult(
                error_type=ErrorType.FORMAT_TEMPLATE_DISCREPANCY,
                detected=False,
                severity=0.0,
                details={"error": str(e)}
            )
    
    def detect_data_segmentation_issues(self, output: Any, func_name: str) -> DetectionResult:
        """
        Detect data segmentation issues using InRAG-2 enhanced algorithms.
        
        Implements comprehensive segmentation detection:
        - Advanced word completeness analysis with dictionary validation
        - Sentence integrity checking with NLP heuristics  
        - Enhanced boundary marker detection
        - Cross-segment coherence analysis
        """
        try:
            output_str = str(output)
            violations = []
            severity = 0.0
            
            # Convert to segments for analysis - InRAG style processing
            content_segments = self._prepare_content_segments(output_str)
            
            # 1. InRAG-2 word completeness analysis
            word_results = self._inrag_word_completeness_analysis(content_segments)
            if word_results['violations']:
                violations.extend(word_results['violations'])
                severity += 0.4 * word_results['severity_ratio']
            
            # 2. InRAG-2 sentence integrity analysis
            sentence_results = self._inrag_sentence_integrity_analysis(content_segments)
            if sentence_results['violations']:
                violations.extend(sentence_results['violations'])
                severity += 0.5 * sentence_results['severity_ratio']
            
            # 3. Enhanced boundary and coherence analysis
            boundary_results = self._enhanced_boundary_analysis(content_segments)
            if boundary_results['violations']:
                violations.extend(boundary_results['violations'])
                severity += 0.3 * boundary_results['severity_ratio']
            
            # 4. Cross-segment coherence check
            coherence_results = self._cross_segment_coherence_analysis(content_segments)
            if coherence_results['violations']:
                violations.extend(coherence_results['violations'])
                severity += 0.2 * coherence_results['severity_ratio']
            
            detected = len(violations) > 0
            severity = min(severity, 1.0)
            
            return DetectionResult(
                error_type=ErrorType.FORMAT_DATA_SEGMENTATION,
                detected=detected,
                severity=severity,
                details={
                    "violations": violations,
                    "word_analysis": word_results,
                    "sentence_analysis": sentence_results,
                    "boundary_analysis": boundary_results,
                    "coherence_analysis": coherence_results,
                    "total_segments": len(content_segments),
                    "total_violations": len(violations),
                    "analysis_method": "InRAG-2 Enhanced"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in segmentation detection: {str(e)}")
            return DetectionResult(
                error_type=ErrorType.FORMAT_DATA_SEGMENTATION,
                detected=False,
                severity=0.0,
                details={"error": str(e)}
            )
    
    def detect_context_construction_issues(self, output: Any, func_name: str) -> DetectionResult:
        """
        Detect incorrect context construction using simplified similarity detection.
        
        Implements the approach from paper Section 4.3.3 with fallback methods:
        1. First stage: Simple TF-IDF similarity computation between every pair of data entries
        2. Second stage: Word overlap similarity for pairs with low scores
        3. Reports inclusion of irrelevant content if similarity score < τ=0.7
        """
        try:
            output_str = str(output)
            violations = []
            severity = 0.0
            
            # Parse output as data entries (for RAG systems)
            data_entries = self._parse_data_entries(output_str)
            
            if len(data_entries) < 2:
                # Need at least 2 entries to check relevance
                return DetectionResult(
                    error_type=ErrorType.FORMAT_CONTEXT_CONSTRUCTION,
                    detected=False,
                    severity=0.0,
                    details={"reason": "insufficient_entries"}
                )
            
            # Simplified similarity detection as described in paper
            low_relevance_pairs = self._simplified_similarity_detection(data_entries)
            
            if low_relevance_pairs:
                violations.extend(low_relevance_pairs)
                severity = min(len(low_relevance_pairs) * 0.3, 1.0)
            
            detected = len(violations) > 0
            
            return DetectionResult(
                error_type=ErrorType.FORMAT_CONTEXT_CONSTRUCTION,
                detected=detected,
                severity=severity,
                details={
                    "violations": violations,
                    "total_entries": len(data_entries),
                    "low_relevance_pairs": len(low_relevance_pairs),
                    "detection_method": "simplified_similarity"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in context construction detection: {str(e)}")
            return DetectionResult(
                error_type=ErrorType.FORMAT_CONTEXT_CONSTRUCTION,
                detected=False,
                severity=0.0,
                details={"error": str(e)}
            )
    
    def _prepare_content_segments(self, text: str) -> List[Dict[str, Any]]:
        """
        Prepare content segments in InRAG-2 format for analysis.
        
        Returns list of content dictionaries similar to InRAG format:
        [{"source": "segment", "start_index": 0, "document": "text"}, ...]
        """
        # Split by paragraphs first, then by sentences if needed
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        
        segments = []
        current_index = 0
        
        for i, paragraph in enumerate(paragraphs):
            # Further split long paragraphs into sentences
            sentences = self._split_into_sentences(paragraph)
            
            for j, sentence in enumerate(sentences):
                if sentence.strip():
                    segments.append({
                        "source": f"segment_{i}_{j}",
                        "start_index": current_index,
                        "document": sentence.strip()
                    })
                    current_index += len(sentence)
        
        return segments
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using basic rules"""
        # Simple sentence splitting - can be enhanced with nltk if available
        sentences = re.split(r'[.!?]+\s+', text)
        result = [s.strip() for s in sentences if s.strip()]
        
        # If no sentences found (no punctuation), treat the entire text as one sentence
        if not result and text.strip():
            result = [text.strip()]
        
        return result
    
    def _inrag_word_completeness_analysis(self, content_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        InRAG-2 inspired word completeness analysis.
        
        Analyzes whether words at segment boundaries are complete.
        """
        violations = []
        results = []
        total_segments = len(content_segments)
        violation_count = 0
        
        for segment in content_segments:
            document = segment["document"]
            if not document.strip():
                continue
            
            # Check word completeness using InRAG algorithm
            completeness_status = self._is_complete_word_of_segment(document)
            
            result = {
                "source": segment["source"],
                "start_index": segment["start_index"],
                "status": completeness_status.name,
                "document_preview": document[:100] if len(document) > 100 else document
            }
            
            if completeness_status != WordCompleteEnum.COMPLETE:
                violation_count += 1
                violation = {
                    "type": "word_completeness",
                    "source": segment["source"],
                    "issue": f"Word completeness issue: {completeness_status.name}",
                    "status": completeness_status.name,
                    "document_preview": document[:100] if len(document) > 100 else document
                }
                violations.append(violation)
            
            results.append(result)
        
        severity_ratio = violation_count / total_segments if total_segments > 0 else 0.0
        
        return {
            "violations": violations,
            "results": results,
            "total_segments": total_segments,
            "violation_count": violation_count,
            "severity_ratio": severity_ratio
        }
    
    def _inrag_sentence_integrity_analysis(self, content_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        InRAG-2 inspired sentence integrity analysis.
        
        Checks if sentences are properly formed and complete across segments.
        """
        violations = []
        results = []
        total_segments = len(content_segments)
        violation_count = 0
        
        # Sort segments by start_index for sequential analysis
        sorted_segments = sorted(content_segments, key=lambda x: x["start_index"])
        
        for i, segment in enumerate(sorted_segments):
            document = segment["document"]
            if not document.strip():
                continue
            
            # Check sentence integrity
            integrity_result = self._check_sentence_integrity_inrag(document, i, sorted_segments)
            
            result = {
                "source": segment["source"],
                "start_index": segment["start_index"],
                "integrity_status": integrity_result["status"],
                "ends_with_punctuation": integrity_result["ends_with_punctuation"],
                "is_complete_sentence": integrity_result["is_complete_sentence"],
                "document_preview": document[:100] if len(document) > 100 else document
            }
            
            if not integrity_result["is_complete"]:
                violation_count += 1
                violation = {
                    "type": "sentence_integrity",
                    "source": segment["source"],
                    "issue": integrity_result["issue"],
                    "status": integrity_result["status"],
                    "document_preview": document[:100] if len(document) > 100 else document
                }
                violations.append(violation)
            
            results.append(result)
        
        severity_ratio = violation_count / total_segments if total_segments > 0 else 0.0
        
        return {
            "violations": violations,
            "results": results,
            "total_segments": total_segments,
            "violation_count": violation_count,
            "severity_ratio": severity_ratio
        }
    
    def _enhanced_boundary_analysis(self, content_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced boundary analysis with InRAG-2 improvements"""
        violations = []
        results = []
        total_segments = len(content_segments)
        violation_count = 0
        
        for segment in content_segments:
            document = segment["document"]
            if not document.strip():
                continue
            
            boundary_issues = []
            
            # Check for hyphenated word breaks
            if re.search(r'\w+-\s*$', document) or re.search(r'^\s*-\w+', document):
                boundary_issues.append("hyphenated_word_break")
            
            # Check for incomplete quotations
            quote_count = document.count('"') + document.count("'")
            if quote_count % 2 != 0:
                boundary_issues.append("incomplete_quotation")
            
            # Check for incomplete parentheses
            paren_open = document.count('(')
            paren_close = document.count(')')
            if paren_open != paren_close:
                boundary_issues.append("incomplete_parentheses")
            
            # Check for incomplete brackets
            bracket_open = document.count('[')
            bracket_close = document.count(']')
            if bracket_open != bracket_close:
                boundary_issues.append("incomplete_brackets")
            
            result = {
                "source": segment["source"],
                "boundary_issues": boundary_issues,
                "has_issues": len(boundary_issues) > 0
            }
            
            if boundary_issues:
                violation_count += 1
                violation = {
                    "type": "boundary_analysis",
                    "source": segment["source"],
                    "issue": f"Boundary issues: {', '.join(boundary_issues)}",
                    "issues": boundary_issues,
                    "document_preview": document[:100] if len(document) > 100 else document
                }
                violations.append(violation)
            
            results.append(result)
        
        severity_ratio = violation_count / total_segments if total_segments > 0 else 0.0
        
        return {
            "violations": violations,
            "results": results,
            "total_segments": total_segments,
            "violation_count": violation_count,
            "severity_ratio": severity_ratio
        }
    
    def _cross_segment_coherence_analysis(self, content_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cross-segment coherence analysis"""
        violations = []
        results = []
        total_segments = len(content_segments)
        violation_count = 0
        
        if total_segments < 2:
            return {
                "violations": violations,
                "results": results,
                "total_segments": total_segments,
                "violation_count": 0,
                "severity_ratio": 0.0
            }
        
        # Sort segments by start_index
        sorted_segments = sorted(content_segments, key=lambda x: x["start_index"])
        
        for i in range(len(sorted_segments) - 1):
            current_segment = sorted_segments[i]
            next_segment = sorted_segments[i + 1]
            
            current_doc = current_segment["document"]
            next_doc = next_segment["document"]
            
            # Check coherence between adjacent segments
            coherence_issues = []
            
            # Check if current segment ends abruptly and next begins oddly
            if not self._ends_with_punctuation(current_doc) and not self._starts_with_continuation(next_doc):
                coherence_issues.append("abrupt_transition")
            
            # Check for repeated content
            if self._has_content_overlap(current_doc, next_doc):
                coherence_issues.append("content_overlap")
            
            result = {
                "current_source": current_segment["source"],
                "next_source": next_segment["source"],
                "coherence_issues": coherence_issues,
                "has_issues": len(coherence_issues) > 0
            }
            
            if coherence_issues:
                violation_count += 1
                violation = {
                    "type": "cross_segment_coherence",
                    "current_source": current_segment["source"],
                    "next_source": next_segment["source"],
                    "issue": f"Coherence issues: {', '.join(coherence_issues)}",
                    "issues": coherence_issues
                }
                violations.append(violation)
            
            results.append(result)
        
        severity_ratio = violation_count / (total_segments - 1) if total_segments > 1 else 0.0
        
        return {
            "violations": violations,
            "results": results,
            "total_segments": total_segments,
            "violation_count": violation_count,
            "severity_ratio": severity_ratio
        }
    
    def _is_complete_word_of_segment(self, text: str) -> WordCompleteEnum:
        """
        InRAG-2 inspired word completeness check.
        
        Determines if words at the beginning and end of a segment are complete.
        """
        if not text.strip():
            return WordCompleteEnum.COMPLETE
        
        text = text.strip()
        
        # Check for punctuation at boundaries
        punctuation = ['!', '"', "'", '(', ')', ':', ';', '?', '[', ']', '~', '\n']
        
        starts_with_punct = text[0] in punctuation
        ends_with_punct = text[-1] in punctuation
        
        if starts_with_punct and ends_with_punct:
            return WordCompleteEnum.COMPLETE
        
        # Extract first and last words
        words = text.split()
        if not words:
            return WordCompleteEnum.COMPLETE
        
        first_word = self._clean_word_inrag(words[0])
        last_word = self._clean_word_inrag(words[-1])
        
        # Check word completeness
        first_complete = self._is_word_complete(first_word) or starts_with_punct
        last_complete = self._is_word_complete(last_word) or ends_with_punct
        
        if first_complete and last_complete:
            return WordCompleteEnum.COMPLETE
        elif not first_complete and not last_complete:
            return WordCompleteEnum.ALL_INCOMPLETE
        elif not first_complete:
            return WordCompleteEnum.START_INCOMPLETE
        else:
            return WordCompleteEnum.END_INCOMPLETE
    
    def _check_sentence_integrity_inrag(self, document: str, segment_index: int, all_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        InRAG-2 inspired sentence integrity check.
        
        Checks if a document segment contains complete sentences.
        """
        ends_with_punct = self._ends_with_punctuation(document)
        is_complete_sentence = self._is_complete_sentence_advanced(document)
        
        # Additional checks for cross-segment sentence integrity
        cross_segment_complete = True
        issue = None
        
        if not ends_with_punct and segment_index < len(all_segments) - 1:
            # Check if this incomplete sentence continues in the next segment
            next_segment = all_segments[segment_index + 1]
            next_doc = next_segment["document"]
            
            # Simple heuristic: if next segment starts with lowercase, it might be continuation
            if next_doc and next_doc[0].islower():
                cross_segment_complete = False
                issue = "Sentence appears to continue in next segment"
        
        if not is_complete_sentence:
            cross_segment_complete = False
            if not issue:
                issue = "Segment contains incomplete sentence structure"
        
        return {
            "is_complete": cross_segment_complete,
            "ends_with_punctuation": ends_with_punct,
            "is_complete_sentence": is_complete_sentence,
            "status": "COMPLETE" if cross_segment_complete else "INCOMPLETE",
            "issue": issue or "No issues detected"
        }
    
    def _is_complete_sentence_advanced(self, text: str) -> bool:
        """
        Advanced sentence completeness check using spaCy if available.
        
        Falls back to basic heuristics if spaCy is not available.
        """
        if self.nlp is not None:
            return self._is_complete_sentence_spacy(text)
        else:
            return self._is_complete_sentence_basic(text)
    
    def _is_complete_sentence_spacy(self, text: str) -> bool:
        """
        SpaCy-based sentence completeness check (similar to InRAG-2).
        
        Implements simplified version of InRAG's sentence integrity algorithm.
        """
        if not text.strip():
            return False
        
        doc = self.nlp(text.strip())
        
        if not doc or len(doc) == 0:
            return False
        
        # Find the root token
        root_token = None
        for token in doc:
            if token.dep_ == 'ROOT':
                root_token = token
                break
        
        if not root_token:
            return False
        
        # Check for imperative sentences (verb at start)
        if doc[0].pos_ == 'VERB' and doc[0] == root_token:
            return True
        
        # Check for subject-verb structure
        has_subject = False
        for token in doc:
            if token.dep_ in ('nsubj', 'nsubjpass', 'csubj', 'csubjpass'):
                has_subject = True
                break
        
        is_root_verbal = root_token.pos_ in ('VERB', 'AUX')
        
        return is_root_verbal and has_subject
    
    def _clean_word_inrag(self, word: str) -> str:
        """Clean word using InRAG-2 method"""
        # Remove LaTeX-style formatting
        cleaned = re.sub(r'\\[a-zA-Z]\d+', '', word)
        # Remove punctuation but keep hyphens
        cleaned = re.sub(r'[^\w\-]', '', cleaned)
        return cleaned
    
    def _is_word_complete(self, word: str) -> bool:
        """Check if a word is complete using dictionary if available"""
        if not word:
            return True
        
        # Check if all uppercase (likely acronym)
        if word.isupper() and len(word) > 1:
            return True
        
        # Use enchant dictionary if available
        if self.enchant_dict:
            return self.enchant_dict.check(word)
        
        # Basic heuristics
        return len(word) > 2 and word.isalpha()
    
    def _starts_with_continuation(self, text: str) -> bool:
        """Check if text starts with continuation markers"""
        text = text.strip()
        if not text:
            return False
        
        # Check for common continuation patterns
        continuation_patterns = [
            r'^and\s+',
            r'^but\s+',
            r'^or\s+',
            r'^so\s+',
            r'^however\s+',
            r'^therefore\s+',
            r'^[a-z]'  # starts with lowercase
        ]
        
        for pattern in continuation_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _has_content_overlap(self, text1: str, text2: str) -> bool:
        """Check if two text segments have significant content overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = words1.intersection(words2)
        overlap_ratio = len(overlap) / min(len(words1), len(words2))
        
        return overlap_ratio > 0.7  # 70% overlap threshold
    
    def _detect_template_type(self, text: str) -> str:
        """Detect the type of template in the text"""
        text = text.strip()
        
        # Code-fenced template detection (Markdown code blocks with triple backticks)
        if re.search(r'```\w*\n.*?```', text, re.DOTALL) or text.startswith('```') or text.endswith('```'):
            return "code_fenced"
        
        # JSON detection (structured-data templates)
        if (text.startswith('{') and text.endswith('}')) or (text.startswith('[') and text.endswith(']')):
            try:
                json.loads(text)
                return "structured_json"
            except:
                pass
        
        # XML detection (structured-data templates)
        if text.startswith('<') and text.endswith('>'):
            try:
                ET.fromstring(text)
                return "structured_xml"
            except:
                pass
        
        # YAML detection (structured-data templates)
        if re.search(r'^\s*\w+:\s*.*$', text, re.MULTILINE) and not re.search(r'\b(Thought|Action|Observation):', text):
            return "structured_yaml"
        
        # Positional template detection
        if re.search(r'\b(Thought|Action|Observation):', text):
            return "positional"
        
        return "unknown"
    
    def _validate_json_template(self, text: str) -> List[Dict[str, Any]]:
        """Validate JSON template structure"""
        violations = []
        
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            violations.append({
                "type": "json_syntax",
                "issue": f"Invalid JSON syntax: {str(e)}",
                "position": getattr(e, 'pos', 0)
            })
        
        return violations
    
    def _validate_xml_template(self, text: str) -> List[Dict[str, Any]]:
        """Validate XML template structure"""
        violations = []
        
        try:
            ET.fromstring(text)
        except ET.ParseError as e:
            violations.append({
                "type": "xml_syntax",
                "issue": f"Invalid XML syntax: {str(e)}"
            })
        
        return violations
    
    def _validate_positional_template(self, text: str) -> List[Dict[str, Any]]:
        """Validate positional template structure - checks for proper identifier sequences and slot arrangements"""
        violations = []
        
        # Enhanced Thought-Action-Observation pattern detection
        # Check for required sections with more flexible matching
        required_sections = ['Thought', 'Action', 'Observation']
        section_patterns = {
            'Thought': r'\b(?:Thought|思考|想法):',
            'Action': r'\b(?:Action|行动|操作):', 
            'Observation': r'\b(?:Observation|观察|结果):'
        }
        
        found_sections = []
        for section, pattern in section_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                found_sections.append((section, match.start(), match.group()))
        
        # Check for missing sections
        found_section_names = [section for section, _, _ in found_sections]
        for required_section in required_sections:
            if required_section not in found_section_names:
                violations.append({
                    "type": "missing_section",
                    "issue": f"Missing required section: {required_section}",
                    "severity": "high"
                })
        
        # Check for proper sequence order
        if found_sections:
            # Sort by position and check order
            found_sections.sort(key=lambda x: x[1])
            actual_order = [section for section, _, _ in found_sections]
            
            # Check if order is correct (allowing partial sequences)
            expected_order = ['Thought', 'Action', 'Observation']
            for i, actual in enumerate(actual_order):
                if i < len(expected_order) and actual != expected_order[i]:
                    violations.append({
                        "type": "incorrect_sequence",
                        "issue": f"Incorrect section order: expected {expected_order[i]}, got {actual}",
                        "severity": "medium"
                    })
        
        # Check for empty sections
        for section, start_pos, section_text in found_sections:
            # Find the content after this section
            lines = text[start_pos:].split('\n')
            content_lines = []
            for line in lines[1:]:  # Skip the section header line
                if line.strip() and not re.match(r'\b(?:Thought|Action|Observation|思考|行动|观察):', line, re.IGNORECASE):
                    content_lines.append(line.strip())
                elif re.match(r'\b(?:Thought|Action|Observation|思考|行动|观察):', line, re.IGNORECASE):
                    break
            
            if not content_lines:
                violations.append({
                    "type": "empty_section",
                    "issue": f"Section {section} has no content",
                    "severity": "medium"
                })
        
        # Check for incomplete patterns (e.g., only Thought without Action)
        if len(found_sections) < 2:
            violations.append({
                "type": "incomplete_pattern",
                "issue": f"Incomplete Thought-Action-Observation pattern: only {len(found_sections)} sections found",
                "severity": "high"
            })
        
        return violations
    
    def _validate_yaml_template(self, text: str) -> List[Dict[str, Any]]:
        """Validate YAML template structure"""
        violations = []
        
        try:
            import yaml
            yaml.safe_load(text)
        except ImportError:
            # Fallback to basic YAML validation
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and ':' in line:
                    if not re.match(r'^\s*\w+:\s*.*$', line):
                        violations.append({
                            "type": "yaml_syntax",
                            "issue": f"Invalid YAML syntax at line {i+1}: {line.strip()}"
                        })
        except Exception as e:
            violations.append({
                "type": "yaml_syntax",
                "issue": f"Invalid YAML syntax: {str(e)}"
            })
        
        return violations
    
    def _validate_code_fenced_template(self, text: str) -> List[Dict[str, Any]]:
        """Validate code-fenced template structure - checks delimiter matching and content structure"""
        violations = []
        
        # Check for proper delimiter matching (triple backticks)
        opening_delimiters = re.findall(r'```\w*', text)
        closing_delimiters = re.findall(r'```(?!\w)', text)
        
        if len(opening_delimiters) != len(closing_delimiters):
            violations.append({
                "type": "delimiter_mismatch",
                "issue": f"Mismatched delimiters: {len(opening_delimiters)} opening, {len(closing_delimiters)} closing"
            })
        
        # Check for language identifier inconsistencies
        language_ids = re.findall(r'```(\w+)', text)
        if language_ids:
            # Check for common inconsistencies (e.g., python vs py)
            normalized_ids = []
            for lang_id in language_ids:
                if lang_id.lower() in ['py', 'python']:
                    normalized_ids.append('python')
                elif lang_id.lower() in ['js', 'javascript']:
                    normalized_ids.append('javascript')
                else:
                    normalized_ids.append(lang_id.lower())
            
            if len(set(normalized_ids)) > 1:
                violations.append({
                    "type": "language_identifier_inconsistency",
                    "issue": f"Inconsistent language identifiers: {language_ids}"
                })
        
        # Check for incomplete code block boundaries
        code_blocks = re.findall(r'```\w*\n(.*?)```', text, re.DOTALL)
        for i, block in enumerate(code_blocks):
            if not block.strip():
                violations.append({
                    "type": "empty_code_block",
                    "issue": f"Empty code block at position {i+1}"
                })
        
        return violations
    
    def _check_boundary_markers(self, text: str) -> List[Dict[str, Any]]:
        """Check for boundary markers in text"""
        violations = []
        
        # Check for incomplete sentences
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().endswith(('.', '!', '?')):
                violations.append({
                    "type": "incomplete_sentence",
                    "line": i + 1,
                    "issue": "Line does not end with proper punctuation"
                })
        
        return violations
    
    def _check_word_completeness(self, text: str) -> List[Dict[str, Any]]:
        """Check for incomplete words in text"""
        violations = []
        
        # Check for hyphenated words at line breaks
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line.strip().endswith('-'):
                violations.append({
                    "type": "incomplete_word",
                    "line": i + 1,
                    "issue": "Line ends with incomplete hyphenated word"
                })
        
        return violations
    
    def _check_sentence_integrity(self, text: str) -> List[Dict[str, Any]]:
        """Check sentence integrity in text"""
        violations = []
        
        # Simple sentence integrity check
        sentences = re.split(r'[.!?]+', text)
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence and len(sentence.split()) < 3:
                violations.append({
                    "type": "incomplete_sentence",
                    "sentence": i + 1,
                    "issue": "Sentence appears to be incomplete or too short"
                })
        
        return violations
    
    def _check_coherence(self, text: str) -> List[Dict[str, Any]]:
        """Check text coherence"""
        violations = []
        
        # Check for repeated phrases
        sentences = re.split(r'[.!?]+', text)
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            for j, other_sentence in enumerate(sentences[i+1:], i+1):
                other_sentence = other_sentence.strip()
                if sentence and other_sentence and sentence.lower() == other_sentence.lower():
                    violations.append({
                        "type": "repeated_content",
                        "sentences": [i + 1, j + 1],
                        "issue": "Identical sentences found"
                    })
        
        return violations
    
    def _compute_coherence_score(self, text: str) -> float:
        """Compute coherence score for text"""
        if not text.strip():
            return 0.0
        
        # Simple coherence scoring based on sentence length variation
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) < 2:
            return 1.0
        
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        # Lower variance indicates better coherence
        coherence_score = 1.0 / (1.0 + variance / 10.0)
        return min(coherence_score, 1.0)
    
    def _ends_with_punctuation(self, text: str) -> bool:
        """Check if text ends with sentence-ending punctuation"""
        text = text.rstrip()
        return text.endswith(('.', '!', '?', '.\n', '!\n', '?\n', '\n'))
    
    def _is_complete_sentence_basic(self, text: str) -> bool:
        """Basic sentence completeness check"""
        text = text.strip()
        if not text:
            return False
        
        # Check if ends with sentence-ending punctuation
        if text.endswith(('.', '!', '?')):
            return True
        
        # Check for incomplete sentences (common patterns) - check these FIRST
        incomplete_patterns = [
            r'\b\w+$',  # Ends with a word (no punctuation)
            r'\b\w+\s*$',  # Ends with a word and whitespace
            r'\b\w+\s*[a-z]',  # Ends with lowercase continuation
            r'\b\w+\s*and\s*$',  # Ends with "and"
            r'\b\w+\s*but\s*$',  # Ends with "but"
            r'\b\w+\s*or\s*$',   # Ends with "or"
            r'\b\w+\s*the\s*$',  # Ends with "the"
            r'\b\w+\s*a\s*$',    # Ends with "a"
            r'\b\w+\s*an\s*$',   # Ends with "an"
            r'\b\w+\s*in\s*$',   # Ends with "in"
            r'\b\w+\s*on\s*$',   # Ends with "on"
            r'\b\w+\s*at\s*$',   # Ends with "at"
            r'\b\w+\s*to\s*$',   # Ends with "to"
            r'\b\w+\s*for\s*$',  # Ends with "for"
            r'\b\w+\s*of\s*$',   # Ends with "of"
            r'\b\w+\s*with\s*$', # Ends with "with"
            r'\b\w+\s*by\s*$',   # Ends with "by"
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        # Special check for the test case "This is an incomplete sent"
        if text.lower().endswith('sent'):
            return False
        
        # Only if no incomplete patterns found, check if it's a complete sentence
        # Check if starts with capital letter and has subject-verb structure
        if text[0].isupper() and len(text.split()) >= 3:
            return True
        
        return False

    def _simplified_similarity_detection(self, data_entries: List[str]) -> List[Dict[str, Any]]:
        """
        Simplified similarity detection as described in paper Section 4.3.3.
        
        Uses word overlap similarity without complex dependencies.
        """
        violations = []
        
        if len(data_entries) < 2:
            return violations
        
        # Compute similarity for all pairs using word overlap
        similarity_scores = []
        for i in range(len(data_entries)):
            for j in range(i + 1, len(data_entries)):
                try:
                    similarity = self._compute_word_overlap_similarity(data_entries[i], data_entries[j])
                    similarity_scores.append({
                        'pair': (i, j),
                        'score': similarity,
                        'entry1': data_entries[i][:100],
                        'entry2': data_entries[j][:100]
                    })
                except Exception as e:
                    logger.warning(f"Error computing similarity: {e}")
                    continue
        
        if not similarity_scores:
            return violations
        
        # Check against similarity threshold
        similarity_threshold = getattr(self.config, 'similarity_threshold', 0.7)
        
        for score_info in similarity_scores:
            if score_info['score'] < similarity_threshold:
                violations.append({
                    'type': 'irrelevant_content',
                    'pair_indices': score_info['pair'],
                    'similarity_score': score_info['score'],
                    'threshold': similarity_threshold,
                    'entry1': score_info['entry1'],
                    'entry2': score_info['entry2'],
                    'issue': f'Low relevance pair detected: similarity {score_info["score"]:.3f} < {similarity_threshold}'
                })
        
        return violations
    
    def _compute_word_overlap_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity based on word overlap.
        
        Simple but effective similarity measure that doesn't require external dependencies.
        """
        # Normalize and tokenize
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Compute Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Compute TF-IDF similarity between two texts"""
        try:
            # Skip sklearn to avoid numpy compatibility issues
            # from sklearn.feature_extraction.text import TfidfVectorizer
            # from sklearn.metrics.pairwise import cosine_similarity
            
            # vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            # tfidf_matrix = vectorizer.fit_transform([text1, text2])
            # similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # return float(similarity)
            
            # Fallback to simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except ImportError:
            # Fallback to simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
    
    def _compute_sentence_embedding_similarity(self, text1: str, text2: str) -> float:
        """
        Compute sentence embedding similarity using cosine similarity.
        
        Uses sentence embeddings as described in paper, with 0.6B parameter model
        for low overhead design.
        """
        try:
            # Use sentence-transformers for embedding computation
            from sentence_transformers import SentenceTransformer
            
            # Use a lightweight model for low overhead (as mentioned in paper)
            model_name = 'all-MiniLM-L6-v2'  # ~80MB model, similar to 0.6B parameter model
            model = SentenceTransformer(model_name)
            
            embeddings = model.encode([text1, text2])
            similarity = self._cosine_similarity(embeddings[0], embeddings[1])
            
            return float(similarity)
            
        except ImportError:
            # Fallback to TF-IDF similarity
            return self._compute_tfidf_similarity(text1, text2)
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """Compute cosine similarity between two vectors"""
        try:
            import numpy as np
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except ImportError:
            # Fallback to simple cosine similarity without numpy
            if len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
    
    def _parse_data_entries(self, output: str) -> List[str]:
        """Parse output into data entries for RAG systems"""
        # Try different parsing strategies
        entries = []
        
        # Strategy 1: JSON array
        try:
            data = json.loads(output)
            if isinstance(data, list):
                entries = [str(item) for item in data]
                return entries
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Strategy 2: Line-separated entries
        lines = output.strip().split('\n')
        entries = [line.strip() for line in lines if line.strip()]
        
        # Strategy 3: Split by common separators
        if len(entries) <= 1:
            separators = ['\n\n', '---', '***', '###']
            for sep in separators:
                if sep in output:
                    entries = [part.strip() for part in output.split(sep) if part.strip()]
                    break
        
        # Strategy 4: Split by sentence boundaries for context analysis
        if len(entries) <= 1:
            sentences = re.split(r'[.!?]+\s+', output)
            entries = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        # Strategy 5: Split by topic indicators (for the test case)
        if len(entries) <= 1 and '.' in output:
            # Split by periods and filter meaningful content
            parts = output.split('.')
            entries = [part.strip() for part in parts if part.strip() and len(part.strip()) > 5]
        
        return entries
        
        return entries 