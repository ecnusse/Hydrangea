"""
Lightweight FormatDetector used by Hydrangea.
"""

import json
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Any
import logging
from enum import Enum

from .types import DetectionResult, ErrorType
from .config import Config


logger = logging.getLogger(__name__)


class WordCompleteEnum(Enum):
    COMPLETE = 1
    START_INCOMPLETE = 2
    END_INCOMPLETE = 3
    ALL_INCOMPLETE = 4


class FormatDetector:
    def __init__(self, config: Config):
        self.config = config
        self.nlp = None
        self.enchant_dict = None

    def detect_template_discrepancy(self, output: Any, func_name: str) -> DetectionResult:
        try:
            output_str = str(output)
            violations: List[Dict[str, Any]] = []
            severity = 0.0

            template_type = self._detect_template_type(output_str)

            if template_type.startswith("structured_"):
                if template_type == "structured_json":
                    issues = self._validate_json_template(output_str)
                elif template_type == "structured_xml":
                    issues = self._validate_xml_template(output_str)
                else:
                    issues = self._validate_yaml_template(output_str)
                violations.extend(issues)
                severity += 0.5 if issues else 0.0
            elif template_type == "positional":
                issues = self._validate_positional_template(output_str)
                violations.extend(issues)
                severity += 0.3 if issues else 0.0
            elif template_type == "code_fenced":
                issues = self._validate_code_fenced_template(output_str)
                violations.extend(issues)
                severity += 0.4 if issues else 0.0

            element_threshold = getattr(self.config, 'element_threshold', 3)
            if len(violations) > element_threshold:
                severity = min(severity * 1.5, 1.0)

            return DetectionResult(
                error_type=ErrorType.FORMAT_TEMPLATE_DISCREPANCY,
                detected=len(violations) > 0,
                severity=severity,
                details={
                    "violations": violations,
                    "template_type": template_type,
                    "element_count": len(violations),
                    "element_threshold": element_threshold,
                },
            )
        except Exception as e:
            logger.error(f"Error in template detection: {e}")
            return DetectionResult(
                error_type=ErrorType.FORMAT_TEMPLATE_DISCREPANCY,
                detected=False,
                severity=0.0,
                details={"error": str(e)},
            )

    def detect_data_segmentation_issues(self, output: Any, func_name: str) -> DetectionResult:
        try:
            output_str = str(output)
            content_segments = self._prepare_content_segments(output_str)

            word_results = self._inrag_word_completeness_analysis(content_segments)
            sentence_results = self._inrag_sentence_integrity_analysis(content_segments)
            boundary_results = self._enhanced_boundary_analysis(content_segments)
            coherence_results = self._cross_segment_coherence_analysis(content_segments)

            violations: List[Dict[str, Any]] = []
            severity = 0.0

            if word_results['violations']:
                violations.extend(word_results['violations'])
                severity += 0.4 * word_results['severity_ratio']
            if sentence_results['violations']:
                violations.extend(sentence_results['violations'])
                severity += 0.5 * sentence_results['severity_ratio']
            if boundary_results['violations']:
                violations.extend(boundary_results['violations'])
                severity += 0.3 * boundary_results['severity_ratio']
            if coherence_results['violations']:
                violations.extend(coherence_results['violations'])
                severity += 0.2 * coherence_results['severity_ratio']

            return DetectionResult(
                error_type=ErrorType.FORMAT_DATA_SEGMENTATION,
                detected=len(violations) > 0,
                severity=min(severity, 1.0),
                details={
                    "violations": violations,
                    "word_analysis": word_results,
                    "sentence_analysis": sentence_results,
                    "boundary_analysis": boundary_results,
                    "coherence_analysis": coherence_results,
                    "total_segments": len(content_segments),
                    "total_violations": len(violations),
                    "analysis_method": "InRAG-2 Enhanced",
                },
            )
        except Exception as e:
            logger.error(f"Error in segmentation detection: {e}")
            return DetectionResult(
                error_type=ErrorType.FORMAT_DATA_SEGMENTATION,
                detected=False,
                severity=0.0,
                details={"error": str(e)},
            )

    def detect_context_construction_issues(self, output: Any, func_name: str) -> DetectionResult:
        try:
            output_str = str(output)
            entries = self._parse_data_entries(output_str)
            if len(entries) < 2:
                return DetectionResult(
                    error_type=ErrorType.FORMAT_CONTEXT_CONSTRUCTION,
                    detected=False,
                    severity=0.0,
                    details={"reason": "insufficient_entries"},
                )
            low_rel = self._simplified_similarity_detection(entries)
            detected = len(low_rel) > 0
            return DetectionResult(
                error_type=ErrorType.FORMAT_CONTEXT_CONSTRUCTION,
                detected=detected,
                severity=min(len(low_rel) * 0.3, 1.0) if detected else 0.0,
                details={
                    "violations": low_rel,
                    "total_entries": len(entries),
                    "low_relevance_pairs": len(low_rel),
                    "detection_method": "simplified_similarity",
                },
            )
        except Exception as e:
            logger.error(f"Error in context construction detection: {e}")
            return DetectionResult(
                error_type=ErrorType.FORMAT_CONTEXT_CONSTRUCTION,
                detected=False,
                severity=0.0,
                details={"error": str(e)},
            )

    # --- Helpers below are copied with minimal changes ---

    def _prepare_content_segments(self, text: str) -> List[Dict[str, Any]]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()] or [text]
        segments: List[Dict[str, Any]] = []
        current_index = 0
        for i, paragraph in enumerate(paragraphs):
            sentences = self._split_into_sentences(paragraph)
            for j, sentence in enumerate(sentences):
                s = sentence.strip()
                if not s:
                    continue
                segments.append({
                    "source": f"segment_{i}_{j}",
                    "start_index": current_index,
                    "document": s,
                })
                current_index += len(sentence)
        return segments

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+\s+', text)
        result = [s.strip() for s in sentences if s.strip()]
        return result or ([text.strip()] if text.strip() else [])

    def _inrag_word_completeness_analysis(self, content_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        total_segments = len(content_segments)
        violation_count = 0
        for segment in content_segments:
            document = segment["document"]
            if not document.strip():
                continue
            completeness_status = self._is_complete_word_of_segment(document)
            results.append({
                "source": segment["source"],
                "start_index": segment["start_index"],
                "status": completeness_status.name,
                "document_preview": document[:100] if len(document) > 100 else document,
            })
            if completeness_status != WordCompleteEnum.COMPLETE:
                violation_count += 1
                violations.append({
                    "type": "word_completeness",
                    "source": segment["source"],
                    "issue": f"Word completeness issue: {completeness_status.name}",
                    "status": completeness_status.name,
                    "document_preview": document[:100] if len(document) > 100 else document,
                })
        severity_ratio = violation_count / total_segments if total_segments > 0 else 0.0
        return {"violations": violations, "results": results, "total_segments": total_segments, "violation_count": violation_count, "severity_ratio": severity_ratio}

    def _inrag_sentence_integrity_analysis(self, content_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        total_segments = len(content_segments)
        violation_count = 0
        sorted_segments = sorted(content_segments, key=lambda x: x["start_index"])
        for i, segment in enumerate(sorted_segments):
            document = segment["document"]
            if not document.strip():
                continue
            integrity_result = self._check_sentence_integrity_inrag(document, i, sorted_segments)
            results.append({
                "source": segment["source"],
                "start_index": segment["start_index"],
                "integrity_status": integrity_result["status"],
                "ends_with_punctuation": integrity_result["ends_with_punctuation"],
                "is_complete_sentence": integrity_result["is_complete_sentence"],
                "document_preview": document[:100] if len(document) > 100 else document,
            })
            if not integrity_result["is_complete"]:
                violation_count += 1
                violations.append({
                    "type": "sentence_integrity",
                    "source": segment["source"],
                    "issue": integrity_result["issue"],
                    "status": integrity_result["status"],
                    "document_preview": document[:100] if len(document) > 100 else document,
                })
        severity_ratio = violation_count / total_segments if total_segments > 0 else 0.0
        return {"violations": violations, "results": results, "total_segments": total_segments, "violation_count": violation_count, "severity_ratio": severity_ratio}

    def _enhanced_boundary_analysis(self, content_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        total_segments = len(content_segments)
        violation_count = 0
        for segment in content_segments:
            document = segment["document"]
            if not document.strip():
                continue
            boundary_issues = []
            if re.search(r'\w+-\s*$', document) or re.search(r'^\s*-\w+', document):
                boundary_issues.append("hyphenated_word_break")
            quote_count = document.count('"') + document.count("'")
            if quote_count % 2 != 0:
                boundary_issues.append("incomplete_quotation")
            if document.count('(') != document.count(')'):
                boundary_issues.append("incomplete_parentheses")
            if document.count('[') != document.count(']'):
                boundary_issues.append("incomplete_brackets")
            results.append({"source": segment["source"], "boundary_issues": boundary_issues, "has_issues": len(boundary_issues) > 0})
            if boundary_issues:
                violation_count += 1
                violations.append({
                    "type": "boundary_analysis",
                    "source": segment["source"],
                    "issue": f"Boundary issues: {', '.join(boundary_issues)}",
                    "issues": boundary_issues,
                    "document_preview": document[:100] if len(document) > 100 else document,
                })
        severity_ratio = violation_count / total_segments if total_segments > 0 else 0.0
        return {"violations": violations, "results": results, "total_segments": total_segments, "violation_count": violation_count, "severity_ratio": severity_ratio}

    def _cross_segment_coherence_analysis(self, content_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        total_segments = len(content_segments)
        violation_count = 0
        if total_segments < 2:
            return {"violations": violations, "results": results, "total_segments": total_segments, "violation_count": 0, "severity_ratio": 0.0}
        sorted_segments = sorted(content_segments, key=lambda x: x["start_index"])
        for i in range(len(sorted_segments) - 1):
            current_doc = sorted_segments[i]["document"]
            next_doc = sorted_segments[i + 1]["document"]
            coherence_issues = []
            if not self._ends_with_punctuation(current_doc) and not self._starts_with_continuation(next_doc):
                coherence_issues.append("abrupt_transition")
            if self._has_content_overlap(current_doc, next_doc):
                coherence_issues.append("content_overlap")
            results.append({
                "current_source": sorted_segments[i]["source"],
                "next_source": sorted_segments[i + 1]["source"],
                "coherence_issues": coherence_issues,
                "has_issues": len(coherence_issues) > 0,
            })
            if coherence_issues:
                violation_count += 1
                violations.append({
                    "type": "cross_segment_coherence",
                    "current_source": sorted_segments[i]["source"],
                    "next_source": sorted_segments[i + 1]["source"],
                    "issue": f"Coherence issues: {', '.join(coherence_issues)}",
                    "issues": coherence_issues,
                })
        severity_ratio = violation_count / (total_segments - 1) if total_segments > 1 else 0.0
        return {"violations": violations, "results": results, "total_segments": total_segments, "violation_count": violation_count, "severity_ratio": severity_ratio}

    def _is_complete_word_of_segment(self, text: str) -> WordCompleteEnum:
        text = text.strip()
        if not text:
            return WordCompleteEnum.COMPLETE
        punctuation = ['!', '"', "'", '(', ')', ':', ';', '?', '[', ']', '~', '\n']
        starts_with_punct = text[0] in punctuation
        ends_with_punct = text[-1] in punctuation
        if starts_with_punct and ends_with_punct:
            return WordCompleteEnum.COMPLETE
        words = text.split()
        if not words:
            return WordCompleteEnum.COMPLETE
        first_word = self._clean_word_inrag(words[0])
        last_word = self._clean_word_inrag(words[-1])
        first_complete = self._is_word_complete(first_word) or starts_with_punct
        last_complete = self._is_word_complete(last_word) or ends_with_punct
        if first_complete and last_complete:
            return WordCompleteEnum.COMPLETE
        if not first_complete and not last_complete:
            return WordCompleteEnum.ALL_INCOMPLETE
        if not first_complete:
            return WordCompleteEnum.START_INCOMPLETE
        return WordCompleteEnum.END_INCOMPLETE

    def _check_sentence_integrity_inrag(self, document: str, segment_index: int, all_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        ends_with_punct = self._ends_with_punctuation(document)
        is_complete_sentence = self._is_complete_sentence_basic(document)
        cross_segment_complete = True
        issue = None
        if not ends_with_punct and segment_index < len(all_segments) - 1:
            next_doc = all_segments[segment_index + 1]["document"]
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
            "issue": issue or "No issues detected",
        }

    def _clean_word_inrag(self, word: str) -> str:
        cleaned = re.sub(r'\\[a-zA-Z]\d+', '', word)
        cleaned = re.sub(r'[^\w\-]', '', cleaned)
        return cleaned

    def _is_word_complete(self, word: str) -> bool:
        if not word:
            return True
        if word.isupper() and len(word) > 1:
            return True
        return len(word) > 2 and word.isalpha()

    def _starts_with_continuation(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        for pattern in [r'^and\s+', r'^but\s+', r'^or\s+', r'^so\s+', r'^however\s+', r'^therefore\s+', r'^[a-z]']:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        return False

    def _has_content_overlap(self, text1: str, text2: str) -> bool:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return False
        overlap = words1.intersection(words2)
        overlap_ratio = len(overlap) / min(len(words1), len(words2))
        return overlap_ratio > 0.7

    def _detect_template_type(self, text: str) -> str:
        text = text.strip()
        if re.search(r'```\w*\n.*?```', text, re.DOTALL) or text.startswith('```') or text.endswith('```'):
            return "code_fenced"
        if (text.startswith('{') and text.endswith('}')) or (text.startswith('[') and text.endswith(']')):
            try:
                json.loads(text)
                return "structured_json"
            except Exception:
                pass
        if text.startswith('<') and text.endswith('>'):
            try:
                ET.fromstring(text)
                return "structured_xml"
            except Exception:
                pass
        if re.search(r'^\s*\w+:\s*.*$', text, re.MULTILINE) and not re.search(r'\b(Thought|Action|Observation):', text):
            return "structured_yaml"
        if re.search(r'\b(Thought|Action|Observation):', text):
            return "positional"
        return "unknown"

    def _validate_json_template(self, text: str) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            violations.append({"type": "json_syntax", "issue": f"Invalid JSON syntax: {str(e)}", "position": getattr(e, 'pos', 0)})
        return violations

    def _validate_xml_template(self, text: str) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        try:
            ET.fromstring(text)
        except ET.ParseError as e:
            violations.append({"type": "xml_syntax", "issue": f"Invalid XML syntax: {str(e)}"})
        return violations

    def _validate_positional_template(self, text: str) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        required_sections = ['Thought', 'Action', 'Observation']
        section_patterns = {
            'Thought': r'\b(?:Thought|思考|想法):',
            'Action': r'\b(?:Action|行动|操作):',
            'Observation': r'\b(?:Observation|观察|结果):',
        }
        found_sections = []
        for section, pattern in section_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                found_sections.append((section, match.start(), match.group()))
        found_names = [s for s, _, _ in found_sections]
        for req in required_sections:
            if req not in found_names:
                violations.append({"type": "missing_section", "issue": f"Missing required section: {req}", "severity": "high"})
        if found_sections:
            found_sections.sort(key=lambda x: x[1])
            actual_order = [s for s, _, _ in found_sections]
            expected_order = ['Thought', 'Action', 'Observation']
            for i, actual in enumerate(actual_order):
                if i < len(expected_order) and actual != expected_order[i]:
                    violations.append({"type": "incorrect_sequence", "issue": f"Incorrect section order: expected {expected_order[i]}, got {actual}", "severity": "medium"})
        for section, start_pos, _ in found_sections:
            lines = text[start_pos:].split('\n')
            content_lines = []
            for line in lines[1:]:
                if line.strip() and not re.match(r'\b(?:Thought|Action|Observation|思考|行动|观察):', line, re.IGNORECASE):
                    content_lines.append(line.strip())
                elif re.match(r'\b(?:Thought|Action|Observation|思考|行动|观察):', line, re.IGNORECASE):
                    break
            if not content_lines:
                violations.append({"type": "empty_section", "issue": f"Section {section} has no content", "severity": "medium"})
        if len(found_sections) < 2:
            violations.append({"type": "incomplete_pattern", "issue": f"Incomplete Thought-Action-Observation pattern: only {len(found_sections)} sections found", "severity": "high"})
        return violations

    def _validate_yaml_template(self, text: str) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        try:
            import yaml
            yaml.safe_load(text)
        except ImportError:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and ':' in line:
                    if not re.match(r'^\s*\w+:\s*.*$', line):
                        violations.append({"type": "yaml_syntax", "issue": f"Invalid YAML syntax at line {i+1}: {line.strip()}"})
        except Exception as e:
            violations.append({"type": "yaml_syntax", "issue": f"Invalid YAML syntax: {str(e)}"})
        return violations

    def _validate_code_fenced_template(self, text: str) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        opening_delimiters = re.findall(r'```\w*', text)
        closing_delimiters = re.findall(r'```(?!\w)', text)
        if len(opening_delimiters) != len(closing_delimiters):
            violations.append({"type": "delimiter_mismatch", "issue": f"Mismatched delimiters: {len(opening_delimiters)} opening, {len(closing_delimiters)} closing"})
        language_ids = re.findall(r'```(\w+)', text)
        if language_ids:
            normalized = []
            for lid in language_ids:
                if lid.lower() in ['py', 'python']:
                    normalized.append('python')
                elif lid.lower() in ['js', 'javascript']:
                    normalized.append('javascript')
                else:
                    normalized.append(lid.lower())
            if len(set(normalized)) > 1:
                violations.append({"type": "language_identifier_inconsistency", "issue": f"Inconsistent language identifiers: {language_ids}"})
        for i, block in enumerate(re.findall(r'```\w*\n(.*?)```', text, re.DOTALL)):
            if not block.strip():
                violations.append({"type": "empty_code_block", "issue": f"Empty code block at position {i+1}"})
        return violations

    def _ends_with_punctuation(self, text: str) -> bool:
        text = text.rstrip()
        return text.endswith(('.', '!', '?', '.\n', '!\n', '?\n', '\n'))

    def _is_complete_sentence_basic(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        if text.endswith(('.', '!', '?')):
            return True
        incomplete_patterns = [
            r'\b\w+$', r'\b\w+\s*$', r'\b\w+\s*[a-z]', r'\b\w+\s*and\s*$', r'\b\w+\s*but\s*$',
            r'\b\w+\s*or\s*$', r'\b\w+\s*the\s*$', r'\b\w+\s*a\s*$', r'\b\w+\s*an\s*$',
            r'\b\w+\s*in\s*$', r'\b\w+\s*on\s*$', r'\b\w+\s*at\s*$', r'\b\w+\s*to\s*$',
            r'\b\w+\s*for\s*$', r'\b\w+\s*of\s*$', r'\b\w+\s*with\s*$', r'\b\w+\s*by\s*$',
        ]
        for pattern in incomplete_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        if text.lower().endswith('sent'):
            return False
        if text and text[0].isupper() and len(text.split()) >= 3:
            return True
        return False

    def _simplified_similarity_detection(self, data_entries: List[str]) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        if len(data_entries) < 2:
            return violations
        scores = []
        for i in range(len(data_entries)):
            for j in range(i + 1, len(data_entries)):
                try:
                    score = self._compute_word_overlap_similarity(data_entries[i], data_entries[j])
                    scores.append({'pair': (i, j), 'score': score, 'entry1': data_entries[i][:100], 'entry2': data_entries[j][:100]})
                except Exception:
                    continue
        if not scores:
            return violations
        threshold = getattr(self.config, 'similarity_threshold', 0.7)
        for info in scores:
            if info['score'] < threshold:
                violations.append({
                    'type': 'irrelevant_content',
                    'pair_indices': info['pair'],
                    'similarity_score': info['score'],
                    'threshold': threshold,
                    'entry1': info['entry1'],
                    'entry2': info['entry2'],
                    'issue': f"Low relevance pair detected: similarity {info['score']:.3f} < {threshold}",
                })
        return violations

    def _compute_word_overlap_similarity(self, text1: str, text2: str) -> float:
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        if not words1 or not words2:
            return 0.0
        inter = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return inter / union if union else 0.0

    def _parse_data_entries(self, output: str) -> List[str]:
        """Parse output into data entries for downstream relevance checks."""
        entries: List[str] = []
        if not output:
            return entries
        # Strategy 1: JSON array
        try:
            data = json.loads(output)
            if isinstance(data, list):
                return [str(item) for item in data]
        except (json.JSONDecodeError, TypeError):
            pass
        # Strategy 2: line-separated
        lines = output.strip().split('\n')
        entries = [line.strip() for line in lines if line.strip()]
        # Strategy 3: common separators
        if len(entries) <= 1:
            for sep in ['\n\n', '---', '***', '###']:
                if sep in output:
                    entries = [part.strip() for part in output.split(sep) if part.strip()]
                    break
        # Strategy 4: sentence boundaries
        if len(entries) <= 1:
            sentences = re.split(r'[.!?]+\s+', output)
            entries = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        # Strategy 5: split by periods as fallback
        if len(entries) <= 1 and '.' in output:
            parts = output.split('.')
            entries = [part.strip() for part in parts if part.strip() and len(part.strip()) > 5]
        return entries


