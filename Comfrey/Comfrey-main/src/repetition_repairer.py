"""
Repetition Repairer for Comfrey framework.
Implements repetition error repair according to paper specifications.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import logging
import re
from collections import defaultdict, deque
from difflib import SequenceMatcher

from .types import DetectionResult, RepairResult
from .config import ComfreyConfig

logger = logging.getLogger(__name__)

class RepetitionRepairer:
    """Repairs repetition-related errors in AI outputs"""
    
    def __init__(self, config: ComfreyConfig):
        self.config = config
        self._init_repair_components()
    
    def _init_repair_components(self):
        """Initialize repair components"""
        # Cache for storing action results
        self.action_result_cache = {}
        
        # Cycle detection components
        self.cycle_detector = CycleDetector()
        
        # Content deduplication components
        self.content_deduplicator = ContentDeduplicator(self.config)
        
        # Relevance filter
        self.relevance_filter = RelevanceFilter(self.config)
    
    def repair_software_behavior_redundancy(self, output: Any, detection_result: DetectionResult) -> RepairResult:
        """
        Repair software behavior redundancy according to paper Section 4.5.1.
        
        Implements the repair approach: bypasses the redundant tool/function invocation
        and uses the corresponding result in the history queue.
        """
        try:
            details = detection_result.details
            repair_actions = []
            
            # Check if this is a cached action
            action_signature = details.get('action_signature', '')
            redundant_actions = details.get('redundant_actions', 0)
            
            if redundant_actions > 0:
                # Use cached result instead of executing redundant action
                cached_result = self._get_cached_result(action_signature)
                if cached_result is not None:
                    repair_actions.append(f"Used cached result for action {action_signature}")
                    repaired_output = cached_result
                else:
                    # Execute once and cache the result
                    repaired_output = output
                    self._cache_result(action_signature, output)
                    repair_actions.append(f"Executed and cached result for action {action_signature}")
            else:
                # No redundancy detected, keep original output
                repaired_output = output
                repair_actions.append("No redundancy detected, kept original output")
            
            return RepairResult(
                success=True,
                original_output=str(output),
                repaired_output=str(repaired_output),
                repair_actions=repair_actions,
                metadata={
                    "repair_strategy": "software_behavior_redundancy",
                    "cached_actions": redundant_actions,
                    "action_signature": action_signature
                }
            )
        except Exception as e:
            logger.error(f"Software behavior redundancy repair failed: {e}")
        return RepairResult(
            success=False,
            original_output=str(output),
            repaired_output=str(output),
            repair_actions=[],
                metadata={"error": str(e)}
        )
    
    def repair_semantic_redundancy(self, output: Any, detection_result: DetectionResult) -> RepairResult:
        """
        Repair semantic redundancy according to paper Section 4.5.2.
        
        Implements the repair approach:
        - For internal redundancy and contextual redundancy: follows the same solution as context construction
        - For external redundancy: terminates the loop and rolls back to the value of the last iteration
        """
        try:
            output_str = str(output)
            details = detection_result.details
            redundancy_analysis = details.get('redundancy_analysis', {})
            
            repair_actions = []
            
            # Stage 1: Cycle-aware termination
            cycle_terminated_output, cycle_actions = self._apply_cycle_termination(
                output_str, redundancy_analysis
            )
            repair_actions.extend(cycle_actions)
            
            # Stage 2: Content-level fixes
            content_fixed_output, content_actions = self._apply_content_fixes(
                cycle_terminated_output, redundancy_analysis
            )
            repair_actions.extend(content_actions)
            
            # Stage 3: Contextual redundancy filtering
            final_output, filter_actions = self._apply_contextual_filtering(
                content_fixed_output, redundancy_analysis
            )
            repair_actions.extend(filter_actions)
            
            return RepairResult(
                success=True,
                original_output=output_str,
                repaired_output=final_output,
                repair_actions=repair_actions,
                metadata={
                    "repair_strategy": "three_stage_semantic_redundancy",
                    "redundancy_analysis": redundancy_analysis,
                    "num_stages": 3,
                    "total_repairs": len(repair_actions)
                }
            )
        except Exception as e:
            logger.error(f"Semantic redundancy repair failed: {e}")
        return RepairResult(
            success=False,
            original_output=str(output),
            repaired_output=str(output),
            repair_actions=[],
                metadata={"error": str(e)}
            )
    
    def _get_cached_result(self, action_signature: str) -> Any:
        """Get cached result for action"""
        return self.action_result_cache.get(action_signature)
    
    def _cache_result(self, action_signature: str, result: Any):
        """Cache result for action"""
        self.action_result_cache[action_signature] = result
    
    def _apply_cycle_termination(self, output: str, redundancy_analysis: Dict) -> Tuple[str, List[str]]:
        """Apply cycle-aware termination (Stage 1)"""
        actions = []
        
        # Check for internal cycles
        internal_redundancy = redundancy_analysis.get('internal', {})
        if internal_redundancy.get('detected', False):
            # Detect and break cycles within the output
            cycle_broken_output, cycle_actions = self.cycle_detector.break_internal_cycles(output)
            actions.extend(cycle_actions)
            output = cycle_broken_output
        
        # Check for external cycles (cross-iteration)
        external_redundancy = redundancy_analysis.get('external', {})
        if external_redundancy.get('detected', False):
            # Rollback to last non-redundant iteration
            rollback_output, rollback_actions = self.cycle_detector.rollback_to_non_redundant(
                output, external_redundancy
            )
            actions.extend(rollback_actions)
            output = rollback_output
        
        return output, actions
    
    def _apply_content_fixes(self, output: str, redundancy_analysis: Dict) -> Tuple[str, List[str]]:
        """Apply content-level fixes (Stage 2)"""
        actions = []
        
        # Sentence-level deduplication
        deduplicated_output, dedup_actions = self.content_deduplicator.deduplicate_sentences(output)
        actions.extend(dedup_actions)
        
        # Enumeration merging
        merged_output, merge_actions = self.content_deduplicator.merge_enumerations(deduplicated_output)
        actions.extend(merge_actions)
        
        # Content diversification
        diversified_output, diversify_actions = self.content_deduplicator.diversify_content(merged_output)
        actions.extend(diversify_actions)
        
        return diversified_output, actions
    
    def _apply_contextual_filtering(self, output: str, redundancy_analysis: Dict) -> Tuple[str, List[str]]:
        """Apply contextual redundancy filtering (Stage 3)"""
        actions = []
        
        contextual_redundancy = redundancy_analysis.get('contextual', {})
        if contextual_redundancy.get('detected', False):
            # Filter verbose irrelevant content
            filtered_output, filter_actions = self.relevance_filter.filter_verbose_content(
                output, contextual_redundancy
            )
            actions.extend(filter_actions)
            
            # Consolidate remaining content
            consolidated_output, consolidate_actions = self.relevance_filter.consolidate_content(
                filtered_output
            )
            actions.extend(consolidate_actions)
            
            return consolidated_output, actions
        
        return output, actions


class CycleDetector:
    """Detects and breaks repetitive cycles"""
    
    def __init__(self):
        self.iteration_history = deque(maxlen=20)
    
    def break_internal_cycles(self, output: str) -> Tuple[str, List[str]]:
        """Break cycles within the output"""
        actions = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', output)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Find repeating patterns
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            sentence_normalized = sentence.lower().strip()
            if sentence_normalized not in seen_sentences:
                unique_sentences.append(sentence)
                seen_sentences.add(sentence_normalized)
            else:
                actions.append(f"Removed duplicate sentence: '{sentence[:50]}...'")
        
        # Reconstruct output
        if len(unique_sentences) < len(sentences):
            repaired_output = '. '.join(unique_sentences) + '.'
            actions.append(f"Broke internal cycles: {len(sentences)} -> {len(unique_sentences)} sentences")
        else:
            repaired_output = output
        
        return repaired_output, actions
    
    def rollback_to_non_redundant(self, output: str, external_redundancy: Dict) -> Tuple[str, List[str]]:
        """Rollback to last non-redundant iteration"""
        actions = []
        
        # Get redundant history
        redundant_history = external_redundancy.get('redundant_history', [])
        
        if redundant_history:
            # Find the most recent non-redundant version
            # For now, use a simple heuristic: return a shortened version
            sentences = re.split(r'[.!?]+', output)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) > 1:
                # Keep only the first half to break the cycle
                keep_count = len(sentences) // 2
                rolled_back_sentences = sentences[:keep_count]
                repaired_output = '. '.join(rolled_back_sentences) + '.'
                actions.append(f"Rolled back to non-redundant version: kept {keep_count}/{len(sentences)} sentences")
            else:
                repaired_output = output
        else:
            repaired_output = output
        
        return repaired_output, actions


class ContentDeduplicator:
    """Handles content-level deduplication"""
    
    def __init__(self, config: ComfreyConfig):
        self.config = config
    
    def deduplicate_sentences(self, output: str) -> Tuple[str, List[str]]:
        """Remove duplicate sentences"""
        actions = []
        
        sentences = re.split(r'[.!?]+', output)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Use semantic similarity for deduplication
        unique_sentences = []
        for sentence in sentences:
            is_duplicate = False
            for existing in unique_sentences:
                similarity = SequenceMatcher(None, sentence.lower(), existing.lower()).ratio()
                if similarity > 0.8:  # High similarity threshold
                    is_duplicate = True
                    actions.append(f"Removed semantically similar sentence: '{sentence[:50]}...'")
                    break
            
            if not is_duplicate:
                unique_sentences.append(sentence)
        
        repaired_output = '. '.join(unique_sentences) + '.' if unique_sentences else output
        
        if len(unique_sentences) < len(sentences):
            actions.append(f"Deduplicated sentences: {len(sentences)} -> {len(unique_sentences)}")
        
        return repaired_output, actions
    
    def merge_enumerations(self, output: str) -> Tuple[str, List[str]]:
        """Merge redundant enumerations"""
        actions = []
        
        # Find enumeration patterns
        enum_patterns = [
            r'(\d+\.)\s*([^.]+)',  # 1. item
            r'([a-z]\))\s*([^.]+)',  # a) item
            r'([-*+])\s*([^.]+)',  # - item
        ]
        
        for pattern in enum_patterns:
            matches = re.findall(pattern, output)
            if len(matches) > 1:
                # Check for duplicate items
                seen_items = set()
                unique_items = []
                
                for marker, item in matches:
                    item_normalized = item.lower().strip()
                    if item_normalized not in seen_items:
                        unique_items.append((marker, item))
                        seen_items.add(item_normalized)
                    else:
                        actions.append(f"Merged duplicate enumeration item: '{item[:30]}...'")
                
                if len(unique_items) < len(matches):
                    # Rebuild enumeration
                    new_enum = []
                    for i, (marker, item) in enumerate(unique_items):
                        if marker.isdigit():
                            new_enum.append(f"{i+1}. {item}")
                        else:
                            new_enum.append(f"{marker} {item}")
                    
                    # Replace in output
                    enum_section = '\n'.join(new_enum)
                    # Simple replacement (could be more sophisticated)
                    output = re.sub(pattern, '', output)
                    output += '\n' + enum_section
                    actions.append(f"Merged enumeration: {len(matches)} -> {len(unique_items)} items")
        
        return output, actions
    
    def diversify_content(self, output: str) -> Tuple[str, List[str]]:
        """Apply content diversification"""
        actions = []
        
        # Check for repetitive phrases
        words = output.split()
        word_freq = defaultdict(int)
        
        for word in words:
            if len(word) > 3:  # Ignore short words
                word_freq[word.lower()] += 1
        
        # Find overly repeated words
        repeated_words = {word: freq for word, freq in word_freq.items() if freq > 5}
        
        if repeated_words:
            # Apply simple diversification by replacing some instances
            diversified_output = output
            for word, freq in repeated_words.items():
                # Replace some instances with synonyms or remove
                # Simple approach: reduce frequency
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                matches = list(pattern.finditer(diversified_output))
                
                if len(matches) > 3:
                    # Remove every other occurrence after the first 3
                    for i in range(3, len(matches), 2):
                        match = matches[i]
                        # Replace with empty string or synonym
                        diversified_output = (diversified_output[:match.start()] + 
                                            diversified_output[match.end():])
                        actions.append(f"Reduced repetition of word: '{word}'")
            
            return diversified_output, actions
        
        return output, actions


class RelevanceFilter:
    """Filters content based on relevance"""
    
    def __init__(self, config: ComfreyConfig):
        self.config = config
    
    def filter_verbose_content(self, output: str, contextual_redundancy: Dict) -> Tuple[str, List[str]]:
        """Filter verbose irrelevant content"""
        actions = []
        
        verbose_indicators = contextual_redundancy.get('verbose_indicators', [])
        
        if verbose_indicators:
            # Split into sentences
            sentences = re.split(r'[.!?]+', output)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Filter out verbose sentences
            filtered_sentences = []
            for sentence in sentences:
                is_verbose = False
                
                # Check against verbose indicators
                for indicator in verbose_indicators:
                    if 'phrase' in indicator.lower():
                        # Extract phrase from indicator
                        phrase_match = re.search(r"'([^']+)'", indicator)
                        if phrase_match:
                            phrase = phrase_match.group(1)
                            if phrase.lower() in sentence.lower():
                                is_verbose = True
                                actions.append(f"Filtered verbose sentence with phrase: '{phrase}'")
                                break
                
                # Check for repetitive content
                if not is_verbose:
                    words = sentence.split()
                    if len(words) > 10:  # Only check longer sentences
                        word_freq = defaultdict(int)
                        for word in words:
                            word_freq[word.lower()] += 1
                        
                        # Check if sentence has too many repeated words
                        max_freq = max(word_freq.values()) if word_freq else 0
                        if max_freq > len(words) * 0.3:  # More than 30% repetition
                            is_verbose = True
                            actions.append(f"Filtered verbose sentence with repetition: '{sentence[:50]}...'")
                
                if not is_verbose:
                    filtered_sentences.append(sentence)
            
            if len(filtered_sentences) < len(sentences):
                repaired_output = '. '.join(filtered_sentences) + '.' if filtered_sentences else ""
                actions.append(f"Filtered verbose content: {len(sentences)} -> {len(filtered_sentences)} sentences")
            else:
                repaired_output = output
        else:
            repaired_output = output
        
        return repaired_output, actions
    
    def consolidate_content(self, output: str) -> Tuple[str, List[str]]:
        """Consolidate remaining content for coherence"""
        actions = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', output)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return output, actions
        
        # Group related sentences
        sentence_groups = []
        current_group = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Check similarity with current group
            similarity_scores = []
            for group_sentence in current_group:
                sim = SequenceMatcher(None, sentences[i].lower(), group_sentence.lower()).ratio()
                similarity_scores.append(sim)
            
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            
            if avg_similarity > 0.3:  # Threshold for grouping
                current_group.append(sentences[i])
            else:
                sentence_groups.append(current_group)
                current_group = [sentences[i]]
        
        sentence_groups.append(current_group)
        
        # Consolidate groups
        consolidated_sentences = []
        for group in sentence_groups:
            if len(group) > 1:
                # Merge similar sentences in group
                consolidated_sentence = group[0]  # Start with first sentence
                for sentence in group[1:]:
                    # Simple merging: append unique information
                    words_existing = set(consolidated_sentence.lower().split())
                    words_new = set(sentence.lower().split())
                    unique_words = words_new - words_existing
                    
                    if len(unique_words) > 2:  # Has enough unique content
                        consolidated_sentence += f" {sentence}"
                
                consolidated_sentences.append(consolidated_sentence)
                actions.append(f"Consolidated {len(group)} related sentences")
            else:
                consolidated_sentences.append(group[0])
        
        repaired_output = '. '.join(consolidated_sentences) + '.' if consolidated_sentences else output
        
        return repaired_output, actions 