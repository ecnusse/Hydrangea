"""Format analyzer module that integrates format_detector"""

import os
import sys
import subprocess
import shutil
import re
from pathlib import Path
from typing import Optional, Dict, List, Any
import ast
from functools import lru_cache

# Local lightweight components
from .format_detector import FormatDetector
from .config import Config
from .types import ErrorType, DetectionResult

class FormatAnalyzer:
    """Format analyzer that uses format detector to analyze LLM inputs and outputs"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.repos_dir = self.base_dir / "repos"
        self.apps_csv = self.base_dir / "application.csv"
        # Lightweight configuration tweaks
        self.config = Config.create_lightweight_config()
        self.config.enable_embedding_similarity = False
        self.config.enable_detailed_logging = False
        self.config.max_processing_time_ms = 400
        # Runtime controls
        self.max_prompts_per_file = 6
        self.max_completions_per_file = 6
        self.max_text_chars = 4000  # limit per-analysis text length
        self.stop_after_files_with_issues = None  # optional early stop threshold
        # Result cache to avoid re-analyzing identical text
        self._analysis_cache: Dict[tuple, Dict[str, DetectionResult]] = {}
        self.format_detector = FormatDetector(self.config)
    
    def get_app_info(self, app_name: str) -> Optional[Dict[str, str]]:
        """Get application information from application.csv"""
        if not self.apps_csv.exists():
            return None
        
        import csv
        try:
            with open(self.apps_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if app_name.lower() in row.get('APP', '').lower():
                        return {
                            'app': row.get('APP', '').strip(),
                            'url': row.get('url', '').strip(),
                            'commit_id': row.get('commit id', '').strip(),
                            'classification': row.get('classification', '').strip(),
                            'llm': row.get('LLM', '').strip()
                        }
        except Exception as e:
            print(f"Error reading application.csv: {e}")
        
        return None
    
    def clone_repo(self, app_url: str, app_name: str, commit_id: str) -> bool:
        """Clone repository and checkout to specific commit"""
        # Create repos directory if it doesn't exist
        self.repos_dir.mkdir(exist_ok=True)
        
        # Extract repository name from URL
        repo_name = app_name.split('/')[-1]
        repo_path = self.repos_dir / repo_name
        
        # Check if repository already exists
        if repo_path.exists():
            print(f"Repository {repo_name} already exists. Checking out to commit {commit_id}...")
            try:
                subprocess.run(['git', 'checkout', commit_id], cwd=repo_path, check=True, capture_output=True)
                return True
            except subprocess.CalledProcessError as e:
                print(f"Failed to checkout to commit {commit_id}: {e}")
                return False
        
        # Clone repository
        print(f"Cloning repository {app_url}...")
        try:
            subprocess.run(['git', 'clone', app_url, str(repo_path)], check=True, capture_output=True)
            subprocess.run(['git', 'checkout', commit_id], cwd=repo_path, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e}")
            if repo_path.exists():
                shutil.rmtree(repo_path)
            return False
    
    def find_llm_interactions(self, repo_path: str) -> List[Dict[str, Any]]:
        """Find files in repository that interact with LLMs"""
        interactions = []
        
        # Directory filter configuration (tunable)
        include_dir_names = {
            'src', 'app', 'apps', 'backend', 'server', 'api', 'services', 'agent', 'agents', 'bot', 'cli', 'cmd'
        }
        exclude_dir_names = {
            'tests', 'test', 'docs', 'doc', 'example', 'examples', 'example_data', 'data', 'dataset', 'datasets',
            'migrations', 'notebooks', 'third_party', 'third-party', 'node_modules', 'dist', 'build', '.venv', '.git',
            '__pycache__'
        }

        # File filter configuration
        max_file_size_bytes = 250 * 1024  # 250KB: skip oversized files to reduce noise

        # Grouped keyword dictionary for maintainability and extensibility
        llm_keywords = {
            # --- 1. Mainstream LLM providers ---
            "providers": [
                "openai", "anthropic", "claude", "gemini", "google", "palm",
                "cohere", "mistral", "llama", "groq", "xai", "deepseek", "huggingface",
                "transformers", "ollama", "replicate"
            ],

            # --- 2. Typical function or class names ---
            "calls": [
                "ChatCompletion", "Completion", "Inference", "Pipeline",
                "generate", "completion", "chat", "respond", "infer", "predict"
            ],

            # --- 3. Prompt / Input / Output related ---
            "prompt": [
                "prompt", "instruction", "system_message", "user_message", "messages=",
                "context", "query", "input_text", "task_description", "prompt_template"
            ],
            "output": [
                "response", "answer", "reply", "output_text", "model_output", "generated_text"
            ],

            # --- 4. Parameter hints (indicative of LLM calls) ---
            "params": [
                "model=", "temperature=", "max_tokens", "top_p", "nucleus", "frequency_penalty",
                "presence_penalty", "stop_sequences", "stream=True"
            ],

            # --- 5. API configuration / auth ---
            "config": [
                "api_key", "api_base", "endpoint", "client", "session", "headers"
            ],

            # --- 6. LLM framework wrappers ---
            "frameworks": [
                "langchain", "llama_index", "transformers", "guidance", "vllm",
                "autogen", "haystack", "griptape", "fastchat", "litellm"
            ]
        }
        
        # Prioritize common application directories
        priority_dirs = ['scripts', 'src', 'app', 'main']
        
        # Search priority directories first
        for dir_name in priority_dirs:
            priority_path = os.path.join(repo_path, dir_name)
            if os.path.exists(priority_path) and os.path.isdir(priority_path):
                interactions.extend(self._search_dir_for_llm(
                    priority_path, repo_path, llm_keywords,
                    include_dir_names, exclude_dir_names, max_file_size_bytes
                ))
        
        # Then search the whole repository
        interactions.extend(self._search_dir_for_llm(
            str(repo_path), str(repo_path), llm_keywords,
            include_dir_names, exclude_dir_names, max_file_size_bytes
        ))
        
        # Deduplicate by file path
        unique_interactions = {item['file_path']: item for item in interactions}.values()
        return list(unique_interactions)
        
    def _search_dir_for_llm(self, search_path: str, repo_path: str, llm_keywords: dict,
                             include_dir_names: set, exclude_dir_names: set,
                             max_file_size_bytes: int) -> List[Dict[str, Any]]:
        """Search for LLM interaction files with directory filtering + AST + keyword fusion."""
        interactions = []
        
        for root, dirs, files in os.walk(search_path):
            # Directory-level filtering: prune early
            pruned_dirs = []
            for d in list(dirs):
                full = os.path.join(root, d)
                name = os.path.basename(full)
                # Skip excluded directories unless explicitly included
                if name in exclude_dir_names and name not in include_dir_names:
                    pruned_dirs.append(d)
            for d in pruned_dirs:
                dirs.remove(d)

            # If root is not under include dirs, mark soft skip to reduce scanning intensity
            root_parts = set([p for p in os.path.relpath(root, repo_path).split(os.sep) if p and p != '.'])
            soft_skip_root = (len(root_parts.intersection(include_dir_names)) == 0)

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    # File size filter
                    try:
                        if os.path.getsize(file_path) > max_file_size_bytes and soft_skip_root:
                            continue
                    except OSError:
                        continue
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                            # 1) AST detection (precise, preferred)
                            ast_detected, ast_info = self._ast_has_llm_calls(content)

                            # 2) Keyword detection as supplement: require provider/framework + call-like category
                            has_keyword, hit_categories = self._contains_llm_keyword(content, llm_keywords)
                            strong_keyword = False
                            if has_keyword:
                                has_provider = any(cat in hit_categories for cat in ['providers', 'frameworks'])
                                has_call_like = any(cat in hit_categories for cat in ['calls', 'params', 'output', 'prompt'])
                                strong_keyword = has_provider and has_call_like
                            
                            # Decision: AST hit OR strong keyword hit; avoid heavy excluded roots
                            if ast_detected or (strong_keyword and not soft_skip_root):
                                prompts = self._extract_prompts(content)
                                completions = self._extract_completions(content)
                                
                                # Add file even when no explicit prompt/completion is extracted to avoid misses
                                interactions.append({
                                    'file_path': file_path,
                                    'relative_path': os.path.relpath(file_path, repo_path),
                                    'prompts': prompts,
                                    'completions': completions,
                                    'hit_categories': hit_categories,  # record matched keyword categories
                                    'ast_calls': ast_info.get('calls', []),
                                    'ast_providers': ast_info.get('providers', [])
                                })
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
                        continue
        
        return interactions
        
    def _contains_llm_keyword(self, content: str, llm_keywords: dict) -> tuple[bool, list]:
        """Check whether content contains LLM-related keywords and return matched categories."""
        import re
        hit_categories = []
        
        # Fast path: framework names are very indicative
        if any(fw in content.lower() for fw in llm_keywords.get('frameworks', [])):
            hit_categories.append('frameworks')
            return True, hit_categories
        
        # Line-wise keyword scan
        for line in content.split('\n'):
            for category, words in llm_keywords.items():
                for word in words:
                    # Use boundary-aware regex; treat tokens with '=' specially
                    if '=' in word:
                        pattern = rf"{re.escape(word)}"
                    else:
                        pattern = rf"\b{re.escape(word)}\b"
                    
                    if re.search(pattern, line, re.IGNORECASE):
                        if category not in hit_categories:
                            hit_categories.append(category)
        
        return len(hit_categories) > 0, hit_categories
    
    def _extract_prompts(self, content: str) -> List[str]:
        """Extract potential LLM prompts from code"""
        prompts = []
        
        # Enhanced regex patterns
        patterns = [
            r'(prompt|messages?|system_message|user_message|instructions?|context)\s*=\s*[\'"](.*?)[\'"]',
            r'prompt\s*:\s*[\'"](.*?)[\'"]',
            r'(system|user|assistant)\s*=\s*[\'"](.*?)[\'"]',
            r'create_chat_message\(\s*[\'"](system|user|assistant)[\'"\s,]+[\'"](.*?)[\'"]',
            r'"content"\s*:\s*[\'"](.*?)[\'"]',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # 处理元组匹配，取最后一个非关键词元素
                    prompt_text = match[-1].strip() if len(match) > 1 else ''
                    if prompt_text:
                        prompts.append(prompt_text)
                else:
                    prompts.append(match.strip())
        
        # Multiline patterns
        multiline_patterns = [
            r'(prompt|messages?)\s*=\s*\[([\s\S]*?)\]',  # list-like prompt assignments
            r'"""([\s\S]*?)"""',  # triple-quoted strings
        ]
        
        for pattern in multiline_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) > 1 and match[1].strip():
                        prompts.append(match[1].strip())
                elif match.strip():
                    prompts.append(match.strip())
        
        # Deduplicate and drop very short strings
        return list(set([p for p in prompts if p and len(p) > 10]))
    
    def _extract_completions(self, content: str) -> List[str]:
        """Extract potential LLM completions or response handling from code"""
        completions = []
        
        # Enhanced regex patterns
        patterns = [
            r'(completion|response|result|output)\s*=\s*client\.(create|generate|complete|chat_completion)',
            r'(completion|response|result|output)\s*=\s*openai\.(ChatCompletion|Completion|Image)\.create',
            r'LLM\.(generate|create|predict|call|query)',
            r'(response|output|result)\s*\.\s*(content|text|choices|data)',
            r'(response|output|result)\s*\[\s*["\']choices["\']\s*\]',
            r'choices\[\d+\]\.\s*(message|content)',
            r'call_ai_function',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    completions.append(' '.join(match))
                else:
                    completions.append(match)
        
        # JSON parsing patterns on responses
        json_patterns = [
            r'json\.loads\(\s*(response|output|result)',
            r'\.json\(\)\s*',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            completions.extend(matches)
        
        return list(set(completions))

    def _ast_has_llm_calls(self, content: str) -> tuple[bool, Dict[str, Any]]:
        """Use AST to detect concrete LLM calls; returns (hit, details)."""
        providers = set()
        calls = []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return False, {'providers': [], 'calls': []}

        # Map import aliases, e.g., `import openai as ai` -> {'ai': 'openai'}
        alias_to_module: Dict[str, str] = {}
        from_import_members: Dict[str, str] = {}

        llm_provider_modules = {
            'openai', 'anthropic', 'google', 'cohere', 'transformers', 'langchain', 'llama_index', 'vllm',
            'groq', 'mistral', 'replicate', 'ollama', 'litellm'
        }

        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    name = alias.name.split('.')[0]
                    asname = alias.asname or name
                    alias_to_module[asname] = name
                    if name in llm_provider_modules:
                        providers.add(name)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                base = (node.module or '').split('.')[0]
                for alias in node.names:
                    asname = alias.asname or alias.name
                    from_import_members[asname] = f"{base}.{alias.name}" if base else alias.name
                    mod = base or alias.name.split('.')[0]
                    if mod in llm_provider_modules:
                        providers.add(mod)

        class CallVisitor(ast.NodeVisitor):
            def get_attr_chain(self, node: ast.AST) -> str:
                parts = []
                cur = node
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                parts.reverse()
                return '.'.join(parts)

            def visit_Call(self, node: ast.Call) -> None:
                target = ''
                if isinstance(node.func, ast.Attribute):
                    target = self.get_attr_chain(node.func)
                elif isinstance(node.func, ast.Name):
                    # Might come from `from ... import ...`
                    target = from_import_members.get(node.func.id, node.func.id)

                # Normalize root module using import alias map
                if target:
                    first = target.split('.')[0]
                    root_mod = alias_to_module.get(first, first)
                    norm = '.'.join([root_mod] + target.split('.')[1:]) if target else ''

                    # Common suspicious call suffixes for LLM APIs
                    suspicious_suffixes = (
                        'ChatCompletion.create', 'Completion.create', 'chat.completions.create',
                        'responses.create', 'generate', 'invoke', 'predict', 'run', 'call', 'complete'
                    )

                    for suf in suspicious_suffixes:
                        if norm.endswith(suf) or norm == suf:
                            calls.append(norm)
                            break

                self.generic_visit(node)

        ImportVisitor().visit(tree)
        CallVisitor().visit(tree)

        hit = len(calls) > 0 or len(providers) > 0
        return hit, {'providers': sorted(providers), 'calls': calls}
    
    def analyze_format(self, text: str, func_name: str = "default") -> Dict[str, DetectionResult]:
        """Analyze text for format issues using format detector"""
        results = {
            'template_discrepancy': self.format_detector.detect_template_discrepancy(text, func_name),
            'data_segmentation': self.format_detector.detect_data_segmentation_issues(text, func_name),
            'context_construction': self.format_detector.detect_context_construction_issues(text, func_name)
        }
        return results

    def _analyze_format_cached(self, text: str, func_name: str) -> Dict[str, DetectionResult]:
        """Cached analysis keyed by (func_name, text) to reduce duplicate work."""
        key = (func_name, text)
        cached = self._analysis_cache.get(key)
        if cached is not None:
            return cached
        # Trim long text to avoid slow analysis
        trimmed = self._trim_text(text)
        results = self.analyze_format(trimmed, func_name)
        self._analysis_cache[key] = results
        return results

    def _trim_text(self, text: str) -> str:
        if not text:
            return text
        if len(text) <= self.max_text_chars:
            return text
        # Keep head and tail to preserve structure
        head = text[: self.max_text_chars // 2]
        tail = text[- self.max_text_chars // 2 :]
        return head + "\n...\n" + tail

    def _prioritize_samples(self, items: List[str], limit: int) -> List[str]:
        """Score and take top-N candidates to improve signal-to-noise ratio."""
        if limit <= 0:
            return []
        if len(items) <= limit:
            return items
        def score(s: str) -> int:
            s_low = s.lower()
            sc = 0
            # Structural hints
            if '{' in s and '}' in s:
                sc += 3
            if '```' in s:
                sc += 3
            if ':' in s and '\n' in s:
                sc += 2
            # Chain-of-thought-like markers
            if 'thought:' in s_low or 'action:' in s_low or 'observation:' in s_low:
                sc += 2
            # JSON-related markers
            if '"content"' in s or "'content'" in s:
                sc += 1
            # Slightly prefer shorter, denser samples
            sc -= max(0, len(s) // 2000)
            return sc
        ranked = sorted(items, key=score, reverse=True)
        # Deduplicate while preserving order
        seen = set()
        unique_ranked = []
        for x in ranked:
            if x in seen:
                continue
            seen.add(x)
            unique_ranked.append(x)
        return unique_ranked[:limit]
    
    def analyze_application(self, app_name: str) -> Dict[str, Any]:
        """Analyze an application for LLM input/output format issues"""
        # Get application information
        app_info = self.get_app_info(app_name)
        if not app_info:
            return {
                'success': False,
                'error': f"Application '{app_name}' not found in application.csv"
            }
        
        # Clone repository if needed
        repo_name = app_info['app'].split('/')[-1]
        if not self.clone_repo(app_info['url'], app_info['app'], app_info['commit_id']):
            return {
                'success': False,
                'error': f"Failed to clone or checkout repository for {app_info['app']}"
            }
        
        repo_path = self.repos_dir / repo_name
        
        # Find LLM interactions
        print(f"Finding LLM interactions in {repo_name}...")
        interactions = self.find_llm_interactions(repo_path)
        
        # Analyze each interaction
        analysis_results = []
        files_with_issues = 0
        for interaction in interactions:
            file_results = {
                'file': interaction['relative_path'],
                'prompt_issues': [],
                'completion_issues': []
            }
            
            # Analyze prompts
            # Sample and deduplicate
            prompt_candidates = self._prioritize_samples(list(set(interaction['prompts'])), self.max_prompts_per_file)
            for prompt in prompt_candidates:
                prompt_analysis = self._analyze_format_cached(prompt, "prompt")
                for issue_type, result in prompt_analysis.items():
                    if result.detected:
                        file_results['prompt_issues'].append({
                            'type': issue_type,
                            'severity': result.severity,
                            'details': result.details
                        })
            
            # Analyze completions (or potential completion handling)
            completion_candidates = self._prioritize_samples(list(set(interaction['completions'])), self.max_completions_per_file)
            for completion in completion_candidates:
                completion_analysis = self._analyze_format_cached(completion, "completion")
                for issue_type, result in completion_analysis.items():
                    if result.detected:
                        file_results['completion_issues'].append({
                            'type': issue_type,
                            'severity': result.severity,
                            'details': result.details
                        })
            
            # Only add to results if there are issues
            if file_results['prompt_issues'] or file_results['completion_issues']:
                analysis_results.append(file_results)
                files_with_issues += 1
                if self.stop_after_files_with_issues and files_with_issues >= self.stop_after_files_with_issues:
                    break
        
        return {
            'success': True,
            'app_info': app_info,
            'analysis_results': analysis_results,
            'total_files_analyzed': len(interactions),
            'files_with_issues': len(analysis_results)
        }