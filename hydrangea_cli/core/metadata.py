"""Metadata management module"""

import os
import csv
import yaml
from typing import List, Dict, Optional, Any
from pathlib import Path


class MetadataManager:
    """Metadata manager"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.apps_csv = self.base_dir / "application.csv"
        self.db_dir = self.base_dir / "db"
        
    def load_applications(self) -> List[Dict[str, Any]]:
        """Load application data"""
        apps = []
        if not self.apps_csv.exists():
            return apps
            
        try:
            with open(self.apps_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Clean data
                    app_data = {
                        'app': row.get('APP', '').strip(),
                        'url': row.get('url', '').strip(),
                        'classification': row.get('classification', '').strip(),
                        'commit_id': row.get('commit id', '').strip(),
                        'llm': row.get('LLM', '').strip(),
                        'llm_deployment': row.get('LLM_Deployment Environments', '').strip(),
                        'vectordb': row.get('vectordb', '').strip(),
                        'vectordb_deployment': row.get('VectorDB_Deployment Environments', '').strip(),
                        'langchain': row.get('langchain', '').strip(),
                        'language': row.get('language', '').strip()
                    }
                    if app_data['app']:  # Only add non-empty applications
                        apps.append(app_data)
        except Exception as e:
            print(f"Warning: Failed to load applications from {self.apps_csv}: {e}")
        return apps
    
    def load_defects(self) -> List[Dict[str, Any]]:
        """Load defect data"""
        defects = []
        if not self.db_dir.exists():
            return defects
            
        for yaml_file in self.db_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    defect_data = yaml.safe_load(f)
                    if defect_data:
                        defects.append(defect_data)
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file}: {e}")
                continue
        return defects
    
    def get_apps_by_filters(self, 
                           classification: Optional[str] = None,
                           llm: Optional[str] = None,
                           llm_deployment: Optional[str] = None,
                           vdb: Optional[str] = None,
                           vdb_deployment: Optional[str] = None,
                           langchain: Optional[str] = None,
                           language: Optional[str] = None) -> List[str]:
        """Filter applications by multiple criteria"""
        apps = self.load_applications()
        filtered_apps = []
        
        for app in apps:
            # Check classification filter condition
            if classification and classification.lower() not in app.get('classification', '').lower():
                continue
                
            # Check LLM filter condition
            if llm and llm.lower() not in app.get('llm', '').lower():
                continue
                
            # Check LLM deployment environment filter condition
            if llm_deployment and llm_deployment.lower() not in app.get('llm_deployment', '').lower():
                continue
                
            # Check vector database filter condition
            if vdb and vdb.lower() not in app.get('vectordb', '').lower():
                continue
                
            # Check vector database deployment environment filter condition
            if vdb_deployment and vdb_deployment.lower() not in app.get('vectordb_deployment', '').lower():
                continue
                
            # Check LangChain filter condition
            if langchain and langchain.lower() not in app.get('langchain', '').lower():
                continue
                
            # Check programming language filter condition
            if language and language.lower() not in app.get('language', '').lower():
                continue
                
            filtered_apps.append(app['app'])
        
        return sorted(list(set(filtered_apps)))  # Remove duplicates and sort
    
    def get_defect_ids(self, app: Optional[str] = None) -> List[str]:
        """Get list of defect IDs"""
        defects = self.load_defects()
        defect_ids = []
        
        for defect in defects:
            if app and app.lower() not in defect.get('app', '').lower():
                continue
            defect_ids.append(defect.get('defect_id', ''))
        
        return sorted([did for did in defect_ids if did])  # Filter empty values and sort
    
    def get_defect_info(self, app: str, defect_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for specific defect"""
        defects = self.load_defects()
        
        for defect in defects:
            if (defect.get('app', '').lower() == app.lower() and 
                defect.get('defect_id', '') == defect_id):
                return defect
        return None
