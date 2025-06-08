"""
Workspace Indexer - Scans and catalogs the entire project structure
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class WorkspaceIndexer:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.index_data = {
            "generated_at": datetime.now().isoformat(),
            "root_path": str(self.root_path),
            "total_files": 0,
            "total_directories": 0,
            "file_types": {},
            "directories": {},
            "agents": {},
            "key_files": {}        }
    
    def scan_workspace(self):
        """Scan the entire workspace and build index"""
        print("Indexing workspace...")
        
        for root, dirs, files in os.walk(self.root_path):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if not self._should_exclude_dir(d)]
            
            rel_path = Path(root).relative_to(self.root_path)
            
            # Index directory
            self._index_directory(rel_path, files)
            
            # Index files
            for file in files:
                if not self._should_exclude_file(file):
                    self._index_file(rel_path / file)
        
        self._identify_agents()
        self._identify_key_files()
        
    def _should_exclude_dir(self, dirname: str) -> bool:
        exclude_dirs = {
            '__pycache__', '.git', 'node_modules', 'venv', 
            '.pytest_cache', '.vscode', '.idea', 'dist', 'build'
        }
        return dirname in exclude_dirs or dirname.startswith('.')
    
    def _should_exclude_file(self, filename: str) -> bool:
        exclude_extensions = {'.pyc', '.pyo', '.pyd', '.so', '.dll'}
        return Path(filename).suffix in exclude_extensions
    
    def _index_directory(self, rel_path: Path, files: List[str]):
        """Index a directory and its contents"""
        self.index_data["total_directories"] += 1
        
        dir_info = {
            "file_count": len(files),
            "python_files": len([f for f in files if f.endswith('.py')]),
            "config_files": len([f for f in files if f.endswith(('.json', '.yml', '.yaml', '.toml'))]),
            "has_init": '__init__.py' in files,
            "has_readme": any(f.lower().startswith('readme') for f in files)
        }
        
        self.index_data["directories"][str(rel_path)] = dir_info
    
    def _index_file(self, file_path: Path):
        """Index individual file"""
        self.index_data["total_files"] += 1
        
        suffix = file_path.suffix.lower()
        self.index_data["file_types"][suffix] = self.index_data["file_types"].get(suffix, 0) + 1
    
    def _identify_agents(self):
        """Identify agent directories and their capabilities"""
        agents_dir = self.root_path / "agents"
        if agents_dir.exists():
            for agent_dir in agents_dir.iterdir():
                if agent_dir.is_dir() and not agent_dir.name.startswith('.'):
                    agent_info = self._analyze_agent(agent_dir)
                    self.index_data["agents"][agent_dir.name] = agent_info
    
    def _analyze_agent(self, agent_path: Path) -> Dict[str, Any]:
        """Analyze an agent directory"""
        config_files = list(agent_path.glob("**/*.json")) + list(agent_path.glob("**/*.yml"))
        python_files = list(agent_path.glob("**/*.py"))
        
        return {
            "path": str(agent_path.relative_to(self.root_path)),
            "python_files": len(python_files),
            "config_files": len(config_files),
            "has_main": (agent_path / "main.py").exists(),
            "has_config": (agent_path / "config.json").exists() or (agent_path / "config.yml").exists(),
            "subdirectories": [d.name for d in agent_path.iterdir() if d.is_dir()],
            "last_modified": os.path.getmtime(agent_path)
        }
    
    def _identify_key_files(self):
        """Identify important project files"""
        key_patterns = {
            "requirements": ["requirements.txt", "requirements*.txt"],
            "config": ["config.json", "config.yml", "*.config.js"],
            "documentation": ["README.md", "WHITEPAPER.md", "docs/**/*.md"],
            "setup": ["setup.py", "pyproject.toml", "package.json"],
            "startup": ["startup.py", "main.py", "app.py"]        }
        
        for category, patterns in key_patterns.items():
            found_files = []
            for pattern in patterns:
                found_files.extend(self.root_path.glob(pattern))
            
            self.index_data["key_files"][category] = [
                str(f.relative_to(self.root_path)) for f in found_files
            ]
    
    def save_index(self):
        """Save index to file"""
        output_path = self.root_path / ".vscode" / "workspace_index.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.index_data, f, indent=2)
        print(f"[OK] Workspace index saved: {output_path}")
        print(f"[STATS] Indexed {self.index_data['total_files']} files in {self.index_data['total_directories']} directories")

if __name__ == "__main__":
    indexer = WorkspaceIndexer()
    indexer.scan_workspace()
    indexer.save_index()