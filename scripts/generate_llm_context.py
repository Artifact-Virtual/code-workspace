"""
LLM Context Generator - Creates comprehensive context files for AI assistance
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class LLMContextGenerator:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.index_file = self.root_path / ".vscode" / "workspace_index.json"
        
    def load_workspace_index(self) -> Dict[str, Any]:
        """Load the workspace index"""
        if not self.index_file.exists():
            print("❌ Workspace index not found. Run index_workspace.py first.")
            return {}
        
        with open(self.index_file, 'r') as f:
            return json.load(f)
    
    def generate_context(self):
        """Generate comprehensive LLM context"""
        index_data = self.load_workspace_index()
        if not index_data:
            return
        
        context = self._build_context_markdown(index_data)
          # Save context file
        output_path = self.root_path / ".vscode" / "llm_context.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(context)
        
        print(f"[OK] LLM context generated: {output_path}")
        
        # Also generate JSON context for structured access
        json_context = self._build_context_json(index_data)
        json_output_path = self.root_path / ".vscode" / "llm_context.json"
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(json_context, f, indent=2)
        
        print(f"[OK] LLM JSON context generated: {json_output_path}")
    
    def _build_context_markdown(self, index_data: Dict[str, Any]) -> str:
        """Build markdown context for LLMs"""
        context = f"""# ArtifactVirtual Workspace Context

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Root Path:** `{index_data.get('root_path', 'Unknown')}`

## 📊 Workspace Overview

- **Total Files:** {index_data.get('total_files', 0):,}
- **Total Directories:** {index_data.get('total_directories', 0):,}
- **Agents Discovered:** {len(index_data.get('agents', {})):,}

## 🤖 Agents & Components

"""
        
        # Add agent information
        agents = index_data.get('agents', {})
        for agent_name, agent_info in agents.items():
            context += f"""### {agent_name}
- **Path:** `{agent_info.get('path', 'Unknown')}`
- **Python Files:** {agent_info.get('python_files', 0)}
- **Config Files:** {agent_info.get('config_files', 0)}
- **Has Main:** {'✅' if agent_info.get('has_main') else '❌'}
- **Has Config:** {'✅' if agent_info.get('has_config') else '❌'}
- **Subdirectories:** {', '.join(agent_info.get('subdirectories', []))}

"""
        
        # Add file type distribution
        context += "## 📁 File Type Distribution\n\n"
        file_types = index_data.get('file_types', {})
        for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            context += f"- **{ext or 'No extension'}:** {count:,} files\n"
        
        # Add key files
        context += "\n## 🔑 Key Files\n\n"
        key_files = index_data.get('key_files', {})
        for category, files in key_files.items():
            if files:
                context += f"### {category.title()}\n"
                for file in files:
                    context += f"- `{file}`\n"
                context += "\n"
        
        # Add directory structure insights
        context += "## 📂 Directory Insights\n\n"
        directories = index_data.get('directories', {})
        python_dirs = [(path, info) for path, info in directories.items() 
                      if info.get('python_files', 0) > 0]
        
        context += f"**Python-containing directories:** {len(python_dirs)}\n\n"
        for path, info in sorted(python_dirs, key=lambda x: x[1]['python_files'], reverse=True)[:10]:
            context += f"- `{path}`: {info['python_files']} Python files\n"
        
        return context
    
    def _build_context_json(self, index_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build structured JSON context for programmatic access"""
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "workspace_root": str(self.root_path),
                "total_files": index_data.get('total_files', 0),
                "total_directories": index_data.get('total_directories', 0)
            },
            "agents": index_data.get('agents', {}),
            "file_types": index_data.get('file_types', {}),
            "key_files": index_data.get('key_files', {}),
            "capabilities": self._extract_capabilities(index_data),
            "development_patterns": self._analyze_patterns(index_data)
        }
    
    def _extract_capabilities(self, index_data: Dict[str, Any]) -> List[str]:
        """Extract system capabilities from the workspace"""
        capabilities = []
        
        # Check for web frameworks
        file_types = index_data.get('file_types', {})
        if '.html' in file_types or '.tsx' in file_types:
            capabilities.append("Web Frontend Development")
        
        if '.py' in file_types:
            capabilities.append("Python Backend Development")
        
        # Check for specific technologies
        key_files = index_data.get('key_files', {})
        if any('package.json' in f for f in key_files.get('setup', [])):
            capabilities.append("Node.js/JavaScript Development")
        
        if any('requirements.txt' in f for f in key_files.get('requirements', [])):
            capabilities.append("Python Dependency Management")
        
        # Check for agents
        agents = index_data.get('agents', {})
        if agents:
            capabilities.append(f"AI Agent System ({len(agents)} agents)")
        
        return capabilities
    
    def _analyze_patterns(self, index_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze development patterns in the workspace"""
        patterns = {
            "has_testing": False,
            "has_documentation": False,
            "has_ci_cd": False,
            "modular_structure": False
        }
        
        directories = index_data.get('directories', {})
        
        # Check for testing
        if any('test' in path.lower() for path in directories.keys()):
            patterns["has_testing"] = True
        
        # Check for documentation
        if any('doc' in path.lower() for path in directories.keys()):
            patterns["has_documentation"] = True
        
        # Check for CI/CD
        if '.github/workflows' in directories:
            patterns["has_ci_cd"] = True
        
        # Check for modular structure
        if len(directories) > 10:  # Arbitrary threshold
            patterns["modular_structure"] = True
        
        return patterns

if __name__ == "__main__":
    generator = LLMContextGenerator()
    generator.generate_context()