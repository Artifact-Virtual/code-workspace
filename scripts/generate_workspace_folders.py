"""
Dynamic Workspace Folder Generator - Updates VS Code workspace with discovered folders
Windows compatible version with proper Unicode handling
"""
import json
import os
from pathlib import Path
from typing import List, Dict

class WorkspaceFolderGenerator:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.workspace_file = self.root_path / ".vscode" / "context.code-workspace"
        
    def discover_folders(self) -> List[Dict[str, str]]:
        """Discover important folders in the workspace"""
        folders = [
            {"name": "ARTIFACT:0 [root]", "path": "."}
        ]
        
        # Agent folders
        agents_dir = self.root_path / "agents"
        if agents_dir.exists():
            for i, agent_dir in enumerate(sorted(agents_dir.iterdir())):
                if agent_dir.is_dir() and not agent_dir.name.startswith('.'):
                    folders.append({
                        "name": f"ARTIFACT:{i+1} {agent_dir.name}",
                        "path": f"./agents/{agent_dir.name}"
                    })
        
        # Core system folders
        core_folders = [
            ("docs", "./administration/docs"),
            ("scripts", "./.vscode/scripts"),
            ("frontend", "./frontend"),
            ("ecosystem", "./ecosystem"),
            ("research", "./research"),
            ("projects", "./projects"),
            ("library", "./library"),
            ("worxpace", "./worxpace")
        ]
        
        folder_counter = len(folders)
        for name, path in core_folders:
            if (self.root_path / path.lstrip("./")).exists():
                folders.append({
                    "name": f"{name} [A{folder_counter}]",
                    "path": path
                })
                folder_counter += 1
        
        return folders
    
    def update_workspace_file(self):
        """Update the workspace configuration file"""
        workspace_config = {
            "folders": [],
            "settings": {},
            "extensions": {}
        }
        
        if self.workspace_file.exists():
            try:
                # Read current workspace config
                with open(self.workspace_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content:
                    # Parse JSON (handling comments)
                    import re
                    json_content = re.sub(r'//.*', '', content)  # Remove comments
                    workspace_config = json.loads(json_content)
                else:
                    print("Empty workspace file, creating default config")
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Error reading workspace file: {e}")
                print("Creating new workspace config")
        else:
            print(f"Creating new workspace file: {self.workspace_file}")
        
        # Update folders
        discovered_folders = self.discover_folders()
        workspace_config["folders"] = discovered_folders
        
        # Ensure settings and extensions exist
        if "settings" not in workspace_config:
            workspace_config["settings"] = {}
        if "extensions" not in workspace_config:
            workspace_config["extensions"] = {}
        
        # Write back with proper formatting
        updated_content = self._format_workspace_json(workspace_config)
        
        try:
            with open(self.workspace_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"[OK] Updated workspace with {len(discovered_folders)} folders")
            for folder in discovered_folders:
                print(f"   - {folder['name']}")
        except Exception as e:
            print(f"[ERROR] Failed to write workspace file: {e}")
    
    def _format_workspace_json(self, config: dict) -> str:
        """Format workspace JSON with comments and proper structure"""
        settings = config.get("settings", {})
        extensions = config.get("extensions", {})
        tasks = config.get("tasks", {})
        
        return f'''{{
  // VS Code Workspace Configuration - Auto-generated
  "folders": {json.dumps(config["folders"], indent=2)},
  "settings": {json.dumps(settings, indent=2)},
  "extensions": {json.dumps(extensions, indent=2)},
  "tasks": {json.dumps(tasks, indent=2)}
}}'''

if __name__ == "__main__":
    generator = WorkspaceFolderGenerator()
    generator.update_workspace_file()