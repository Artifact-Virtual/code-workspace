"""
Master Workspace Automation Script - Windows Compatible Version
Runs all workspace indexing and context generation with improved dependency management
No emoji characters to avoid Windows console encoding issues
"""
import subprocess
import sys
import os
from pathlib import Path
import json

def check_and_install_dependencies() -> bool:
    """Check for requirements.txt and install missing dependencies"""
    requirements_file = Path(__file__).parent.parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("WARNING: No requirements.txt found, creating one...")
        create_requirements_file()
        return True
    
    try:
        print("Checking dependencies...")
        import importlib
        
        # Map pip package names to their import names (this fixes the Beautiful Soup issue)
        package_mapping = {
            'langgraph': 'langgraph',
            'langchain-core': 'langchain_core', 
            'matplotlib': 'matplotlib',
            'pandas': 'pandas',
            'requests': 'requests',
            'beautifulsoup4': 'bs4',  # This was the issue - bs4 is the import name
            'numpy': 'numpy'
        }
        
        missing_packages = []
        print("Checking installed packages...")
        
        for pip_name, import_name in package_mapping.items():
            try:
                importlib.import_module(import_name)
                print(f"[OK] {pip_name} is available")
            except ImportError:
                print(f"[MISSING] {pip_name} is missing")
                missing_packages.append(pip_name)
        
        if missing_packages:
            print(f"Installing missing packages: {', '.join(missing_packages)}")
            
            # Install packages individually for better error handling
            for package in missing_packages:
                print(f"   Installing {package}...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"   [OK] Successfully installed {package}")
                else:
                    print(f"   [ERROR] Failed to install {package}: {result.stderr}")
                    # Continue with other packages instead of failing completely
            
            print("Dependency installation completed!")
            return True
        else:
            print("All dependencies are already installed!")
            return True
            
    except Exception as e:
        print(f"ERROR: Error checking dependencies: {e}")
        return False

def create_requirements_file():
    """Create a requirements.txt file with necessary packages"""
    requirements_content = """# Core LangGraph and LangChain packages
langgraph>=0.0.40
langchain-core>=0.1.0
langchain-openai>=0.1.0
langchain-anthropic>=0.1.0
langchain-google-genai>=1.0.0
langchain-community>=0.0.20
langchain-experimental>=0.0.50

# Data analysis and visualization
matplotlib>=3.7.0
pandas>=2.0.0
numpy>=1.24.0
seaborn>=0.12.0
plotly>=5.15.0

# Web scraping and API tools
requests>=2.31.0
beautifulsoup4>=4.12.0
duckduckgo-search>=3.8.0

# Utility packages
python-dotenv>=1.0.0
pydantic>=2.0.0
"""
    
    requirements_file = Path(__file__).parent.parent.parent / "requirements.txt"
    with open(requirements_file, 'w') as f:
        f.write(requirements_content)
    print(f"Created requirements.txt at {requirements_file}")

def setup_startup_automation():
    """Setup automatic startup execution"""
    try:
        # Create startup configuration
        vscode_dir = Path(__file__).parent.parent
        tasks_file = vscode_dir / "tasks.json"
        settings_file = vscode_dir / "settings.json"
        
        # VS Code tasks.json for running on workspace open
        automation_task = {
            "label": "Workspace Automation",
            "type": "shell",
            "command": sys.executable,
            "args": [str(Path(__file__).resolve())],
            "group": "build",
            "presentation": {
                "echo": True,
                "reveal": "always",
                "focus": False,
                "panel": "new"
            },
            "runOptions": {
                "runOn": "folderOpen"
            },
            "problemMatcher": []
        }
        
        # Load existing tasks if present and merge
        if tasks_file.exists():
            try:
                with open(tasks_file, 'r') as f:
                    existing_tasks = json.load(f)
                
                # Remove old automation task and add new one
                tasks_list = existing_tasks.get('tasks', [])
                tasks_list = [task for task in tasks_list if task.get('label') != 'Workspace Automation']
                tasks_list.append(automation_task)
                
                tasks_config = {
                    "version": "2.0.0",
                    "tasks": tasks_list
                }
            except json.JSONDecodeError:
                print("WARNING: Invalid tasks.json, creating new one")
                tasks_config = {
                    "version": "2.0.0",
                    "tasks": [automation_task]
                }
        else:
            tasks_config = {
                "version": "2.0.0", 
                "tasks": [automation_task]
            }
        
        with open(tasks_file, 'w') as f:
            json.dump(tasks_config, f, indent=4)
        
        # VS Code settings.json for auto-run
        settings_config = {
            "python.defaultInterpreterPath": sys.executable,
            "files.watcherExclude": {
                "**/.artifacts/**": True,
                "**/node_modules/**": True,
                "**/__pycache__/**": True,
                "**/venv/**": True,
                "**/env/**": True
            },
            "task.allowAutomaticTasks": "on",
            "files.autoSave": "onFocusChange",
            "python.analysis.autoImportCompletions": True
        }
        
        # Load and merge existing settings
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        existing_settings = json.loads(content)
                        existing_settings.update(settings_config)
                        settings_config = existing_settings
            except (json.JSONDecodeError, Exception) as e:
                print(f"WARNING: Could not read existing settings.json: {e}")
        
        with open(settings_file, 'w') as f:
            json.dump(settings_config, f, indent=4)
        
        print("Startup automation configured!")
        print(f"   - Tasks: {tasks_file}")
        print(f"   - Settings: {settings_file}")
        
        return True
        
    except Exception as e:
        print(f"WARNING: Could not setup startup automation: {e}")
        return False

def run_script(script_name: str) -> bool:
    """Run a Python script and return success status"""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"WARNING: Script not found: {script_name}")
        return False
        
    try:
        print(f"Running {script_name}...")
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            if result.stdout.strip():
                print(result.stdout)
            print(f"[OK] {script_name} completed successfully")
            return True
        else:
            print(f"[ERROR] Error in {script_name}:")
            if result.stderr:
                print(result.stderr)
            if result.stdout:
                print("Output:", result.stdout)
            return False
    except Exception as e:
        print(f"[ERROR] Failed to run {script_name}: {e}")
        return False

def main():
    """Run all workspace automation scripts with dependency management"""
    print("Starting Enhanced Workspace Automation (Windows Compatible Version)...")
    print(f"Working directory: {Path.cwd()}")
    print(f"Python executable: {sys.executable}")
    
    # Step 1: Check and install dependencies (with Beautiful Soup fix)
    print("\n=== STEP 1: Dependency Check ===")
    if not check_and_install_dependencies():
        print("Failed to install dependencies. Continuing anyway...")
    
    # Step 2: Setup startup automation (one-time setup)
    print("\n=== STEP 2: Setup Automation ===")
    setup_startup_automation()
    
    # Step 3: Run core automation scripts
    print("\n=== STEP 3: Running Scripts ===")
    scripts = [
        "index_workspace.py",
        "generate_workspace_folders.py", 
        "generate_llm_context.py"
    ]
    
    # Check if artifact_visualizer.py exists before adding it
    artifact_viz_path = Path(__file__).parent / "artifact_visualizer.py"
    if artifact_viz_path.exists():
        scripts.append("artifact_visualizer.py")
    
    success_count = 0
    for script in scripts:
        if run_script(script):
            success_count += 1
        print()  # Add spacing between script runs
    
    print(f"=== RESULTS ===")
    print(f"Completed: {success_count}/{len(scripts)} scripts successful")
    
    if success_count == len(scripts):
        print("Workspace automation completed successfully!")
        print("All systems ready for AI-powered development!")
    else:
        print("Some scripts failed. Check output above.")
        print("Try running individual scripts to debug issues.")

if __name__ == "__main__":
    main()
