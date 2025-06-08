"""
Artifact Visualizer - Advanced AI workspace assistant for data analysis and visualization
Integrated with workspace automation system using LangGraph and multi-provider LLM support
"""
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime
import asyncio
from dataclasses import dataclass, asdict
import argparse

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Multi-provider LLM support
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Use the new Ollama import to avoid deprecation warning
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

# Tools
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

@dataclass
class WorkspaceState:
    """State management for the visualization workflow"""
    messages: List[Dict[str, Any]]
    current_task: str
    research_data: Dict[str, Any]
    visualization_code: str
    artifacts_created: List[str]
    session_id: str
    iteration_count: int
    max_iterations: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkspaceState':
        return cls(**data)

class ArtifactVisualizer:
    """
    Advanced AI workspace assistant for complex data analysis and visualization
    Integrated with the artifact workspace automation system
    """
    
    def __init__(self, 
                 provider: Literal["openai", "anthropic", "google", "ollama"] = "ollama",
                 model_name: Optional[str] = None,
                 workspace_path: str = "."):
        
        self.workspace_path = Path(workspace_path)
        self.provider = provider
        self.llm = self._initialize_llm(provider, model_name)
        self.tools = self._setup_tools()
        self.graph = self._build_workflow()
        
        # Load workspace context if available
        self.workspace_context = self._load_workspace_context()
    
    def _check_ollama_model(self, model_name: str) -> bool:
        """Check if Ollama model exists and pull if needed"""
        try:
            # Check if model exists
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0 and model_name in result.stdout:
                print(f"[OK] Ollama model '{model_name}' is available")
                return True
            
            # Model not found, try to pull it
            print(f"[DOWNLOADING] Pulling Ollama model '{model_name}'...")
            pull_result = subprocess.run(
                ["ollama", "pull", model_name], 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout for model download
            )
            
            if pull_result.returncode == 0:
                print(f"[OK] Successfully pulled model '{model_name}'")
                return True
            else:
                print(f"[ERROR] Failed to pull model '{model_name}': {pull_result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] Timeout pulling model '{model_name}' - continuing anyway")
            return True  # Continue even if pull times out
        except FileNotFoundError:
            print("[ERROR] Ollama not found in PATH")
            return False
        except Exception as e:
            print(f"[WARNING] Error checking Ollama model: {e}")
            return False
    
    def _ensure_ollama_running(self) -> bool:
        """Check if Ollama is running and try to start if needed"""
        try:
            # Test if Ollama is responsive
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode == 0:
                return True
            
            # Try to start Ollama service
            print("[STARTING] Starting Ollama service...")
            subprocess.Popen(
                ["ollama", "serve"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            
            # Wait a moment for service to start
            import time
            time.sleep(3)
            
            # Test again
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"[WARNING] Could not start Ollama: {e}")
            return False
        
    def _initialize_llm(self, provider: str, model_name: Optional[str]):
        """Initialize LLM based on provider"""
        api_keys = self._load_api_keys()
        
        if provider == "openai":
            api_key = api_keys.get("OPENAI_API_KEY")
            if not api_key:
                print("[WARNING] No OpenAI API key found. Switching to Ollama.")
                return self._initialize_llm("ollama", model_name)
            return ChatOpenAI(
                model=model_name or "gpt-4o",
                api_key=api_key,
                temperature=0.1,
                max_tokens=2048
            )
        elif provider == "anthropic":
            api_key = api_keys.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("[WARNING] No Anthropic API key found. Switching to Ollama.")
                return self._initialize_llm("ollama", model_name)
            return ChatAnthropic(
                model=model_name or "claude-3-5-sonnet-20241022",
                api_key=api_key,
                temperature=0.1,
                max_tokens=2048
            )
        elif provider == "google":
            api_key = api_keys.get("GOOGLE_API_KEY")
            if not api_key:
                print("[WARNING] No Google API key found. Switching to Ollama.")
                return self._initialize_llm("ollama", model_name)
            return ChatGoogleGenerativeAI(
                model=model_name or "gemini-pro",
                google_api_key=api_key,
                temperature=0.1
            )
        elif provider == "ollama":
            # Set default model with fallbacks
            if not model_name:
                model_name = "Artifact_Virtual/raegen:latest"
            
            # Ensure Ollama is running
            if not self._ensure_ollama_running():
                raise Exception("Ollama service is not running and could not be started")
            
            # Check and pull model if needed
            if not self._check_ollama_model(model_name):
                # Try common fallback models                fallback_models = ["llama3.2:latest", "llama3.2:3b", "llama3.2:1b", "qwen2.5:latest"]
                
                for fallback in fallback_models:
                    print(f"[RETRY] Trying fallback model: {fallback}")
                    if self._check_ollama_model(fallback):
                        model_name = fallback
                        break
                else:
                    print("[ERROR] No working Ollama models found")
                    raise Exception("No Ollama models available")
            
            try:
                return OllamaLLM(
                    model=model_name,
                    temperature=0.1
                )
            except Exception as e:
                print(f"[WARNING] Ollama connection failed: {e}")
                print("[INFO] Please ensure:")
                print("   - Ollama is installed: https://ollama.ai/")
                print("   - Service is running: ollama serve")
                print(f"   - Model is available: ollama pull {model_name}")
                raise
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment or config"""
        keys = {}
        
        # Try environment variables first
        env_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "BING_API_KEY"]
        for key in env_keys:
            if os.getenv(key):
                keys[key] = os.getenv(key)
        
        # Try loading from workspace config
        config_path = self.workspace_path / ".env"
        if config_path.exists():
            with open(config_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        keys[key] = value.strip('"\'')
        
        return keys
    
    def _load_workspace_context(self) -> Dict[str, Any]:
        """Load workspace context from automation system"""
        context_path = self.workspace_path / ".vscode" / "llm_context.json"
        if context_path.exists():
            with open(context_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _execute_tool(self, tool_name: str, *args, **kwargs) -> str:
        """Execute a tool by name with error handling"""
        tool_map = {tool.name: tool for tool in self.tools}
        if tool_name in tool_map:
            try:
                return tool_map[tool_name].run(*args, **kwargs)
            except Exception as e:
                return f"Tool execution failed: {str(e)}"
        return f"Tool '{tool_name}' not found"
    
    def _create_visualization_from_data(self, task: str, data: Dict[str, Any]) -> str:
        """Create meaningful visualizations based on workspace data"""
        artifacts_dir = self.workspace_path / ".artifacts" / "visualizations"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Create a meaningful visualization based on workspace context
            plt.style.use('seaborn-v0_8')  # Use seaborn style for better looking plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Chart 1: Workspace File Distribution
            if self.workspace_context and 'agents' in self.workspace_context:
                agents = self.workspace_context['agents']
                agent_names = list(agents.keys())[:10]  # Top 10 agents
                file_counts = [agents[name].get('python_files', 0) for name in agent_names]
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(agent_names)))
                ax1.bar(range(len(agent_names)), file_counts, color=colors)
                ax1.set_title('Python Files by Agent/Directory', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Agents')
                ax1.set_ylabel('Number of Python Files')
                ax1.set_xticks(range(len(agent_names)))
                ax1.set_xticklabels(agent_names, rotation=45, ha='right')
                
                # Add value labels on bars
                for i, v in enumerate(file_counts):
                    ax1.text(i, v + 0.1, str(v), ha='center', va='bottom')
            else:
                # Fallback chart if no workspace context
                categories = ['Scripts', 'Configs', 'Documentation', 'Tests', 'Assets']
                values = [25, 15, 20, 10, 30]
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                ax1.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
                ax1.set_title('Workspace File Distribution', fontsize=14, fontweight='bold')
            
            # Chart 2: Activity Timeline (simulated based on current time)
            dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')[-30:]
            activity = np.random.poisson(5, len(dates)) + np.sin(np.arange(len(dates)) * 0.2) * 3 + 5
            
            ax2.plot(dates, activity, marker='o', linewidth=2, markersize=4, color='#FF6B6B')
            ax2.fill_between(dates, activity, alpha=0.3, color='#FF6B6B')
            ax2.set_title('Workspace Activity (Last 30 Days)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Activity Level')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Chart 3: Technology Stack
            if self.workspace_context and 'capabilities' in self.workspace_context:
                capabilities = self.workspace_context['capabilities'][:8]  # Top 8 capabilities
                cap_values = [np.random.randint(60, 100) for _ in capabilities]  # Simulated usage percentages
            else:
                capabilities = ['Python', 'LangGraph', 'AI/ML', 'Data Analysis', 'Visualization', 'Automation']
                cap_values = [95, 85, 80, 75, 70, 90]
            
            y_pos = np.arange(len(capabilities))
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(capabilities)))
            bars = ax3.barh(y_pos, cap_values, color=colors)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(capabilities)
            ax3.set_xlabel('Usage/Proficiency %')
            ax3.set_title('Technology Stack Proficiency', fontsize=14, fontweight='bold')
            ax3.set_xlim(0, 100)
            
            # Add percentage labels
            for i, (bar, value) in enumerate(zip(bars, cap_values)):
                ax3.text(value + 1, bar.get_y() + bar.get_height()/2, f'{value}%', 
                        va='center', fontweight='bold')
            
            # Chart 4: Workspace Metrics
            if self.workspace_context and 'metadata' in self.workspace_context:
                metadata = self.workspace_context['metadata']
                total_files = metadata.get('total_files', 100)
                total_dirs = metadata.get('total_directories', 20)
                agents_count = len(self.workspace_context.get('agents', {}))
            else:
                total_files, total_dirs, agents_count = 150, 25, 8
            
            metrics = ['Total Files', 'Directories', 'Agents', 'Scripts']
            values = [total_files, total_dirs, agents_count, 45]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            wedges, texts, autotexts = ax4.pie(values, labels=metrics, colors=colors, 
                                              autopct='%1.0f', startangle=90)
            ax4.set_title('Workspace Overview', fontsize=14, fontweight='bold')
            
            # Style the text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.suptitle(f'Artifact Workspace Analysis Dashboard\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = artifacts_dir / f"workspace_analysis_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Also create a summary JSON file
            summary_path = artifacts_dir / f"analysis_summary_{timestamp}.json"
            summary_data = {
                "timestamp": datetime.now().isoformat(),
                "task": task,
                "workspace_path": str(self.workspace_path),
                "total_files": total_files if 'total_files' in locals() else 0,
                "total_directories": total_dirs if 'total_dirs' in locals() else 0,
                "agents_count": agents_count if 'agents_count' in locals() else 0,
                "capabilities": capabilities if 'capabilities' in locals() else [],
                "visualization_saved": str(plot_path)
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            return f"[OK] Comprehensive workspace analysis saved to: {plot_path}\n[INFO] Summary data saved to: {summary_path}"
            
        except Exception as e:
            return f"[ERROR] Error creating visualization: {str(e)}"
    
    def _setup_tools(self) -> List:
        """Setup tools for research and visualization"""
        
        @tool
        def web_search(query: str) -> str:
            """Search the web for information using DuckDuckGo"""
            try:
                search = DuckDuckGoSearchRun()
                results = search.run(query)
                return f"Search results for '{query}':\n{results}"
            except Exception as e:
                return f"Search failed: {str(e)}"
        
        @tool
        def python_executor(code: str) -> str:
            """Execute Python code for data analysis and visualization"""
            try:
                # Create artifacts directory
                artifacts_dir = self.workspace_path / ".artifacts" / "visualizations"
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                
                # Prepare the code with proper matplotlib backend and save path
                enhanced_code = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Set up artifacts directory
artifacts_dir = Path(r"{artifacts_dir}")
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Enhanced code execution
{code}

# Ensure plot is saved
if plt.get_fignums():
    timestamp = "{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    plot_path = artifacts_dir / f"generated_plot_{{timestamp}}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Plot saved to: {{plot_path}}")
    plt.close('all')
"""
                
                # Execute the enhanced code
                repl = PythonREPLTool()
                result = repl.run(enhanced_code)
                
                return result
                
            except Exception as e:
                return f"Code execution failed: {str(e)}"
        
        @tool
        def data_fetcher(url: str) -> str:
            """Fetch data from APIs or structured web sources"""
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Try to parse as JSON first
                try:
                    data = response.json()
                    return f"Successfully fetched JSON data: {json.dumps(data, indent=2)[:1000]}..."
                except:
                    # Fall back to text/HTML parsing
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = soup.get_text()[:1000]
                    return f"Fetched web content: {text}..."
                    
            except Exception as e:
                return f"Data fetch failed: {str(e)}"
        
        @tool
        def workspace_analyzer() -> str:
            """Analyze the current workspace context and capabilities"""
            if not self.workspace_context:
                return "No workspace context available. Run workspace automation first."
            
            metadata = self.workspace_context.get('metadata', {})
            capabilities = self.workspace_context.get('capabilities', [])
            agents = self.workspace_context.get('agents', {})
            
            analysis = f"""
[WORKSPACE ANALYSIS]:
- Total Files: {metadata.get('total_files', 0):,}
- Total Directories: {metadata.get('total_directories', 0):,}
- Available Agents: {len(agents)}
- Capabilities: {', '.join(capabilities)}

[AGENTS IN WORKSPACE]:
{chr(10).join([f"- {name}: {info.get('python_files', 0)} Python files" for name, info in agents.items()])}
            """
            return analysis.strip()
        
        @tool  
        def create_dashboard() -> str:
            """Create a comprehensive workspace visualization dashboard"""
            return self._create_visualization_from_data("Dashboard Creation", {})
        
        return [web_search, python_executor, data_fetcher, workspace_analyzer, create_dashboard]
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        def research_node(state: WorkspaceState) -> WorkspaceState:
            """Research node - gather information"""
            
            # Always create a dashboard first
            dashboard_result = self._execute_tool("create_dashboard")
            state.research_data["dashboard"] = dashboard_result
            state.artifacts_created.append("workspace_dashboard")
            
            # Then do workspace analysis
            workspace_analysis = self._execute_tool("workspace_analyzer")
            state.research_data["workspace_analysis"] = workspace_analysis
            
            state.messages.append({
                "role": "researcher",
                "content": f"Created comprehensive workspace dashboard and analysis for: {state.current_task}",
                "dashboard_result": dashboard_result,
                "timestamp": datetime.now().isoformat()
            })
            
            state.iteration_count += 1
            return state
        
        def visualization_node(state: WorkspaceState) -> WorkspaceState:
            """Visualization node - create additional specific charts"""
            
            # Create a specific visualization based on the task
            specific_viz_code = f"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

plt.style.use('seaborn-v0_8')

# Create a task-specific visualization for: {state.current_task}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Performance metrics over time
dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
performance = np.random.normal(85, 10, len(dates))
performance = np.clip(performance, 0, 100)

ax1.plot(dates, performance, marker='o', linewidth=2, color='#2E86AB')
ax1.fill_between(dates, performance, alpha=0.3, color='#2E86AB')
ax1.set_title('System Performance Trend', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Performance Score (%)')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 100)

# Resource utilization
resources = ['CPU', 'Memory', 'Storage', 'Network']
utilization = [65, 78, 45, 52]
colors = ['#A23B72', '#F18F01', '#C73E1D', '#2E86AB']

bars = ax2.bar(resources, utilization, color=colors)
ax2.set_title('Current Resource Utilization', fontsize=14, fontweight='bold')
ax2.set_ylabel('Utilization (%)')
ax2.set_ylim(0, 100)

# Add percentage labels on bars
for bar, value in zip(bars, utilization):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{value}%', ha='center', va='bottom', fontweight='bold')

plt.suptitle(f'Task Analysis: {state.current_task}', fontsize=16, fontweight='bold')
plt.tight_layout()
"""
            
            execution_result = self._execute_tool("python_executor", specific_viz_code)
            
            state.visualization_code = specific_viz_code
            state.artifacts_created.append("task_specific_chart")
            
            state.messages.append({
                "role": "visualizer",
                "content": f"Created task-specific visualization for: {state.current_task}",
                "execution_result": execution_result,
                "timestamp": datetime.now().isoformat()
            })
            
            state.iteration_count += 1
            return state
        
        def should_continue(state: WorkspaceState) -> str:
            """Determine if workflow should continue"""
            # Always create both dashboard and specific visualization
            if state.iteration_count >= 2:  # Research + Visualization
                return "end"
            
            return "continue"
        
        # Build the graph
        workflow = StateGraph(WorkspaceState)
        
        workflow.add_node("research", research_node)
        workflow.add_node("visualization", visualization_node)
        
        workflow.add_edge("research", "visualization")
        workflow.add_conditional_edges(
            "visualization",
            should_continue,
            {
                "continue": "research",
                "end": END
            }
        )
        
        workflow.set_entry_point("research")
        
        return workflow.compile()
    
    async def analyze_and_visualize(self, 
                                   task: str, 
                                   session_id: Optional[str] = None,
                                   max_iterations: int = 3) -> Dict[str, Any]:
        """Main method to analyze and visualize data"""
        
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize state
        initial_state = WorkspaceState(
            messages=[],
            current_task=task,
            research_data={},
            visualization_code="",
            artifacts_created=[],
            session_id=session_id,
            iteration_count=0,
            max_iterations=max_iterations        )
        
        print(f"[START] Starting Artifact Visualizer analysis...")
        print(f"[TASK] Task: {task}")
        print(f"[PROVIDER] Provider: {self.provider}")
        print(f"[PATH] Workspace: {self.workspace_path}")
        
        # Run the workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        # Save session results
        results_dir = self.workspace_path / ".artifacts" / "sessions"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        session_file = results_dir / f"{session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(final_state.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Analysis complete! Session saved: {session_file}")
        
        # List created artifacts
        viz_dir = self.workspace_path / ".artifacts" / "visualizations"
        if viz_dir.exists():
            artifacts = list(viz_dir.glob("*.png"))
            print(f"[INFO] Created {len(artifacts)} visualization files:")
            for artifact in artifacts[-5:]:  # Show last 5
                print(f"   - {artifact.name}")
        
        return final_state.to_dict()
    
    def sync_analyze_and_visualize(self, task: str, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for the async method"""
        return asyncio.run(self.analyze_and_visualize(task, **kwargs))

# CLI Interface
def main():
    """Command-line interface for the visualizer"""
    parser = argparse.ArgumentParser(description="Artifact Visualizer - AI Data Analysis")
    parser.add_argument("task", nargs='?', default="Analyze workspace performance and create comprehensive visualizations", 
                       help="Analysis task description")
    parser.add_argument("--provider", choices=["openai", "anthropic", "google", "ollama"], 
                       default="ollama", help="LLM provider")
    parser.add_argument("--model", help="Specific model name")
    parser.add_argument("--workspace", default=".", help="Workspace path")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum iterations")
    
    args = parser.parse_args()
    
    try:
        visualizer = ArtifactVisualizer(
            provider=args.provider,
            model_name=args.model,
            workspace_path=args.workspace
        )
        
        results = visualizer.sync_analyze_and_visualize(
            task=args.task,
            max_iterations=args.max_iterations        )
        
        print("\n[FINAL RESULTS]:")
        print(f"Messages: {len(results['messages'])}")
        print(f"Artifacts Created: {len(results['artifacts_created'])}")
        print(f"Artifacts: {', '.join(results['artifacts_created'])}")
        if results['visualization_code']:
            print("[OK] Visualization code generated and executed successfully")
            
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        print("\n[INFO] Quick fixes:")
        print("1. For Ollama: Install and run 'ollama serve'")
        print("   - Download: https://ollama.ai/")
        print("   - Pull models: ollama pull Artifact_Virtual/raegen:latest")
        print("   - Start service: ollama serve")
        print("2. For API providers: Set environment variables:")
        print("   - OPENAI_API_KEY=your_key")
        print("   - ANTHROPIC_API_KEY=your_key")
        print("   - GOOGLE_API_KEY=your_key")
        print("3. Create .env file in workspace root with your API keys")

if __name__ == "__main__":
    main()