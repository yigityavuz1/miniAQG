"""
Prompt Loader for managing versioned markdown prompts.
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path


class PromptLoader:
    """Load and manage versioned markdown prompts for AQG agents."""
    
    def __init__(self, config_path: str = "config/prompt_versions.yaml", prompts_dir: str = "prompts"):
        self.config_path = config_path
        self.prompts_dir = prompts_dir
        self.config = self._load_config()
        
        # Validate prompt files exist
        self._validate_prompt_files()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the prompt configuration YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Configuration file not found: {self.config_path}")
            return {"workflow": {}}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration: {e}")
            return {"workflow": {}}
    
    def _validate_prompt_files(self) -> None:
        """Validate that all configured prompt files exist."""
        # Get agent names from workflow configuration
        workflow = self.config.get('workflow', {})
        
        # List of known agents
        agents = ['content_splitter', 'summarizer', 'question_generator', 'judge', 'fixer']
        
        for agent_name in agents:
            agent_config = workflow.get(agent_name, {})
            version = agent_config.get('prompt_version', 'v1')
            
            # Check if prompt file exists
            prompt_path = os.path.join(self.prompts_dir, agent_name, f"{version}.md")
            if not os.path.exists(prompt_path):
                print(f"Warning: Prompt file not found: {prompt_path}")
    
    def load_prompt(self, agent_name: str, version: Optional[str] = None) -> str:
        """
        Load a prompt template for a specific agent and version.
        
        Args:
            agent_name: Name of the agent (content_splitter, summarizer, etc.)
            version: Specific version to load (defaults to configured version)
            
        Returns:
            The prompt template as a string
            
        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        # Use provided version or get from config
        if version is None:
            agent_config = self.config.get('workflow', {}).get(agent_name, {})
            version = agent_config.get('prompt_version', 'v1')
        
        # Construct file path
        prompt_path = os.path.join(self.prompts_dir, agent_name, f"{version}.md")
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        except Exception as e:
            raise Exception(f"Error loading prompt from {prompt_path}: {e}")
    
    def get_available_versions(self, agent_name: str) -> List[str]:
        """
        Get all available versions for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of available version strings
        """
        agent_dir = os.path.join(self.prompts_dir, agent_name)
        
        if not os.path.exists(agent_dir):
            return []
        
        versions = []
        for file in os.listdir(agent_dir):
            if file.endswith('.md'):
                version = file[:-3]  # Remove .md extension
                versions.append(version)
        
        return sorted(versions)
    
    def get_prompt_metadata(self, agent_name: str, version: str = None) -> Dict[str, Any]:
        """
        Extract metadata from a prompt file (if any).
        Looks for YAML frontmatter at the top of markdown files.
        
        Args:
            agent_name: Name of the agent
            version: Version of the prompt
            
        Returns:
            Dictionary containing metadata
        """
        if version is None:
            agent_config = self.config.get('workflow', {}).get(agent_name, {})
            version = agent_config.get('prompt_version', 'v1')
        
        prompt_path = os.path.join(self.prompts_dir, agent_name, f"{version}.md")
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for YAML frontmatter
            if content.startswith('---\n'):
                parts = content.split('---\n', 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    try:
                        metadata = yaml.safe_load(frontmatter)
                        return metadata if isinstance(metadata, dict) else {}
                    except yaml.YAMLError:
                        pass
            
            return {}
            
        except FileNotFoundError:
            return {}
    
    def list_agents(self) -> List[str]:
        """
        List all available agents.
        
        Returns:
            List of agent names
        """
        workflow = self.config.get('workflow', {})
        agents = ['content_splitter', 'summarizer', 'question_generator', 'judge', 'fixer']
        
        # Filter to only return agents that have directories
        available_agents = []
        for agent in agents:
            agent_dir = os.path.join(self.prompts_dir, agent)
            if os.path.exists(agent_dir):
                available_agents.append(agent)
        
        return available_agents
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Configuration dictionary for the agent
        """
        return self.config.get('workflow', {}).get(agent_name, {})
    
    def reload_config(self) -> None:
        """Reload the configuration from file."""
        self.config = self._load_config()
        self._validate_prompt_files()
    
    def get_prompt_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all prompts and their status.
        
        Returns:
            Summary dictionary with agent info and file status
        """
        summary = {
            "total_agents": 0,
            "agents": {},
            "missing_files": [],
            "config_file": self.config_path,
            "prompts_directory": self.prompts_dir
        }
        
        agents = self.list_agents()
        summary["total_agents"] = len(agents)
        
        for agent_name in agents:
            agent_config = self.get_agent_config(agent_name)
            versions = self.get_available_versions(agent_name)
            configured_version = agent_config.get('prompt_version', 'v1')
            
            # Check if configured version exists
            prompt_path = os.path.join(self.prompts_dir, agent_name, f"{configured_version}.md")
            file_exists = os.path.exists(prompt_path)
            
            if not file_exists:
                summary["missing_files"].append(prompt_path)
            
            summary["agents"][agent_name] = {
                "configured_version": configured_version,
                "available_versions": versions,
                "file_exists": file_exists,
                "model_provider": agent_config.get('model_provider', 'openai'),
                "model": agent_config.get('model', 'gpt-4o')
            }
        
        return summary


# Global instance for easy importing
prompt_loader = PromptLoader() 