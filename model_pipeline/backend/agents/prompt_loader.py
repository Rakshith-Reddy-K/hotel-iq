"""
Prompt Loader
=============

Centralized prompt management system.
Loads all prompts from prompts.yaml for easy modification without code changes.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class PromptLoader:
    """
    Loads and provides access to prompts from prompts.yaml.
    
    Usage:
        prompts = PromptLoader()
        prompt_text = prompts.get("query_agent.extract_hotel_name")
        formatted = prompts.format("query_agent.extract_hotel_name", user_message="Show me hotels")
    """
    
    def __init__(self, prompts_file: str = None):
        """
        Initialize the prompt loader.
        
        Args:
            prompts_file: Path to prompts YAML file. Defaults to ../prompts.yaml
        """
        if prompts_file is None:
            # Default to prompts.yaml in backend directory
            backend_dir = Path(__file__).parent.parent
            prompts_file = backend_dir / "prompts.yaml"
        
        self.prompts_file = Path(prompts_file)
        self._prompts: Dict[str, Any] = {}
        self._load_prompts()
    
    def _load_prompts(self):
        """Load prompts from YAML file."""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                self._prompts = yaml.safe_load(f)
            print(f"✅ Prompts loaded from {self.prompts_file}")
        except FileNotFoundError:
            print(f"⚠️ Prompts file not found: {self.prompts_file}")
            self._prompts = {}
        except yaml.YAMLError as e:
            print(f"⚠️ Error parsing prompts YAML: {e}")
            self._prompts = {}
    
    def get(self, path: str) -> str:
        """
        Get a prompt by its path.
        
        Args:
            path: Dot-separated path to prompt (e.g., "query_agent.extract_hotel_name")
        
        Returns:
            The prompt text, or empty string if not found
        """
        keys = path.split('.')
        value = self._prompts
        
        try:
            for key in keys:
                value = value[key]
            return value if isinstance(value, str) else ""
        except (KeyError, TypeError):
            print(f"⚠️ Prompt not found: {path}")
            return ""
    
    def format(self, path: str, **kwargs) -> str:
        """
        Get a prompt and format it with provided variables.
        
        Args:
            path: Dot-separated path to prompt
            **kwargs: Variables to format into the prompt
        
        Returns:
            Formatted prompt text
        """
        prompt = self.get(path)
        try:
            return prompt.format(**kwargs)
        except KeyError as e:
            print(f"⚠️ Missing variable in prompt {path}: {e}")
            return prompt
    
    def reload(self):
        """Reload prompts from file (useful for development)."""
        self._load_prompts()


# Global instance for easy access across modules
_prompt_loader = None

def get_prompts() -> PromptLoader:
    """
    Get the global prompt loader instance.
    Creates one if it doesn't exist.
    """
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    return _prompt_loader

