"""
Configuration loader and manager for openworld-bench.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration manager for loading and merging YAML configs."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize config manager.
        
        Args:
            config_dir: Path to configs directory. Defaults to package configs.
        """
        if config_dir is None:
            # Default to package configs (go up 2 levels from utils to openworld, then up to root)
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'configs'
            )
        self.config_dir = Path(config_dir)
        
        # Cache loaded configs
        self._cache = {}
    
    def load(self, name: str) -> Dict[str, Any]:
        """
        Load a YAML config file.
        
        Args:
            name: Config name (without .yaml extension)
            
        Returns:
            Parsed config dictionary
        """
        if name in self._cache:
            return self._cache[name]
        
        # Try multiple paths - handle both simple names and paths with slashes
        name_parts = name.replace('/', os.sep)
        paths_to_try = [
            self.config_dir / f"{name_parts}.yaml",
            self.config_dir / f"{name_parts}.yml",
            self.config_dir / f"{name}.yaml",
            self.config_dir / f"{name}.yml",
        ]
        
        for path in paths_to_try:
            if path.exists():
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                self._cache[name] = config
                return config
        
        raise FileNotFoundError(f"Config '{name}' not found in {self.config_dir}")
    
    def get_method_config(self, method_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific method.
        
        Args:
            method_name: Name of the method (e.g., 'dann', 'icarl')
            
        Returns:
            Method-specific config dictionary
        """
        method = method_name.lower()
        
        # Determine method type
        da_methods = ['dann', 'cdan', 'mcd', 'toalign', 'pmtrans']
        dg_methods = ['irm', 'vrex', 'coral', 'swad', 'miro', 'eoa']
        cl_methods = ['icarl', 'der', 'lwf', 'coda_prompt', 'xder', 'memo']
        
        if method in da_methods:
            config = self.load('methods/da')
        elif method in dg_methods:
            config = self.load('methods/dg')
        elif method in cl_methods:
            config = self.load('methods/cl')
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        return config.get(method, {})
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset config dictionary
        """
        datasets_config = self.load('datasets')
        return datasets_config.get(dataset_name.lower(), {})
    
    def get_default(self) -> Dict[str, Any]:
        """Get default configuration."""
        return self.load('default')
    
    def get_experiments(self) -> Dict[str, Any]:
        """Get experiment configurations."""
        return self.load('experiments')
    
    def merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two config dictionaries.
        
        Args:
            base: Base configuration
            override: Override values
            
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge(result[key], value)
            else:
                result[key] = value
        
        return result


# Global config instance
_config = None


def get_config(config_dir: Optional[str] = None) -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = Config(config_dir)
    return _config


def load_method_config(method_name: str) -> Dict[str, Any]:
    """Convenience function to load method config."""
    return get_config().get_method_config(method_name)


def load_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Convenience function to load dataset config."""
    return get_config().get_dataset_config(dataset_name)
