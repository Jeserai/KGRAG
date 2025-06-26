#!/usr/bin/env python3
"""
Environment verification script for Knowledge Graph RAG system.
Checks all required components and automatically installs missing dependencies.
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentVerifier:
    """Verifies the Knowledge Graph RAG environment setup and auto-installs missing components."""
    
    def __init__(self, auto_install: bool = True):
        self.results = {}
        self.errors = []
        self.warnings = []
        self.auto_install = auto_install
        self.installed_packages = []
    
    def print_header(self, title: str):
        """Print a formatted header."""
        print(f"\n{'='*50}")
        print(f" {title}")
        print(f"{'='*50}")
    
    def print_result(self, component: str, status: str, details: str = ""):
        """Print a formatted result."""
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} {component}: {status}")
        if details:
            print(f"   {details}")
        self.results[component] = status
    
    def install_package(self, package_name: str, pip_name: Optional[str] = None) -> bool:
        """Install a Python package using pip."""
        if not self.auto_install:
            return False
        
        pip_package = pip_name or package_name
        print(f"   🔧 Installing {pip_package}...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", pip_package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            self.installed_packages.append(pip_package)
            print(f"   ✅ Successfully installed {pip_package}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {pip_package}: {e}")
            return False
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        self.print_header("Python Environment")
        
        version = sys.version_info
        self.print_result(
            "Python Version",
            "PASS" if version >= (3, 9) else "FAIL",
            f"{version.major}.{version.minor}.{version.micro}"
        )
        
        return version >= (3, 9)
    
    def check_pytorch(self) -> bool:
        """Check PyTorch installation and CUDA availability."""
        self.print_header("PyTorch & CUDA")
        
        try:
            import torch
            self.print_result("PyTorch", "PASS", f"Version: {torch.__version__}")
            
            cuda_available = torch.cuda.is_available()
            self.print_result("CUDA Available", "PASS" if cuda_available else "WARN", 
                            f"{cuda_available}")
            
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                self.print_result("GPU Count", "PASS", f"{gpu_count} GPUs")
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                    self.print_result(f"GPU {i}", "PASS", f"{gpu_name} ({gpu_memory:.1f}GB)")
            
            return True
            
        except ImportError as e:
            self.print_result("PyTorch", "FAIL", f"Import error: {e}")
            if self.install_package("torch"):
                # Try importing again after installation
                try:
                    import torch
                    self.print_result("PyTorch", "PASS", f"Version: {torch.__version__} (installed)")
                    return True
                except ImportError:
                    pass
            return False
    
    def check_ml_libraries(self) -> bool:
        """Check ML library installations."""
        self.print_header("Machine Learning Libraries")
        
        libraries = [
            ("transformers", "Transformers", "transformers"),
            ("sentence_transformers", "Sentence Transformers", "sentence-transformers"),
            ("accelerate", "Accelerate", "accelerate"),
            ("bitsandbytes", "BitsAndBytes", "bitsandbytes"),
            ("peft", "PEFT", "peft"),
            ("vllm", "vLLM", "vllm"),
            ("optimum", "Optimum", "optimum")
        ]
        
        all_passed = True
        for module_name, display_name, pip_name in libraries:
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, '__version__', 'Unknown')
                self.print_result(display_name, "PASS", f"Version: {version}")
            except ImportError as e:
                self.print_result(display_name, "FAIL", f"Import error: {e}")
                if self.install_package(module_name, pip_name):
                    # Try importing again after installation
                    try:
                        module = importlib.import_module(module_name)
                        version = getattr(module, '__version__', 'Unknown')
                        self.print_result(display_name, "PASS", f"Version: {version} (installed)")
                    except ImportError:
                        all_passed = False
                else:
                    all_passed = False
        
        return all_passed
    
    def check_graph_database(self) -> bool:
        """Check Neo4j database connectivity."""
        self.print_header("Graph Database (Neo4j)")
        
        try:
            import neo4j
            self.print_result("Neo4j Python Driver", "PASS", f"Version: {neo4j.__version__}")
            
            # Try to connect to Neo4j
            try:
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(
                    "bolt://localhost:7687",
                    auth=("neo4j", "password")
                )
                
                with driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    test_value = result.single()["test"]
                
                driver.close()
                
                if test_value == 1:
                    self.print_result("Neo4j Connection", "PASS", "Successfully connected")
                    return True
                else:
                    self.print_result("Neo4j Connection", "FAIL", "Unexpected response")
                    return False
                    
            except Exception as e:
                self.print_result("Neo4j Connection", "FAIL", f"Connection failed: {e}")
                self.warnings.append("Neo4j database is not running or not accessible")
                return False
                
        except ImportError as e:
            self.print_result("Neo4j Python Driver", "FAIL", f"Import error: {e}")
            if self.install_package("neo4j"):
                # Try importing again after installation
                try:
                    import neo4j
                    self.print_result("Neo4j Python Driver", "PASS", f"Version: {neo4j.__version__} (installed)")
                    return True
                except ImportError:
                    pass
            return False
    
    def check_vector_libraries(self) -> bool:
        """Check vector operation libraries."""
        self.print_header("Vector Operations")
        
        try:
            import faiss
            self.print_result("FAISS", "PASS", f"Version: {faiss.__version__}")
        except ImportError as e:
            self.print_result("FAISS", "FAIL", f"Import error: {e}")
            if self.install_package("faiss-cpu"):
                try:
                    import faiss
                    self.print_result("FAISS", "PASS", f"Version: {faiss.__version__} (installed)")
                except ImportError:
                    return False
            else:
                return False
        
        try:
            import numpy as np
            self.print_result("NumPy", "PASS", f"Version: {np.__version__}")
        except ImportError as e:
            self.print_result("NumPy", "FAIL", f"Import error: {e}")
            if self.install_package("numpy"):
                try:
                    import numpy as np
                    self.print_result("NumPy", "PASS", f"Version: {np.__version__} (installed)")
                except ImportError:
                    return False
            else:
                return False
        
        return True
    
    def check_langchain(self) -> bool:
        """Check LangChain installation."""
        self.print_header("LangChain")
        
        try:
            import langchain
            self.print_result("LangChain", "PASS", f"Version: {langchain.__version__}")
            
            # Check LangChain components
            try:
                from langchain_community import embeddings
                self.print_result("LangChain Community", "PASS", "Available")
            except ImportError:
                self.print_result("LangChain Community", "WARN", "Not available")
                if self.install_package("langchain-community"):
                    try:
                        from langchain_community import embeddings
                        self.print_result("LangChain Community", "PASS", "Available (installed)")
                    except ImportError:
                        pass
            
            return True
            
        except ImportError as e:
            self.print_result("LangChain", "FAIL", f"Import error: {e}")
            if self.install_package("langchain"):
                try:
                    import langchain
                    self.print_result("LangChain", "PASS", f"Version: {langchain.__version__} (installed)")
                    return True
                except ImportError:
                    pass
            return False
    
    def check_additional_libraries(self) -> bool:
        """Check additional utility libraries."""
        self.print_header("Additional Libraries")
        
        libraries = [
            ("psutil", "psutil", "psutil"),
            ("pyyaml", "PyYAML", "pyyaml"),
            ("python-dotenv", "python-dotenv", "python-dotenv"),
            ("typer", "Typer", "typer"),
            ("rich", "Rich", "rich"),
            ("scikit-learn", "scikit-learn", "scikit-learn"),
            ("pandas", "pandas", "pandas"),
            ("tqdm", "tqdm", "tqdm")
        ]
        
        all_passed = True
        for module_name, display_name, pip_name in libraries:
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, '__version__', 'Unknown')
                self.print_result(display_name, "PASS", f"Version: {version}")
            except ImportError as e:
                self.print_result(display_name, "FAIL", f"Import error: {e}")
                if self.install_package(module_name, pip_name):
                    try:
                        module = importlib.import_module(module_name)
                        version = getattr(module, '__version__', 'Unknown')
                        self.print_result(display_name, "PASS", f"Version: {version} (installed)")
                    except ImportError:
                        all_passed = False
                else:
                    all_passed = False
        
        return all_passed
    
    def check_model_cache(self) -> bool:
        """Check HuggingFace model cache configuration."""
        self.print_header("HuggingFace Cache Configuration")
        
        # Check environment variables
        hf_vars = {
            'HF_HOME': os.environ.get('HF_HOME'),
            'HF_HUB_CACHE': os.environ.get('HF_HUB_CACHE'),
            'HF_DATASETS_CACHE': os.environ.get('HF_DATASETS_CACHE'),
            'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE')
        }
        
        all_configured = True
        for var, value in hf_vars.items():
            if value:
                self.print_result(f"Environment: {var}", "PASS", f"{value}")
            else:
                self.print_result(f"Environment: {var}", "WARN", "Not set")
                all_configured = False
        
        # Check if cache directories exist
        cache_dirs = [
            ('HF_HOME', os.environ.get('HF_HOME')),
            ('HF_HUB_CACHE', os.environ.get('HF_HUB_CACHE')),
            ('HF_DATASETS_CACHE', os.environ.get('HF_DATASETS_CACHE'))
        ]
        
        def get_directory_size(path):
            """Get directory size in human readable format."""
            try:
                total_size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
                if total_size > 1024**3:
                    return f"{total_size / (1024**3):.1f}GB"
                elif total_size > 1024**2:
                    return f"{total_size / (1024**2):.1f}MB"
                else:
                    return f"{total_size / 1024:.1f}KB"
            except:
                return "Unknown"
        
        for name, path in cache_dirs:
            if path and Path(path).exists():
                size = get_directory_size(path)
                self.print_result(f"Directory: {name}", "PASS", f"{path} ({size})")
                
                # List some contents for hub cache
                if name == 'HF_HUB_CACHE':
                    try:
                        models = list(Path(path).glob("*/*"))
                        if models:
                            self.print_result(f"  Cached Models", "PASS", f"{len(models)} models found")
                            for model in models[:3]:  # Show first 3
                                self.print_result(f"    Model", "INFO", str(model.name))
                        else:
                            self.print_result(f"  Cached Models", "WARN", "No models found")
                    except Exception as e:
                        self.print_result(f"  Cached Models", "WARN", f"Error listing: {e}")
                        
            elif path:
                self.print_result(f"Directory: {name}", "FAIL", f"Missing: {path}")
                all_configured = False
            else:
                self.print_result(f"Directory: {name}", "WARN", "Path not configured")
                all_configured = False
        
        # Check offline mode
        if os.environ.get('HF_HUB_OFFLINE') == '1':
            self.print_result("Offline Mode", "PASS", "HF_HUB_OFFLINE=1 (offline mode enabled)")
        else:
            self.print_result("Offline Mode", "WARN", "HF_HUB_OFFLINE not set to 1 (online mode)")
        
        # Fallback to default cache location if custom paths not configured
        if not all_configured:
            default_cache = Path.home() / ".cache" / "huggingface"
            if default_cache.exists():
                cache_size = sum(f.stat().st_size for f in default_cache.rglob('*') if f.is_file())
                cache_size_gb = cache_size / (1024**3)
                self.print_result("Default Cache", "PASS", f"Size: {cache_size_gb:.1f}GB")
                
                # List some cached models
                hub_dir = default_cache / "hub"
                if hub_dir.exists():
                    models = list(hub_dir.glob("*/*"))
                    if models:
                        self.print_result("  Default Models", "PASS", f"{len(models)} models found")
                        for model in models[:3]:  # Show first 3
                            self.print_result("    Model", "INFO", str(model.name))
                    else:
                        self.print_result("  Default Models", "WARN", "No models found")
                else:
                    self.print_result("  Default Models", "WARN", "Hub directory not found")
            else:
                self.print_result("Default Cache", "WARN", "Default cache directory not found")
        
        return True
    
    def check_project_structure(self) -> bool:
        """Check project file structure."""
        self.print_header("Project Structure")
        
        required_files = [
            "requirements.txt",
            "config/model_configs.yaml",
            "src/models/model_config.py",
            "src/models/local_llm.py",
            "src/models/embedding_model.py",
            "src/data/document_processor.py",
            "src/kg/neo4j_connector.py",
            "src/kg/entity_extractor.py",
            "src/kg/graph_builder.py"
        ]
        
        all_present = True
        for file_path in required_files:
            if Path(file_path).exists():
                self.print_result(f"File: {file_path}", "PASS", "Found")
            else:
                self.print_result(f"File: {file_path}", "FAIL", "Missing")
                all_present = False
        
        return all_present
    
    def check_system_resources(self) -> bool:
        """Check system resources."""
        self.print_header("System Resources")
        
        try:
            import psutil
            
            # Memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            self.print_result("Total RAM", "PASS", f"{memory_gb:.1f}GB")
            
            if memory_gb < 8:
                self.print_result("RAM Sufficiency", "WARN", "Less than 8GB RAM")
            else:
                self.print_result("RAM Sufficiency", "PASS", "Sufficient RAM")
            
            # Disk space
            disk = psutil.disk_usage('.')
            disk_gb = disk.free / (1024**3)
            self.print_result("Available Disk", "PASS", f"{disk_gb:.1f}GB")
            
            # CPU cores
            cpu_count = psutil.cpu_count()
            self.print_result("CPU Cores", "PASS", f"{cpu_count} cores")
            
            return True
            
        except ImportError:
            self.print_result("System Resources", "WARN", "psutil not available")
            if self.install_package("psutil"):
                try:
                    import psutil
                    self.print_result("System Resources", "PASS", "psutil installed")
                    return self.check_system_resources()  # Recursive call to check again
                except ImportError:
                    pass
            return False
    
    def run_model_tests(self) -> bool:
        """Run basic model loading tests."""
        self.print_header("Model Loading Tests")
        
        try:
            # Test configuration loading
            from src.models.model_config import config_manager
            
            # Test LLM configuration
            try:
                llm_config = config_manager.get_default_llm_config()
                self.print_result("LLM Config", "PASS", f"Model: {llm_config.name}")
            except Exception as e:
                self.print_result("LLM Config", "FAIL", f"Error: {e}")
                return False
            
            # Test embedding configuration
            try:
                embedding_config = config_manager.get_default_embedding_config()
                self.print_result("Embedding Config", "PASS", f"Model: {embedding_config.name}")
            except Exception as e:
                self.print_result("Embedding Config", "FAIL", f"Error: {e}")
                return False
            
            return True
            
        except ImportError as e:
            self.print_result("Model Tests", "FAIL", f"Import error: {e}")
            return False
    
    def create_directories(self) -> bool:
        """Create necessary directories if they don't exist."""
        self.print_header("Creating Directories")
        
        directories = ["data", "cache", "output", "logs"]
        all_created = True
        
        for directory in directories:
            dir_path = Path(directory)
            if dir_path.exists():
                self.print_result(f"Directory: {directory}", "PASS", "Already exists")
            else:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.print_result(f"Directory: {directory}", "PASS", "Created")
                except Exception as e:
                    self.print_result(f"Directory: {directory}", "FAIL", f"Failed to create: {e}")
                    all_created = False
        
        return all_created
    
    def generate_report(self) -> None:
        """Generate a summary report."""
        self.print_header("Environment Verification Report")
        
        total_checks = len(self.results)
        passed_checks = sum(1 for status in self.results.values() if status == "PASS")
        failed_checks = sum(1 for status in self.results.values() if status == "FAIL")
        warnings = sum(1 for status in self.results.values() if status == "WARN")
        
        print(f"\nSummary:")
        print(f"  Total Checks: {total_checks}")
        print(f"  Passed: {passed_checks}")
        print(f"  Failed: {failed_checks}")
        print(f"  Warnings: {warnings}")
        
        if self.installed_packages:
            print(f"\n📦 Installed Packages:")
            for package in self.installed_packages:
                print(f"  - {package}")
        
        if failed_checks == 0:
            print(f"\n🎉 Environment is ready for Knowledge Graph RAG!")
        else:
            print(f"\n⚠️  Environment has issues that need to be resolved.")
        
        if self.warnings:
            print(f"\nWarnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if failed_checks > 0:
            print(f"\nFailed checks:")
            for component, status in self.results.items():
                if status == "FAIL":
                    print(f"  - {component}")
    
    def verify_all(self) -> bool:
        """Run all verification checks."""
        print("🔍 Verifying Knowledge Graph RAG Environment...")
        
        checks = [
            self.check_python_version,
            self.check_pytorch,
            self.check_ml_libraries,
            self.check_graph_database,
            self.check_vector_libraries,
            self.check_langchain,
            self.check_additional_libraries,
            self.check_model_cache,
            self.check_project_structure,
            self.check_system_resources,
            self.create_directories,
            self.run_model_tests
        ]
        
        all_passed = True
        for check in checks:
            try:
                if not check():
                    all_passed = False
            except Exception as e:
                logger.error(f"Check failed with exception: {e}")
                all_passed = False
        
        self.generate_report()
        return all_passed


def main():
    """Main verification function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Knowledge Graph RAG environment")
    parser.add_argument("--no-install", action="store_true", 
                       help="Don't automatically install missing packages")
    args = parser.parse_args()
    
    verifier = EnvironmentVerifier(auto_install=not args.no_install)
    success = verifier.verify_all()
    
    if success:
        print("\n✅ Environment verification completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Environment verification found issues.")
        sys.exit(1)


if __name__ == "__main__":
    main() 