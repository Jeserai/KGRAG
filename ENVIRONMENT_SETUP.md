# Knowledge Graph RAG Environment Setup Guide

This guide provides comprehensive instructions for setting up the Knowledge Graph RAG environment on compute nodes.

## 🚀 Quick Start

### 1. Automated Setup (Recommended)
```bash
# Make the setup script executable
chmod +x setup_environment.sh

# Run the automated setup
./setup_environment.sh
```

### 2. Verify Environment
```bash
# Run the verification script
python verify_environment.py
```

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.9 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **GPU**: CUDA-compatible GPU (optional, CPU fallback available)

### Recommended Requirements
- **Python**: 3.10 or higher
- **RAM**: 16GB or higher
- **Storage**: 50GB free space
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: 11.8 or higher

## 🔧 Manual Setup Steps

### 1. Python Environment

#### Check Python Version
```bash
python3 --version
# Should be 3.9 or higher
```

#### Install Python Dependencies
```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 2. CUDA Setup (GPU Users)

#### Check CUDA Installation
```bash
nvidia-smi
# Should show GPU information and CUDA version
```

#### Verify PyTorch CUDA Support
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

### 3. Neo4j Database Setup

#### Option A: Docker (Recommended)
```bash
# Pull and run Neo4j container
docker run \
  --name neo4j-kgrag \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["apoc"]' \
  -v neo4j_data:/data \
  -v neo4j_logs:/logs \
  -d neo4j:latest

# Check if container is running
docker ps | grep neo4j
```

#### Option B: Local Installation
```bash
# Ubuntu/Debian
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install neo4j

# Start Neo4j service
sudo systemctl start neo4j
sudo systemctl enable neo4j
```

#### Option C: Package Manager
```bash
# macOS with Homebrew
brew install neo4j

# Start Neo4j
brew services start neo4j
```

### 4. Model Cache Setup

#### Check HuggingFace Cache
```bash
# Check cache location
ls -la ~/.cache/huggingface/

# Check cache size
du -sh ~/.cache/huggingface/
```

#### Download Required Models (Optional)
```python
from transformers import AutoTokenizer, AutoModel

# Download models to cache
models = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen3-Embedding-0.6B"
]

for model_name in models:
    print(f"Downloading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
```

## 🔍 Environment Verification

### Run Comprehensive Verification
```bash
python verify_environment.py
```

### Manual Verification Steps

#### 1. Check Python Packages
```python
import torch
import transformers
import sentence_transformers
import neo4j
import faiss
import langchain

print("All packages imported successfully!")
```

#### 2. Test CUDA
```python
import torch
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
else:
    print("CUDA not available, will use CPU")
```

#### 3. Test Neo4j Connection
```python
from neo4j import GraphDatabase

try:
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password")
    )
    with driver.session() as session:
        result = session.run("RETURN 1 as test")
        print("Neo4j connection successful!")
    driver.close()
except Exception as e:
    print(f"Neo4j connection failed: {e}")
```

#### 4. Test Model Loading
```python
from src.models.model_config import config_manager

# Test LLM config
llm_config = config_manager.get_default_llm_config()
print(f"LLM model: {llm_config.name}")

# Test embedding config
embedding_config = config_manager.get_default_embedding_config()
print(f"Embedding model: {embedding_config.name}")
```

## 🐳 Docker Setup (Alternative)

### Docker Compose Setup
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:latest
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

  kgrag-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USERNAME: neo4j
      NEO4J_PASSWORD: password
    depends_on:
      - neo4j
    volumes:
      - ./data:/app/data
      - ./cache:/app/cache

volumes:
  neo4j_data:
  neo4j_logs:
```

### Dockerfile
Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data cache output logs

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "src.main"]
```

## 🔧 Configuration

### Environment Variables
Create `.env` file:
```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

# Model Configuration
DEFAULT_LLM_MODEL=llama_3_2_1b_instruct
DEFAULT_EMBEDDING_MODEL=qwen_3_embedding_0_6b

# Performance Configuration
MAX_MEMORY_GB=16
ENABLE_MODEL_CACHING=true
CPU_FALLBACK=true

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=kg_rag.log

# Data Configuration
DATA_DIR=./data
CACHE_DIR=./cache
OUTPUT_DIR=./output
```

### Model Configuration
Edit `config/model_configs.yaml` to match your available models:
```yaml
default:
  llm_model: "llama_3_2_1b_instruct"  # Use your cached model
  embedding_model: "qwen_3_embedding_0_6b"  # Use your cached model
```

## 🚨 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Reduce batch size in config
batch_size: 2  # Instead of 8

# Enable gradient checkpointing
enable_gradient_checkpointing: true

# Use CPU fallback
device: "cpu"
```

#### 2. Neo4j Connection Failed
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Check Neo4j logs
docker logs neo4j-kgrag

# Test connection manually
curl http://localhost:7474
```

#### 3. Model Loading Failed
```python
# Check model cache
ls -la ~/.cache/huggingface/hub/

# Clear cache if corrupted
rm -rf ~/.cache/huggingface/hub/

# Re-download models
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
```

#### 4. Python Package Conflicts
```bash
# Create virtual environment
python3 -m venv kgrag_env
source kgrag_env/bin/activate

# Install in clean environment
pip install -r requirements.txt
```

### Performance Optimization

#### 1. Memory Optimization
```yaml
# In config/model_configs.yaml
performance:
  max_memory_gb: 8  # Reduce if needed
  enable_model_caching: true
  cpu_fallback: true
```

#### 2. GPU Optimization
```yaml
# Use 4-bit quantization
quantization: "4bit"
load_in_4bit: true

# Adjust batch sizes
batch_size: 2  # Smaller for larger models
```

#### 3. Database Optimization
```cypher
// Create indexes for better performance
CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX relationship_type_index IF NOT EXISTS FOR (r:Relationship) ON (r.type);
```

## 📊 Monitoring

### System Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor memory usage
htop

# Monitor disk usage
df -h
```

### Application Monitoring
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor model loading
from src.models.local_llm import LocalLLMManager
llm = LocalLLMManager()
print(llm.get_model_info())
```

## ✅ Final Checklist

- [ ] Python 3.9+ installed and working
- [ ] All Python packages installed (`pip install -r requirements.txt`)
- [ ] CUDA available (if using GPU)
- [ ] Neo4j database running and accessible
- [ ] Model cache populated with required models
- [ ] Configuration files updated with correct model names
- [ ] Environment variables set (`.env` file created)
- [ ] Verification script passes (`python verify_environment.py`)
- [ ] Test run successful

## 🎯 Next Steps

After successful environment setup:

1. **Test the system**:
   ```bash
   python -c "from src.kg.graph_builder import KnowledgeGraphBuilder; print('System ready!')"
   ```

2. **Run on sample data**:
   ```python
   from src.kg.graph_builder import KnowledgeGraphBuilder
   
   builder = KnowledgeGraphBuilder()
   stats = builder.build_from_texts(["Sample text for testing..."])
   print(f"Built graph with {stats['entities_stored']} entities")
   ```

3. **Process your dataset**:
   ```python
   # Process files from directory
   stats = builder.build_from_directory("./data/documents")
   ```

4. **Monitor performance** and adjust configuration as needed.

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run the verification script: `python verify_environment.py`
3. Check logs in the `logs/` directory
4. Review the configuration files for errors

For additional help, refer to the main README.md file or create an issue in the project repository. 