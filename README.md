# azure-doc-processing-pipeline

# Azure Document Processing Pipeline

This document outlines the steps to create an Azure pipeline that processes documents through chunking, embedding generation, and storage in Azure AI Search.

## Architecture Overview

```
Input Documents → Chunking → Embedding Generation → Azure AI Search
```

## Prerequisites

1. **Azure Resources**
   - Azure Storage Account (for input documents)
   - Azure OpenAI Service (for embeddings)
   - Azure AI Search service
   - Azure Functions or Azure Container Instances (for processing)

2. **Required Permissions**
   - Storage Blob Data Contributor
   - Cognitive Services OpenAI User
   - Search Index Data Contributor

## Step 1: Set Up Azure Resources

### 1.1 Create Azure AI Search Service
```bash
az search service create \
  --name "your-search-service" \
  --resource-group "your-rg" \
  --sku "standard" \
  --location "eastus"
```

### 1.2 Create Azure OpenAI Service
```bash
az cognitiveservices account create \
  --name "your-openai-service" \
  --resource-group "your-rg" \
  --kind "OpenAI" \
  --sku "S0" \
  --location "eastus"
```

### 1.3 Deploy Embedding Model
```bash
az cognitiveservices account deployment create \
  --name "your-openai-service" \
  --resource-group "your-rg" \
  --deployment-name "text-embedding-3-small" \
  --model-name "text-embedding-3-small" \
  --model-version "1" \
  --model-format "OpenAI" \
  --sku-capacity 10 \
  --sku-name "Standard"
```

## Step 2: Document Chunking Strategies

### 2.1 Fixed-Size Chunking
```python
def fixed_size_chunking(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks
```

### 2.2 Semantic Chunking
```python
def semantic_chunking(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[str]:
    """Split text based on semantic similarity."""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    model = SentenceTransformer(model_name)
    sentences = text.split('.')
    embeddings = model.encode(sentences)
    
    # Use cosine similarity to group semantically similar sentences
    similarity_threshold = 0.7
    chunks = []
    current_chunk = []
    
    for i, sentence in enumerate(sentences):
        if not current_chunk:
            current_chunk.append(sentence)
        else:
            # Calculate similarity with current chunk
            chunk_embedding = np.mean([embeddings[j] for j in range(len(current_chunk))], axis=0)
            sentence_embedding = embeddings[i]
            similarity = np.dot(chunk_embedding, sentence_embedding) / (
                np.linalg.norm(chunk_embedding) * np.linalg.norm(sentence_embedding)
            )
            
            if similarity > similarity_threshold and len(' '.join(current_chunk)) < 1000:
                current_chunk.append(sentence)
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

### 2.3 Document Structure-Based Chunking
```python
def structure_based_chunking(text: str, structure_markers: List[str] = None) -> List[str]:
    """Split text based on document structure (headers, paragraphs, etc.)."""
    if structure_markers is None:
        structure_markers = ['\n\n', '\n#', '\n##', '\n###']
    
    chunks = [text]
    
    for marker in structure_markers:
        new_chunks = []
        for chunk in chunks:
            new_chunks.extend(chunk.split(marker))
        chunks = [c.strip() for c in new_chunks if c.strip()]
    
    return chunks
```

## Step 3: Azure Function Implementation

### 3.1 Function App Structure
```
FunctionApp/
├── requirements.txt
├── host.json
├── function_app.py
└── shared/
    ├── chunking.py
    ├── embedding.py
    └── search_client.py
```

### 3.2 requirements.txt
```txt
azure-functions
azure-storage-blob
azure-search-documents
openai
sentence-transformers
numpy
pandas
python-docx
PyPDF2
```

### 3.3 Main Function (function_app.py)
```python
import azure.functions as func
import logging
import json
from shared.chunking import ChunkingService
from shared.embedding import EmbeddingService
from shared.search_client import SearchService

app = func.FunctionApp()

@app.blob_trigger(arg_name="myblob", 
                  path="documents/{name}",
                  connection="AzureWebJobsStorage")
async def document_processor(myblob: func.InputStream):
    logging.info(f"Processing blob: {myblob.name}")
    
    try:
        # Read document content
        content = myblob.read().decode('utf-8')
        
        # Initialize services
        chunking_service = ChunkingService()
        embedding_service = EmbeddingService()
        search_service = SearchService()
        
        # Process with different chunking strategies
        strategies = ['fixed_size', 'semantic', 'structure_based']
        
        for strategy in strategies:
            # Chunk the document
            chunks = chunking_service.chunk_document(content, strategy)
            
            # Generate embeddings
            embeddings = await embedding_service.generate_embeddings(chunks)
            
            # Prepare documents for search index
            documents = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc = {
                    "id": f"{myblob.name}_{strategy}_{i}",
                    "content": chunk,
                    "embedding": embedding,
                    "source_file": myblob.name,
                    "chunking_strategy": strategy,
                    "chunk_index": i
                }
                documents.append(doc)
            
            # Upload to Azure AI Search
            await search_service.upload_documents(documents)
        
        logging.info(f"Successfully processed {myblob.name}")
        
    except Exception as e:
        logging.error(f"Error processing {myblob.name}: {str(e)}")
        raise
```

### 3.4 Chunking Service (shared/chunking.py)
```python
from typing import List, Dict, Any
import re

class ChunkingService:
    def __init__(self):
        self.strategies = {
            'fixed_size': self._fixed_size_chunking,
            'semantic': self._semantic_chunking,
            'structure_based': self._structure_based_chunking
        }
    
    def chunk_document(self, content: str, strategy: str, **kwargs) -> List[str]:
        """Chunk document using specified strategy."""
        if strategy not in self.strategies:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
        
        return self.strategies[strategy](content, **kwargs)
    
    def _fixed_size_chunking(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Implementation from Step 2.1"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - overlap
        return [c for c in chunks if c]
    
    def _semantic_chunking(self, text: str, **kwargs) -> List[str]:
        """Implementation from Step 2.2"""
        # Simplified version - implement full semantic chunking as needed
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            if len(' '.join(current_chunk + [sentence])) > 1000:
                if current_chunk:
                    chunks.append(' '.join(current_chunk).strip())
                current_chunk = [sentence]
            else:
                current_chunk.append(sentence)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk).strip())
        
        return [c for c in chunks if c]
    
    def _structure_based_chunking(self, text: str, **kwargs) -> List[str]:
        """Implementation from Step 2.3"""
        # Split by paragraphs and headers
        chunks = re.split(r'\n\s*\n|\n#+\s', text)
        return [c.strip() for c in chunks if c.strip()]
```

### 3.5 Embedding Service (shared/embedding.py)
```python
import openai
from typing import List
import os
import asyncio

class EmbeddingService:
    def __init__(self):
        self.client = openai.AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = "text-embedding-3-small"
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text chunks."""
        embeddings = []
        
        # Process in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self._generate_batch_embeddings(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=self.deployment_name
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * 1536 for _ in texts]  # Adjust dimension as needed
```

### 3.6 Search Service (shared/search_client.py)
```python
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
)
from azure.core.credentials import AzureKeyCredential
from typing import List, Dict, Any
import os

class SearchService:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.key = os.getenv("AZURE_SEARCH_KEY")
        self.index_name = "document-chunks"
        
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )
        
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.key)
        )
        
        self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """Create the search index if it doesn't exist."""
        try:
            self.index_client.get_index(self.index_name)
        except:
            # Create index
            fields = [
                SearchField(name="id", type=SearchFieldDataType.String, key=True),
                SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
                SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                           vector_search_dimensions=1536, vector_search_profile_name="default"),
                SearchField(name="source_file", type=SearchFieldDataType.String, filterable=True),
                SearchField(name="chunking_strategy", type=SearchFieldDataType.String, filterable=True),
                SearchField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True)
            ]
            
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="default",
                        algorithm_configuration_name="hnsw-config"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(name="hnsw-config")
                ]
            )
            
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            self.index_client.create_index(index)
    
    async def upload_documents(self, documents: List[Dict[str, Any]]):
        """Upload documents to the search index."""
        try:
            result = self.search_client.upload_documents(documents)
            print(f"Uploaded {len(documents)} documents")
        except Exception as e:
            print(f"Error uploading documents: {e}")
            raise
```

## Step 4: Configuration and Environment Variables

### 4.1 Application Settings (local.settings.json for development)
```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "DefaultEndpointsProtocol=https;AccountName=...",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "AZURE_OPENAI_API_KEY": "your-openai-key",
    "AZURE_OPENAI_ENDPOINT": "https://your-openai.openai.azure.com/",
    "AZURE_SEARCH_ENDPOINT": "https://your-search.search.windows.net",
    "AZURE_SEARCH_KEY": "your-search-key"
  }
}
```

### 4.2 Azure Function Configuration
```json
{
  "version": "2.0",
  "extensionBundle": {
    "id": "Microsoft.Azure.Functions.ExtensionBundle",
    "version": "[4.*, 5.0.0)"
  },
  "functionTimeout": "00:10:00",
  "logging": {
    "applicationInsights": {
      "samplingSettings": {
        "isEnabled": true
      }
    }
  }
}
```

## Step 5: Deployment

### 5.1 Deploy Function App
```bash
# Create Function App
az functionapp create \
  --resource-group "your-rg" \
  --consumption-plan-location "eastus" \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --name "document-processor-func" \
  --storage-account "your-storage"

# Deploy code
func azure functionapp publish document-processor-func
```

### 5.2 Configure Application Settings
```bash
az functionapp config appsettings set \
  --name "document-processor-func" \
  --resource-group "your-rg" \
  --settings \
    AZURE_OPENAI_API_KEY="your-key" \
    AZURE_OPENAI_ENDPOINT="https://your-openai.openai.azure.com/" \
    AZURE_SEARCH_ENDPOINT="https://your-search.search.windows.net" \
    AZURE_SEARCH_KEY="your-search-key"
```

## Step 6: Testing and Monitoring

### 6.1 Test Document Upload
```python
from azure.storage.blob import BlobServiceClient

# Upload test document
blob_service_client = BlobServiceClient.from_connection_string("your-connection-string")
blob_client = blob_service_client.get_blob_client(container="documents", blob="test.txt")

with open("test_document.txt", "rb") as data:
    blob_client.upload_blob(data, overwrite=True)
```

### 6.2 Query Search Index
```python
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

search_client = SearchClient(
    endpoint="https://your-search.search.windows.net",
    index_name="document-chunks",
    credential=AzureKeyCredential("your-key")
)

# Search for documents
results = search_client.search(
    search_text="your query",
    top=5,
    include_total_count=True
)

for result in results:
    print(f"Score: {result['@search.score']}")
    print(f"Content: {result['content'][:200]}...")
    print(f"Strategy: {result['chunking_strategy']}")
    print("---")
```

## Best Practices

1. **Error Handling**: Implement comprehensive error handling and retry logic
2. **Monitoring**: Use Application Insights for monitoring and logging
3. **Performance**: Consider batch processing for large documents
4. **Security**: Use managed identities instead of keys where possible
5. **Cost Optimization**: Monitor API usage and implement caching strategies
6. **Testing**: Implement unit tests for chunking strategies and integration tests

## Scaling Considerations

1. **Function App**: Use Premium or Dedicated plans for consistent performance
2. **Azure AI Search**: Scale up the search service based on index size and query load
3. **OpenAI**: Monitor token usage and implement rate limiting
4. **Storage**: Use appropriate storage tiers based on access patterns

This pipeline provides a flexible foundation for processing documents with multiple chunking strategies and storing them in Azure AI Search for efficient retrieval.
