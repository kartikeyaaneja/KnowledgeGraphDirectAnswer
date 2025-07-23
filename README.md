# GraphRAG: Graph-Augmented Retrieval for Question Answering

GraphRAG is a retrieval-augmented generation (RAG) system that builds and queries a knowledge graph using both semantic embeddings and symbolic entity relationships to generate answers to user questions.

---

## 📁 Project Structure

### Notebooks
- **GraphRAG.ipynb**  
  Interactive notebook to run and test the entire GraphRAG pipeline.

### Python Modules
- **imports.py**  
  Centralized import management and configuration, including model and Neo4j initialization.

- **embeddings.py**  
  Functions to create vector embeddings for graph nodes and labels, and store them in Neo4j.

- **rag.py**  
  Core logic for building the graph, retrieving context using structured and unstructured methods, re-ranking, and generating answers.

- **rag_app.py**  
  FASTAPI app exposing endpoints for building the graph (`/build`) and answering questions (`/ask`).

- **rag_eval.py**  
  Evaluation logic that uses LLaMA to compare predicted vs expected answers and compute recall.

---

## 🚀 Running the Project

### Setup Neo4j

- Create an account on [Neo4j Aura](https://neo4j.com/cloud/aura/) and set up a new project.
- Copy your database connection details and update the following fields in `imports.py`:

```python
NEO4J_URI = "neo4j+s://<your-instance-id>.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "<your-password>"
NEO4J_DATABASE = "neo4j"

AURA_INSTANCEID = "<your-instance-id>"
AURA_INSTANCENAME = "<your-instance-name>"
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Option 1: Run in Jupyter Lab

To interactively run the GraphRAG pipeline:

1. Open a terminal and start Jupyter Lab:
   ```bash
   jupyter lab
   ```

2. Open the notebook:
   ```
   GraphRAG.ipynb
   ```

3. Execute the cells sequentially to test graph construction, embedding, and querying.

---

### Option 2: Run as API using FASTAPI

To serve the GraphRAG system via REST API:

1. In the terminal, run:
   ```bash
   uvicorn rag_app:app --reload
   ```

2. The API will be accessible at:
   ```
   http://127.0.0.1:8000
   ```

---

#### 📌 Build Endpoint

Build the knowledge graph from QA data:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/build" -Method POST
```

#### ❓ Ask Endpoint

Ask a question using a POST request:

```powershell
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" `
   -Method Post `
   -Body '{"question": "What is Programme Contingency and what replaces the role of nominated subcontractors?", "filterLimit": -3, "minimum": 5}' `
   -ContentType "application/json"

$response.answer
```

---

### 🔍 Evaluation (in Jupyter Lab)

To run evaluation and compute recall accuracy using LLaMA:

- Done in the GraphRAG.ipynb notebook
- Run all cells to see results again/Change parameters to run evaluation again

---

## 📌 Notes

- Neo4j connection credentials and API keys are configured in `imports.py`.
- Graph embeddings are based on OpenAI's `text-embedding-3-large`.
- Evaluation logic using LLaMA is provided in `rag_eval.py`.