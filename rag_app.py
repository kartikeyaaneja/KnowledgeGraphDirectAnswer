from fastapi import FastAPI, Query
from pydantic import BaseModel
from rag import *

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    filterLimit: int = -3
    minimum: int = 5

    
"""
End point to build knowledge graph from QAs no input
"""
@app.post("/build")
def build_graph():
    try:
        llm_transformer = LLMGraphTransformer(llm=llm)
        buildKG(verbose=True)
        createEmbeddings()
        return {"status": "Graph and embeddings built successfully"}
    except Exception as e:
        return {"error": str(e)}

    
"""
End point to ask a question

In the format of QuestionRequest as defined in the class
"""
@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        answer = answerQuestion(
            question=request.question,
            filterLimit=request.filterLimit,
            minimum=request.minimum
        )
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
