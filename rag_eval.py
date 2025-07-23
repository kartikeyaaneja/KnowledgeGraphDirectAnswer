from rag import *

"""
Evaluates a single predicted answer using the LLM

- row: row of a dataframe with question, expected answer, and predicted answer from graph RAG system
"""
def check_contains_with_llama(row):
    # Prompt to use LLM as judge    
    prompt = f"""
You are a helpful evaluator. For the given query, determine if the predicted answer is correct and matches with the expected answer.
The predicted can contain other statements not relevant but just determine if it contains the expected statements

Query:
{row['query']}

Expected Answer:
{row['expected']}

Predicted Answer:
{row['predicted']}

Respond only with "Yes" or "No" — no explanations. Does the predicted answer contain the expected answer?
"""
    try:
        response = ollama.chat(
            model='llama3.2',
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )
        answer = response["message"]["content"].strip().lower()
        if "yes" in answer:
            return "Yes"
        elif "no" in answer:
            return "No"
        else:
            return "Unknown"
    except Exception as e:
        print(f"Error for row:\n{row}\n{e}")
        return "Error"

"""
Evaluates the Graph RAG system on quesitions that were used to build it

- runAgain: run the evaluation again or load saved evaluation
"""
def recallEval(runAgain=False):
    if os.path.exists(qaAnswered) and not runAgain:
        df = pd.read_json(qaAnswered)
    else:
        df = pd.read_json(qaInput)
        # Use GraphRAG to predict each question
        df["predicted"] = df["query"].apply(lambda x:answerQuestion(x, filterLimit=0, minimum=4))
        df.to_json(qaAnswered)
    
    # Store LLM's evaluation
    if "llama_eval" not in df.columns:
        df["llama_eval"] = df.apply(check_contains_with_llama, axis=1)
        df.to_json(qaAnswered)
        
    # Return accuracy
    return sum(df["llama_eval"] == "Yes") / len(df)