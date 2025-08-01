from rag import *

def check_contains_with_gpt(row, pertub=False):
    if pertub:
        queryCol = "pertub_query"
        predictedCol = "pertub_predicted"
    else:
        queryCol = "query"
        predictedCol = "predicted"

    system_prompt = "You are an evaluator assessing whether a predicted answer captures the **main meaning** of the expected answer."
    user_prompt = f"""
You are a generous evaluator. Your goal is to determine whether the **predicted answer reflects the intended meaning** of the expected answer, even if the wording is different or the match is only partial.

- Be lenient: allow paraphrases, synonyms, and generalizations.
- The predicted answer does **not** need to include all details, but it should reflect the **main idea** clearly.
- If the core meaning of the expected answer is **clearly present or paraphrased**, consider it a match.
- Do **not** accept answers that are vague, mostly unrelated, or mention the topic without conveying the intent.
- Ignore unrelated or repetitive filler.

Query:
{row[queryCol]}

Expected Answer (main idea to be reflected):
{row['expected']}

Predicted Answer:
{row[predictedCol]}

Does the predicted answer contain the **main idea or intent** of the expected answer in any form (direct, indirect, partial, or paraphrased), as long as it is clear and relevant?

Respond only with "Yes" or "No".
"""

    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        answer = response.content.strip().lower()
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
Evaluates a single predicted answer using the LLM

- row: row of a dataframe with question, expected answer, and predicted answer from graph RAG system
"""
def check_contains_with_llama(row, pertub=False):
    if pertub:
        queryCol = "pertub_query"
        predictedCol = "pertub_predicted"
    else:
        queryCol = "query"
        predictedCol = "predicted"
    
    # Prompt to use LLM as judge
    system_prompt = "You are an evaluator assessing whether a predicted answer captures the **main meaning** of the expected answer."
    prompt = f"""
You are a generous evaluator. Your goal is to determine whether the **predicted answer reflects the intended meaning** of the expected answer, even if the wording is different or the match is only partial.

- Be lenient: allow paraphrases, synonyms, and generalizations.
- The predicted answer does **not** need to include all details, but it should reflect the **main idea** clearly.
- If the core meaning of the expected answer is **clearly present or paraphrased**, consider it a match.
- Do **not** accept answers that are vague, mostly unrelated, or mention the topic without conveying the intent.
- Ignore unrelated or repetitive filler.

Query:
{row[queryCol]}

Expected Answer (main idea to be reflected):
{row['expected']}

Predicted Answer:
{row[predictedCol]}

Does the predicted answer contain the **main idea or intent** of the expected answer in any form (direct, indirect, partial, or paraphrased), as long as it is clear and relevant?

Respond only with "Yes" or "No".
"""

    try:
        response = ollama.chat(
            model='llama3.2',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
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
Use LLaMA 3.2 via Ollama to generate a paraphrased question using synonyms.
"""
def pertub_question(prompt):
    system_prompt = (
        "You are a helpful assistant. Rephrase the following question using synonyms, "
        "preserving its meaning and intent. Provide only ONE rephrased version."
    )
    full_prompt = f"Rephrase this question into a single alternative form: {prompt}"

    try:
        response = ollama.chat(model="llama3.2", messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ])
        return response['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"

"""
Evaluates the Graph RAG system on quesitions that were used to build it

- runAgain: run the evaluation again or load saved evaluation
"""
def ragEval(predictAgain=False, evalAgain=False, dummy=False, pertub=False, dontPredict=False):
    if dummy:            
        qaInp = dummyQA
        qaAns = dummyQAAns
    else:
        if not os.path.exists(actualQA):
            df = pd.read_json(actualQARaw)
            df = df[["question", "answer"]]
            df = df.rename(columns={"question": "query", "answer": "expected"})
            df.to_json(actualQA)
        
        qaInp = actualQA
        qaAns = actualQAAns
    
    if os.path.exists(qaAns):
        df = pd.read_json(qaAns)
    else:
        df = pd.read_json(qaInp)
    
    if pertub:
        if "pertub_query" not in df.columns or predictAgain:
            df["pertub_query"] = df["query"].progress_apply(pertub_question)
            df.to_json(qaAns)
            
        predictedCol = "pertub_predicted"
        queryCol = "pertub_query"
        evalCol = "pertub_eval"
        gptEvalCol = "pertub_eval_gpt"           
    else:
        predictedCol = "predicted"
        queryCol = "query"
        evalCol = "llama_eval"
        gptEvalCol = "gpt_eval"

    
    if predictedCol not in df.columns or predictAgain:
        df[predictedCol] = None  # Initialize the column if not present
    
    if not dontPredict:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if pd.isna(df.at[idx, predictedCol]):
                df.at[idx, predictedCol] = answerQuestion(row[queryCol], filterLimit=0, minimum=4)
                df.to_json(qaAns)
            

    # Store LLM's evaluation
    if evalCol not in df.columns or evalAgain:
        df[evalCol] = df.progress_apply(lambda x:check_contains_with_llama(x, pertub), axis=1)
        df.to_json(qaAns)
        
    if gptEvalCol not in df.columns or evalAgain:
        df[gptEvalCol] = df.progress_apply(lambda x: check_contains_with_gpt(x, pertub), axis=1)
        df.to_json(qaAns)


    # Return accuracy
    return {"llama3.2":sum(df[evalCol] == "Yes") / len(df), "gpt3.5-turbo":sum(df[gptEvalCol] == "Yes") / len(df)}