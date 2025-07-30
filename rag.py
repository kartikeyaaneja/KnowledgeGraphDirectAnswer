from embeddings import *

class Entities(BaseModel):
    names: List = Field(
        description="All the person, organization, or business entities that appear in the text",
    )

"""
Extract entities from the text to be later used in fuzzy search

- text: Text to extract entities from
"""
def extract_entities(text):
    text = remove_lucene_chars(text)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an entity extraction assistant specialized in identifying persons, organizations, and business names."),
            (
                "human",
                "Extract all person, organization, and business entity names from the following text.\n"
                "Return only a JSON array of unique entity names (strings), without explanation or extra text.\n"
                "Text:\n{question}"
            ),
        ]
    )
    
    llmEntityExtractorName = "gpt-3.5-turbo"
    llmEntityExtractor = ChatOpenAI(model=llmEntityExtractorName, temperature=0)
    
    entity_chain = prompt | llmEntityExtractor.with_structured_output(Entities)
    result = entity_chain.invoke({"question": text}).names
    return result

"""
Extract entities from the question and look up wether those entites appear in the Graph database
Extract all relationships of all entites present in the graph database and question

- question: User's question
"""
def fuzzyRetriever(question):
    results = set()
    
    entities = extract_entities(question)
    
    questions = ["~2 AND ".join(entity.split()) + "~2" for entity in entities]

    for fuzzy in questions:
        cypher = f"""
        CALL db.index.fulltext.queryNodes('entity', "{fuzzy}", {{limit: 5}})
        YIELD node, score
        CALL (node) {{
            WITH node
            MATCH (node)-[r:!MENTIONS]->(neighbor)
            RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
            UNION ALL
            WITH node
            MATCH (node)<-[r:!MENTIONS]-(neighbor)
            RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
        }}
        RETURN output LIMIT 50
        """

        ans = kg.query(cypher)
        for i in ans:
            results.add(i["output"])
            
    return results

"""
Retrieve all the relationships of the top k nodes that semantically match the embedding of the question

- question: User's question
- k: Number of nodes to retrieve
"""
def nodeEmbedRetriever(question, k=5):       
    cypher = """
    WITH genai.vector.encode(
        $question,
        "OpenAI",
        {
              token: $openAiApiKey,
              endpoint: $openAiEndpoint,
              model: $embeddingModel
        }) AS question_embedding
    CALL db.index.vector.queryNodes(
        'entityEmbeddings',
        $top_k,
        question_embedding
        )
    YIELD node, score
    WHERE NOT 'labelEmbeds' IN labels(node)
    CALL (node) {
        WITH node
        MATCH (node)-[r:!MENTIONS]->(neighbor)
        RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
        UNION ALL
        WITH node
        MATCH (node)<-[r:!MENTIONS]-(neighbor)
        RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
    }
    RETURN output LIMIT 50
    """

    params={
        "openAiApiKey": OPENAI_API_KEY,
        "openAiEndpoint": OPENAI_ENDPOINT,
        "question": question,
        "top_k": k,
        "embeddingModel": embedding_model_name
    }

    ans = kg.query(cypher, params=params)
    results = set()
    for path in ans:
        results.add(path["output"])
    
    return results

"""
Retrieve the node type that most semantically matches the question using cosine search on embeddings
Return the relationships of all node of that Node Type

- question: User's question
"""
def labelRetriever(question):
    cypher = """
    WITH genai.vector.encode(
        $question,
        "OpenAI",
        {
              token: $openAiApiKey,
              endpoint: $openAiEndpoint,
              model: $embeddingModel
        }) AS question_embedding
    CALL db.index.vector.queryNodes(
        'labelEmbeddings',
        $top_k,
        question_embedding
        )
    YIELD node, score
    RETURN node.name AS name, score
    """
    params={
        "openAiApiKey": OPENAI_API_KEY,
        "openAiEndpoint": OPENAI_ENDPOINT,
        "question": question,
        "top_k": 1,
        "embeddingModel": embedding_model_name
    }
    
    topLabel = kg.query(cypher, params=params)[0]["name"]
    
    cypher = f"""
    MATCH (node)
    WHERE '{topLabel}' IN labels(node)
    CALL (node) {{
        WITH node
        MATCH (node)-[r:!MENTIONS]->(neighbor)
        RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
        UNION ALL
        WITH node
        MATCH (node)<-[r:!MENTIONS]-(neighbor)
        RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
    }}
    RETURN output LIMIT 50
    """
    ans = kg.query(cypher)
    results = set()
    for path in ans:
        results.add(path["output"])
        
    return results

"""
Combines all retreievers to do with structured data (from knowledge graph)

- question: User's query
- k: Number of nodes to retrieve
- fuzzy: True/False wether to use fuzzy search, (in most cases it would be redundant, can eliminate to save time)
"""
def structuredRetriever(question, k=5, fuzzy=True): 
    results = set()
    
    if fuzzy:
        results.update(fuzzyRetriever(question))
        
    results.update(nodeEmbedRetriever(question, k))
    results.update(labelRetriever(question))
    
    return list(results)

"""
Unused in this project

Used to extract raw chunks that match the semantic embeddings of the question using cosine similarity

- question: User's question
- k: Number of raw chunks to retrieve
"""
def unstructuredRetriever(question, k=15):
    cypher = """
    WITH genai.vector.encode(
        $question,
        "OpenAI",
        {
              token: $openAiApiKey,
              endpoint: $openAiEndpoint,
              model: $embeddingModel
        }) AS question_embedding
    CALL db.index.vector.queryNodes(
        'docEmbeddings',
        $top_k,
        question_embedding
        )
    YIELD node, score
    RETURN node.text AS Chunks, score AS cosine_similarity
    """
    params={
        "openAiApiKey": OPENAI_API_KEY,
        "openAiEndpoint": OPENAI_ENDPOINT,
        "question": question,
        "top_k": k,
        "embeddingModel": embedding_model_name
    }
    
    ans = kg.query(cypher, params=params)
    dfDocs = pd.DataFrame(ans)
    
    return dfDocs

"""
Paraphrase a sentence using pegasus paraphraser

- text: text to paraphrase
"""
def paraphrase_pegasus(text):
    return pegasus_paraphraser(text, num_return_sequences=1, clean_up_tokenization_spaces=True)[0]['generated_text']

"""
Hugging face re-ranker to further sort the chunks/relationships

- query: User's question
- df: Contains the Chunks/Relationships to re-rank
- column: Which column to re-rank (Chunks or Relationship)
- filterLimit: Filter based on a minimum score
- minimum: Keep a minimum of x columns regardless of filterLimit
"""
def reRank(query, df, column="Chunks", filterLimit=-100, minimum=-1):
    # Compute a score for each chunk based on the contents of the chunk and query
    rankMatches = [[query, df[column].iloc[idx]] for idx in range(len(df))]
    scores = reRankModel.compute_score(rankMatches)
    df["rerank_score"] = scores     
    
    df = df.sort_values(by="rerank_score", ascending=False)
    
    filtered_df = df[df["rerank_score"] > filterLimit]
    if len(filtered_df) >= minimum:
        df = filtered_df
    else:
        df = df.head(minimum)
        
    return df

"""
Unused in this project

Combines structured data retrieved from knowledge graph with unstructured data (raw chunks) from cosine search

- question: User's question
- k: Number of raw chunks to use in prompt
- verbose: See data passed to prompt in print
- fuzzy: True/False wether to use fuzzy search
"""
def retriever(question, k=5, verbose=False, fuzzy=True):
    structured_data = structuredRetriever(question, fuzzy=fuzzy, k=k)
    structured = "\n".join(structured_data)
    unstructDf = unstructuredRetriever(question, k=15)
    rankedDf = reRank(question, unstructDf, k=k)
    unstructured_data = "DOCUMENT:\n" + "\nDOCUMENT:\n".join(list(rankedDf["Chunks"]))
    
    final_data = f"""
      Structured data: 
{structured}

      Unstructured data: 
{unstructured_data}
    """
    if verbose:
        print(f"\nFinal Data::: ==>{final_data}")

    return final_data

"""
Unused in this project

Use LLM to generate answer to question
"""
def generate(query, k=5, recurse=1, addContext="", fuzzy=True, verbose=False):
    match = re.search(r"(\d{2}/\d{2}/\d{4})", query)
    if match:
        dateQ = match.group(1)  
        query = query.replace(dateQ, convDateWords(dateQ))
    
    today = date.today()
    dateT = convDateWords(today.strftime("%m/%d/%Y"))
    
    context = retriever(addContext + " " + query, k=k, fuzzy=fuzzy)
    
    template = f"""
    Answer the question based only on the following context:
{context}

    You are also given additional context below which may help you reason better, 
    but this content may have been generated in previous attempts and should not be treated as ground truth. 
    Use it only if it logically supports your answer and cross-check with the main context above.

    Additional context to help answer:
{addContext}
    
    Date this question is being asked: {dateT}
    Question: {query}
    Use natural language and be concise. If the information is not present in the context given or cannot be inferred from the context then say answer doesn't exist.
    Answer:"""
    
    if verbose:
        print(template)


    prompt = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt.format()
    
    response = llm.invoke(formatted_prompt).content
    
    
    for i in range(recurse-1):
        addContext = addContext + " " + response
        response = generate(query, k=k, addContext=addContext, verbose=verbose)
    
    return response

"""
Send created knowledge graph to neo4j database
"""
def createGraph(graphDocuments):   
    result = kg.add_graph_documents(
        graphDocuments,
        include_source=True,
        baseEntityLabel=True,
    )

"""
Build graph from a json file of QAs

- verbose: See QAs and extracted nodes and relationships
- override: Build graph again instead of loading saved
"""
def buildGraphQA(verbose=False, override=False, dummy=False):
    if dummy:
        qaGraph = dummyQAGraph
        qaInput = dummyQA
    else:
        qaGraph = actualQAGraph
        qaInput = actualQA
    
    if os.path.exists(qaGraph) and not override:
        with open(qaGraph, 'rb') as f:
            qas_list, graph_documents = pickle.load(f)
    else:
        df = pd.read_json(qaInput)

        qas_list = []
        batch_size = 20
        qas_batch = []

        for i, row in enumerate(df.itertuples(index=False)):
            qas_batch.append(f"Q: {row.query}\nA: {row.expected}\n")
            # Every 20 items or last batch
            if (i + 1) % batch_size == 0:
                qas_list.append("\n".join(qas_batch))
                qas_batch = []

        # Append any remaining QAs
        if qas_batch:
            qas_list.append("\n".join(qas_batch))


        prompt = "Clauses and Numerical values, etc count as nodes"
        llm_transformer = LLMGraphTransformer(llm=llm, additional_instructions=prompt)
        documents = [Document(page_content=qa) for qa in qas_list]
        graph_documents = llm_transformer.convert_to_graph_documents(documents)

        with open(qaGraph, 'wb') as f:
            pickle.dump((qas_list, graph_documents), f)

    if verbose:
        print("\n".join(qas_list))
    createGraph(graph_documents)
    
    if verbose:
        print("Relationships:")
        for doc in graph_documents:
            for rel in doc.relationships:
                print(f"{rel.source.id} -[{rel.type}]-> {rel.target.id}")

        print("\nNodes:")
        seen_ids = set()
        for doc in graph_documents:
            for rel in doc.relationships:
                for node in [rel.source, rel.target]:
                    if node.id not in seen_ids:
                        seen_ids.add(node.id)
                        print(f"Node ID: {node.id}, Type: {node.type}")
                    
"""
Unused in this project as it takes too long

Build knowledge graph from raw chunks of the pdf
""" 
def buildGraphChunks():  
    if os.path.exists(rawChunksGraph):
        with open(rawChunksGraph, 'rb') as f:
            graph_documents = pickle.load(f)
    else:
        df = pd.read_csv(rawChunks)
        df = df[(df["imagePath"].isna()) & (df['Chunks'].str.len() > 100)]
        documents = [Document(page_content=i) for i in list(df["Chunks"])]
        graph_documents = llm_transformer.convert_to_graph_documents(documents)

        with open(rawChunksGraph, 'wb') as f:
            pickle.dump(graph_documents, f)
            
    createGraph(graph_documents)
          
"""
Combines knowledge graphs made from QAs and Raw Chunks

Currently ignores knowledge graph made from raw chunks
""" 
def buildKG(verbose=False, override=False, dummy=False):
    setupData(files, clear=True)
    
    buildGraphQA(verbose=verbose, override=override, dummy=dummy)
    #buildGraphChunks()
            
"""
Answer a user's question using GraphRAG system

- question: User's question
- filterLimit: Filter based on a minimum score (for re-ranker)
- minimum: Keep a minimum of x columns regardless of filterLimit (for re-ranker)
""" 
def answerQuestion(question, filterLimit=-100, minimum=-1):
    res = structuredRetriever(question, k=5, fuzzy=True)
    res = [i.replace("-", "").replace(">", "is").replace("_", " ") for i in res]
    df = pd.DataFrame({"Chunks":res})
    
    df["Paraphrased"] = df["Chunks"].apply(paraphrase_pegasus)
    df = reRank(question, df, column="Paraphrased", filterLimit=filterLimit, minimum=minimum)

    return "\n".join(list(df["Paraphrased"]))