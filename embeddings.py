from imports import *

"""
Create embeddings for nodes and store in database

- nodeType: Name of node type (__Entity__ for all nodes)
- encodeField: Field of node to encode
- embName: Name of vector index
"""
def embedNodes(nodeType, enocdeField, embName):
    entities = kg.query(f"""
    MATCH (n:{nodeType})
    WHERE n.id IS NOT NULL AND n.embedding IS NULL
    RETURN n.id AS id
    """)

    ids = [e["id"] for e in entities]

    for batch in range(0, len(ids), 25):  # batch size = 25
        chunk = ids[batch:batch + 25]
        
        params = {
            "id_list": chunk,
            "openAiApiKey": OPENAI_API_KEY,
            "openAiEndpoint": OPENAI_ENDPOINT,
            "embeddingModel": embedding_model_name
        }
        
        cypher = f"""
        UNWIND $id_list AS entity_id
        MATCH (n:{nodeType} {{id: entity_id}})
        WITH n, genai.vector.encode(
          n.{enocdeField},
          "OpenAI",
          {{
            token: $openAiApiKey,
            endpoint: $openAiEndpoint,
            model: $embeddingModel
          }}) AS vector
        WITH n, vector
        WHERE vector IS NOT NULL
        CALL db.create.setNodeVectorProperty(n, "embedding", vector)
        """
        kg.query(cypher, params)


    index_cypher = f"""
    CREATE VECTOR INDEX {embName} IF NOT EXISTS
    FOR (n:{nodeType}) ON (n.embedding)
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {len_embeddings},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """
    kg.query(index_cypher)

"""
Create a new node type to store all node types in a field
Create embeddings and store embeddings for node types
"""
def embedLabels():
    result = kg.query("""
    CALL db.labels() YIELD label
    WHERE label <> "Document" AND label <> "__Entity__"
    RETURN label
    """)
    types = [row["label"] for row in result]

    type_embeddings = {
        label: embedding_model.embed_query(label) for label in types
    }

    for label, embedding in type_embeddings.items():
        params={
            "label": label,
            "embedding": embedding
        }
        
        kg.query("""
        MERGE (t:labelEmbeds {name: $label})
        SET t.embedding = $embedding
        """, params=params)
        
    index_cypher = f"""
    CREATE VECTOR INDEX labelEmbeddings IF NOT EXISTS
    FOR (n:labelEmbeds) ON (n.embedding)
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {len_embeddings},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """
    kg.query(index_cypher)
    
"""
Creates embeddings all nodes, node types and raw chunks
"""
def createEmbeddings():
    embedNodes("`__Entity__`", "id", "entityEmbeddings")
    embedNodes("Document", "text", "docEmbeddings")
    embedLabels()

"""
Only used to clear the knowledge graph in this project

- files: unused
- resetStoredFiles: unused
- recreate_kg: unused
- clear: True/False to clear the existing data in the database
- includeSummary: unused
"""
def setupData(files, resetStoredFiles=False, recreate_kg=False, clear=False, includeSummary=False):
    if resetStoredFiles:
        for filename in os.listdir(saveDataPath):
            file_path = os.path.join(saveDataPath, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleared: {saveDataPath}")

    if clear:
        kg.query("MATCH (n) DETACH DELETE n")
        indexes = kg.query("SHOW VECTOR INDEXES YIELD name RETURN name")
        for idx in indexes:
            kg.query(f"DROP INDEX {idx['name']} IF EXISTS")
            print(f"Dropped index: {idx['name']}")
            
        print("Cleared existing knowledge graph")
        
    if recreate_kg:
        indexes = kg.query("SHOW VECTOR INDEXES YIELD name RETURN name")
        for idx in indexes:
            kg.query(f"DROP INDEX {idx['name']} IF EXISTS")
            print(f"Dropped index: {idx['name']}")
    
        for filePath in files:
            fileName = filePath.replace(datasetsPath, "")
            fileName = fileName[:fileName.find(".")]
            raw_documents_file = saveDataPath + fileName.lower().replace(" ", "_") + "_raw.pkl"
            if os.path.exists(raw_documents_file):
                continue

            print("Adding", fileName[1:], "to knowledge graph")

            graph_documents = getChunksGraph(filePath, includeSummary=includeSummary)
            createGraph(graph_documents)
            createEmbeddings()