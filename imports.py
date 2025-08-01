import os
import re
import ollama
import shutil
import pickle
import inflect
import pandas as pd

# Standard libraries
from tqdm import tqdm
from datetime import date
from pprint import pprint
from datetime import datetime
from bs4 import BeautifulSoup
from typing import Tuple, List
from dotenv import load_dotenv
from neo4j import GraphDatabase
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.utils import logging as hf_logging

from FlagEmbedding import FlagReranker
from pydantic import BaseModel, Field


# Core LangChain components
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


# LangChain integrations
from langchain.chains import LLMChain
from langchain.text_splitter import TokenTextSplitter
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

import warnings
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
tqdm.pandas()

NEO4J_URI=""
NEO4J_USERNAME=""
NEO4J_PASSWORD=""
NEO4J_DATABASE=""

AURA_INSTANCEID=""
AURA_INSTANCENAME=""

if load_dotenv("neo4j.env"):
    NEO4J_URI=os.getenv("NEO4J_URI")
    NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE=os.getenv("NEO4J_DATABASE")

    AURA_INSTANCEID=os.getenv("AURA_INSTANCEID")
    AURA_INSTANCENAME=os.getenv("AURA_INSTANCENAME")

AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_ENDPOINT = "https://api.openai.com/v1/embeddings"

datasetsPath = "./datasets"
fileType = ".md"
saveDataPath = "./data"

chat_model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model=chat_model_name, temperature=0)

entity_extractor_model_name = "dslim/bert-base-NER"

embedding_model_name = "text-embedding-3-large"
embedding_model = OpenAIEmbeddings(
  model=embedding_model_name, 
  openai_api_key=OPENAI_API_KEY
)
dummyEmbed = embedding_model.embed_query("test")
len_embeddings = len(dummyEmbed)

reRankModelName = "BAAI/bge-reranker-large"
reRankModel = FlagReranker(reRankModelName, use_fp16=True)

kg = Neo4jGraph(
  url=NEO4J_URI,
  username=NEO4J_USERNAME,
  password=NEO4J_PASSWORD,
  database=NEO4J_DATABASE,
)

dummyQA = "./datasets/qa_dummy.json"
dummyQAGraph = dummyQA.replace(".json", "_graph.pkl")
dummyQAAns = dummyQA.replace(".json", "_predicted.json")

actualQARaw = "./datasets/qa_pairs_raw.json"
actualQA = actualQARaw.replace("_raw", "")
actualQAGraph = actualQA.replace(".json", "_graph.pkl")
actualQAAns = actualQA.replace(".json", "_predicted.json")

rawChunks = "./datasets/largeEmbeddings.csv"
rawChunksGraph = "./datasets/rawChunksGraph.pkl"

pegasus_paraphraser = pipeline("text2text-generation", model="tuner007/pegasus_paraphrase", device=0)

files = [
    os.path.join(datasetsPath, f)
    for f in os.listdir(datasetsPath)
    if os.path.isfile(os.path.join(datasetsPath, f)) and f.endswith(fileType)
]
files