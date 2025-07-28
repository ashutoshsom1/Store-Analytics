import streamlit as st
from utils.llm_setup import get_llm_tookits
from sentence_transformers import SentenceTransformer
import pandas as pd
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
import faiss

from dotenv import load_dotenv
load_dotenv(".env")

import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

from logger import logger
from utils.utils import format_metadata_csv_for_prompt


# This is top-level, runs on app startup or script rerun
@st.cache_resource
def load_models(base_dir:str=os.getcwd()):
    llm,db,toolkit,context,tools=get_llm_tookits()
    vec_db_path = os.path.join(base_dir, "data", "ikea_sql_vectors.faiss")
    df_path = os.path.join(base_dir, "data", "ikea_metadata.pkl")
    model = SentenceTransformer("all-MiniLM-L6-v2", token=False)
    
    # Check if IKEA vectors exist, if not use original path temporarily
    if os.path.exists(vec_db_path):
        index = faiss.read_index(vec_db_path)
        df = pd.read_pickle(df_path)
    else:
        # Fallback to original paths if IKEA vectors not created yet
        vec_db_path = os.path.join(base_dir, "data", "sql_vectors.faiss")
        df_path = os.path.join(base_dir, "data", "metadata.pkl")
        index = faiss.read_index(vec_db_path)
        df = pd.read_pickle(df_path)
    
    table_schema = db.get_context()['table_info']
    meta_data_path=os.path.join(base_dir,"data","metadata_table.csv")

    return {
        'llm':llm,'db':db,'toolkit':toolkit,'tools':tools,
        'vec_db_path':vec_db_path,'df_path':df_path,'model':model,
        'index':index,'df':df,'table_schema':table_schema,
        'meta_data_path':meta_data_path,'context':context
    }


@st.cache_resource
def get_retriever_tool(base_dir:str=os.getcwd(),model_name="BAAI/bge-base-en-v1.5"):
    # Build the relative path to the FAISS vector DB
    faiss_path = os.path.join(base_dir, "data","retriever")
    # Load the FAISS vector DB
    embeddings = FastEmbedEmbeddings(model_name=model_name)
    vector_db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    retriever_tool = create_retriever_tool(
        retriever,
        name="query_documents",
        description='''
        Used for querying IKEA store-related documents directly, or for querying them after receiving output from an SQL query.
        It provides contextual insights (e.g., product information, store details, food & beverage offerings, store policies, or customer services) 
        for non-sales questions or to supplement SQL-based answers about IKEA store performance and operations.
        ''',
    )
    return (
        retriever_tool,
        vector_db
        )

resources = load_models()
llm = resources['llm']
db = resources['db']
toolkit = resources['toolkit']
tools = resources['tools']
vec_db_path = resources['vec_db_path']
df_path = resources['df_path']
model = resources['model']
index = resources['index']
df = resources['df']
context = resources['context']
table_schema = resources['table_schema']
metadata_path = resources['meta_data_path']
# get the necessary tools for the retriver
retriever_tool, vector_db = get_retriever_tool()
metadata_content=format_metadata_csv_for_prompt(metadata_path)
logger.info(f"###############✅ Resource Loading Completed################")