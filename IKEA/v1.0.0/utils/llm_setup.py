import re
import langchain
langchain.verbose = False
import os
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from constants.constant import *
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
import streamlit as st
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

load_dotenv(r".env")
os.environ['LANGCHAIN_TRACING_V2']='true'
#os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
api_key = os.getenv("LANGCHAIN_API_KEY")
if api_key is not None:
    os.environ['LANGCHAIN_API_KEY'] = api_key
else:
    raise EnvironmentError("LANGCHAIN_API_KEY is not set in the environment.")
# Set the global cache
set_llm_cache(SQLiteCache(database_path=".langchain_cache.sqlite"))

@st.cache_resource
def get_llm_tookits(temperature:int=0):
    llm = AzureChatOpenAI( 
        openai_api_key=os.getenv("KEY"),
        azure_endpoint=os.getenv("ENDPOINT"),
        openai_api_version= os.getenv("API_VERSION"),
        deployment_name=os.getenv("DEPLOYMENT_NAME"),
        temperature = temperature,
        max_tokens=1500,
        timeout=180,
        max_retries=3,
        cache=True,
        seed=int(os.getenv("SEED"))
    )
    # Get the directory where the current .py file is located
    base_dir = os.getcwd()
    # Build the path to the IKEA SQLite database file (relative to this script)
    db_path = os.path.join(base_dir, "data", "ikea_store_database.db")
    # Create SQLDatabase instance with the full path
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    context = toolkit.get_context()
    tools = toolkit.get_tools()
    return (
        llm,
        db,
        toolkit,
        context,
        tools
        )