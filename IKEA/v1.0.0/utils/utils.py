import re
import pandas as pd
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
import os
import yaml
from typing import List, Union, Optional
from langchain_experimental.utilities import PythonREPL
from logger import logger
from constants.constant import MARKDOWN_IMAGE_REGEX
from PIL import Image
import base64
from io import BytesIO
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import time
import random
import string


def generate_session_id():
    ts = int(time.time())  # e.g. 1716870400
    rand = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{ts}_{rand}"

def remove_markdown_images(text):
    return re.sub(MARKDOWN_IMAGE_REGEX, "", text)

def image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_sql(text: str) -> Optional[str]:
    pattern = r"""
        # === CASE 1: SQLQuery with optional markdown ===
        SQLQuery:\s*                # SQLQuery label
        (?:
            ```sql\s*              # optional markdown block
            (?P<code1>[\s\S]+?)    # group 1: SQL content
            \s*```                 # closing markdown
            |
            (?P<code2>[\s\S]+?)    # OR group 2: raw SQL without backticks
            (?=SQLResult:|$|\n\n)  # until SQLResult or end or double newline
        )
 
        |
 
        # === CASE 2: Only markdown SQL block ===
        ```sql\s*
        (?P<code3>[\s\S]+?)
        \s*```
    """
    match = re.search(pattern, text, re.IGNORECASE | re.VERBOSE)
    if match:
        for group in ("code1", "code2", "code3"):
            if match.group(group):
                return match.group(group).strip()
    else:
        match = re.search(
        r'^\s*(?P<code4>(SELECT|WITH)[\s\S]+)',
        text,
        re.IGNORECASE | re.VERBOSE
        )
        if match:  # Add a check if match is None
            return match.group(1).strip()
    return None

def format_metadata_csv_for_prompt(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    df['Primary Key'] = df['Primary Key'].astype(str).str.strip().str.lower()
    df['Foreign Key'] = df['Foreign Key'].astype(str).str.strip()

    grouped = df.groupby("Table Name")
    formatted_output = ""

    for table, group in grouped:
        pk_columns = group[group['Primary Key'] == 'yes']['Column Name'].tolist()
        # Detect composite foreign keys (same FK target shared by multiple columns)
        fk_targets = group[group['Foreign Key'] != 'No']['Foreign Key'].value_counts()
        composite_fks = fk_targets[fk_targets > 1].index.tolist()
        formatted_output += f"\n### Table: `{table}`\n"
        for _, row in group.iterrows():
            col_name = row['Column Name']
            dtype = row['Data Type']
            is_pk = col_name in pk_columns
            fk_target = row['Foreign Key']
            is_composite_fk = fk_target in composite_fks
            pk = " (PK)" if is_pk else ""
            fk = f" → {fk_target}" if fk_target.lower() != 'no' else ""
            desc = row['Description'] if pd.notna(row['Description']) else ""
            formatted_output += f"- `{col_name}` ({dtype}){pk}{fk}: {desc}\n"
        if len(pk_columns) > 1:
            pk_list = ', '.join([f"`{pk}`" for pk in pk_columns])
            formatted_output += f"Composite Primary Key: ({pk_list})\n"
        if composite_fks:
            for fk in composite_fks:
                formatted_output += f"Composite Foreign Key to: `{fk}` across multiple columns\n"
    return formatted_output


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""
    messages: List[BaseMessage] = Field(default_factory=list)
    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []
# global variable to store the chat message history.
store = {}
def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

def read_yaml(file_path:str)->str:
    """Load YAML file"""
    assert os.path.exists(file_path) and os.path.isfile(file_path) and (file_path.endswith('.yaml')),\
          f"FileNotFoundError: {file_path} doesn't exists or isn't a file or not a yaml file!!"
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data

def check_for_charts(repl_output:str)->bool:
    """checks if a charting function got called or not"""
    indicators = [
        'plt.'
        'st.area_chart',
        'st.bar_chart',
        'st.line_chart',
        'st.map',
        'st.scatter_chart',
        'st.altair_chart',
        'st.bokeh_chart',
        'st.graphviz_chart',
        'st.plotly_chart',
        'st.pydeck_chart',
        'st.pyplot',
        'st.vega_lite_chart'
    ]
    for indicator in indicators:
        if indicator in repl_output:
            return True

def check_for_errors(repl_output:str)->tuple:
    """checks for common python errors in the REPL tool output"""
    error_indicators = [
        "Traceback (most recent call last)",
        "SyntaxError",
        "NameError",
        "TypeError",
        "ValueError",
        "IndexError",
        "KeyError",
        "AttributeError",
        "ImportError",
        "ModuleNotFoundError"
    ]
    for indicator in error_indicators:
        if indicator in repl_output:
            return (
                True, 
                extract_error_message(repl_output)
                )
        elif (
            repl_output=='' 
            or repl_output is None
            ):
            return (
                False, 
                f"this python code returned no value when executed through the REPL tool; received ouput: {repl_output}"
                )
    return False, None

def extract_error_message(output):
    """extracts the error message from repl output"""
    lines = output.strip().split('\n')
    # Usually the last line contains the actual error message
    return lines[-1] if lines else ""

def handle_repl_error(code:str,error_msg:str)->str:
    """helper function that handles REPL execution error"""
    # partial import to mitigate circular import error
    from data.dataloader import llm
    prompt = f"""
        The following python code threw an error when executed with REPL tool for python. 
        Fix and give a improved code that achieves the same task keeping all the variables 
        and the datatypes consistent
        Code:
        {code}

        Error: {error_msg}
        
        Important Context for IKEA Store Analytics:
        - The correct database file is located at: data/ikea_store_database.db
        - The table name is: store_data
        - Use sqlite3.connect('data/ikea_store_database.db') to connect to the database
        - Available columns: Date, Store, Channel, Act, Ly, Visitors, Customers, Conversion, ATV, Item_Sold, Price_Item, Item_Cust, vs_Ly_percent
        - Store values: 'DJA', 'YAS'
        - Channel values: 'Store', 'IFB'
        - Date range: 2025-01-01 to 2025-06-30
        - Currency: AED
        
        You don't need to provide how you've generated the code; just return the fixed code and nothing more or less.
        any deviation from the expected behaviour will end up as a heavy penalty for you.
    """
    return llm.invoke(prompt).content

def run_python_code(code: str) -> str:
    """
    Runs the provided Python code with the necessary imports for streamlit visualization.
    Args:
        code (str): The Python code to be executed.
    Returns:
        The output of the Python code execution, or an error message if there's an issue.
    """
    # Initialize the Python REPL
    python_repl = PythonREPL()
    
    # Check if the code is a SQL query (starts with SELECT, WITH, etc.)
    code_stripped = code.strip()
    if (code_stripped.upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER')) 
        and 'import' not in code_stripped.lower() and 'sqlite3' not in code_stripped.lower()):
        # Convert SQL query to Python code that executes it against the IKEA database
        python_code = f"""
import sqlite3
import pandas as pd

# Connect to the IKEA store database
conn = sqlite3.connect('data/ikea_store_database.db')

# Execute the SQL query
query = '''
{code_stripped}
'''

try:
    # Execute query and fetch results
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    
    # Get column names
    column_names = [description[0] for description in cursor.description]
    
    # Create DataFrame for better display
    df = pd.DataFrame(results, columns=column_names)
    print("Query Results:")
    print(df.to_string(index=False))
    
    # Also print summary statistics if numeric data
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        print("\\nSummary Statistics:")
        print(df[numeric_cols].describe())
    
except Exception as e:
    print(f"Error executing query: {{str(e)}}")
finally:
    conn.close()
"""
        code_string = python_code
    else:
        code_string = code.replace("plt.show()", "")
    
    logger.info(f"👩‍💻 Modified Code =>\n {code_string}")
    
    try:
        repl_res = python_repl.run(python_repl.sanitize_input(code_string))
        error_flag, error_msg = check_for_errors(repl_res)
        if error_flag:
            logger.warning(f"⚠️ Error in Python Code : {error_msg}")
            fixed_code = handle_repl_error(code=code_string, error_msg=error_msg)
            logger.info(f"✅ Code fixed by agent:\n {fixed_code}")
            return python_repl.run(python_repl.sanitize_input(fixed_code))
        return repl_res
    except Exception as e:
        logger.error(f"❌ ERROR Python REPL Tool: {e}")
        return f"Error running code: {str(e)}"

def create_reform_prompt(question:str,chat_history:str)->str:
    """generats the prompt for question reform"""
    from data.dataloader import db
    # prompt_path = os.path.join(os.getcwd(),"configuration","prompts.yaml")
    # prompt_path = os.path.join(os.getcwd(),"configuration","prompts.yaml")
    prompt_path = os.path.join(os.getcwd(),"configuration","ikea_store_prompts.yaml")
    prompts = read_yaml(file_path=prompt_path)
    meta_data_path=os.path.join(os.getcwd(),"data","metadata_table.csv")
    metadata_content=format_metadata_csv_for_prompt(meta_data_path)
    sytem_msg_prompt = prompts['Prompts']['Reform']['Instructions']
    sytem_msg_prompt = sytem_msg_prompt.replace("{metadata_info}", metadata_content)
    sytem_msg_prompt = sytem_msg_prompt.replace("{db_context}", db.get_context()['table_info'])
    return sytem_msg_prompt.replace("{question}",question).replace("{chat_history}",chat_history)

def create_citations(file_path:str)->str:
    """create SAS link for the cite document"""
    if os.path.exists(file_path) and os.path.isfile(file_path):
        # Initialize the blob service client
        blob_service_client = BlobServiceClient.from_connection_string(os.getenv("BLOB_CONNECTION_STRING"))
        # Create a container with a unique name
        container_name = f"sales-pulse-citations-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        container_client = blob_service_client.create_container(container_name)
        use_by_time = datetime.utcnow() + timedelta(minutes=int(os.getenv("BLOB_EXPIRY_MINS")))
        # Get file name from path
        file_name = os.path.basename(file_path)
        # Upload file to blob storage
        blob_client = container_client.get_blob_client(file_name)
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data)
        # Generate SAS token for the blob
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=container_name,
            blob_name=file_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=use_by_time
        )
        # Create the full URL with SAS token
        sas_url = f"{blob_client.url}?{sas_token}"
        return sas_url