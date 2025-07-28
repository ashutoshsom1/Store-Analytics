import re
import streamlit as st
import langchain
langchain.verbose = False
from langchain.agents import (
    AgentExecutor, 
    create_openai_tools_agent
    )
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('agg')
import matplotlib.pyplot as plt
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
from utils.utils import read_yaml
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import AIMessage
from langchain.tools import StructuredTool
from utils.tools import *
from utils.utils import *
from data.dataloader import *

import os
os.environ['CURL_CA_BUNDLE'] = ''
#os.environ['REQUESTS_CA_BUNDLE'] = ''


##############################################################################################################################################
##############################################################################################################################################
tools=[]
sql_desc = """
    This tool analyzes user questions to determine the underlying intent—such as forecasting, anomaly detection, 
    impact assessment, recommendations, or complex data exploration. It dynamically generates advanced SQL queries
    to retrieve precise insights, leveraging techniques such as DISTINCT selections, JOINs across multiple tables, 
    Common Table Expressions (CTEs), subqueries, window functions, GROUP BY with HAVING filters, case-based logic, 
    and dynamic pivoting. The tool adapts its query structure based on data relationships and the analytical depth required, 
    and it clearly explains the output to support deeper understanding and decision-making.
    """
sql_tool = StructuredTool.from_function(
    func=create_retriver_for_sql,
    name="sql_tool",
    description=sql_desc
)
##############################################################################################################################################
repl_desc ="""
    A Python shell for executing valid Python code. Returns the error message if it fails. Use this to analyze and visualize data using libraries
    like Matplotlib or Seaborn. When creating a chart, ensure the data is sorted by the x-axis for meaningful visualization. For monthly data, always
    sort the DataFrame in calendar order from January to December before plotting. Use appropriate chart types of streamlit (e.g. st.area_chart, st.bar_chart,
    st.line_chart,st.map,st.scatter_chart,st.altair_chart,st.bokeh_chart,st.graphviz_chart,st.plotly_chart,st.pydeck_chart,st.pyplot,st.vega_lite_chart) based
    on the data and user questions. If it returns an error, you must retry with the corrected code using this tool. only generate charts using this tool 
    when specifically asked by the user otherwise do not.
    """
repl_tool = StructuredTool.from_function(
    name="python_repl",
    description= repl_desc,
    func=run_python_code,
    handle_tool_error=True
)
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
tools.append(sql_tool)
tools.append(retriever_tool)
tools.append(repl_tool)
# Get the directory of the current script
base_dir = os.getcwd()
# Construct path to the prompt file
# prompt_path = os.path.join(base_dir, "configuration", "prompts.yaml")
prompt_path = os.path.join(base_dir, "configuration", "ikea_store_prompts.yaml")
meta_data_path=os.path.join(base_dir,"data","metadata_table.csv")
metadata_content=format_metadata_csv_for_prompt(meta_data_path)
# Read the file
prompts = read_yaml(file_path=prompt_path)
sytem_msg_prompt = prompts['Prompts']['Base']['Instructions']
sytem_msg_prompt = sytem_msg_prompt.replace("{metadata_file}", metadata_content)
sytem_msg_prompt = sytem_msg_prompt.replace("{db_context}", db.get_context()['table_info'])
messages = [
        SystemMessagePromptTemplate.from_template(sytem_msg_prompt),
        HumanMessagePromptTemplate.from_template("{input},{chat_history}"),
        AIMessage(content="{SQL_FUNCTIONS_SUFFIX}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
]
prompt = ChatPromptTemplate.from_messages(messages)
prompt = prompt.partial(**context)
agent = create_openai_tools_agent(llm, tools, prompt)
##############################################################################################################################################
agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=50,
        max_execution_time=300,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        return_source_documents=True,
        early_stopping_method="force"
    )