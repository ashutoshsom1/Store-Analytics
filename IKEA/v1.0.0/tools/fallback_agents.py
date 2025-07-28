import re
import langchain
langchain.verbose = False
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('agg')
import matplotlib.pyplot as plt
import streamlit as st
from pydantic import BaseModel, Field
st.set_option('deprecation.showPyplotGlobalUse', False)
from utils.utils import read_yaml
from langchain.tools import Tool
# from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.chat import MessagesPlaceholder
# from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages import AIMessage
from langchain.tools import StructuredTool
from utils.fallback_tools import *
from utils.utils import *
from utils.utils import run_python_code
from utils.utils import format_metadata_csv_for_prompt
from data.dataloader import *
from typing import List, Tuple, Any
from langchain_core.agents import AgentAction, AgentFinish, AgentStep

import os
os.environ['CURL_CA_BUNDLE'] = ''
#os.environ['REQUESTS_CA_BUNDLE'] = ''


description_repl = """
    A Python shell for executing valid Python code. Returns the error message if it fails. Use this to analyze and visualize data using libraries
    like Matplotlib or Seaborn. When creating a chart, ensure the data is sorted by the x-axis for meaningful visualization. For monthly data, always
    sort the DataFrame in calendar order from January to December before plotting. Use appropriate chart types of streamlit (e.g. st.area_chart, st.bar_chart,
    st.line_chart,st.map,st.scatter_chart,st.altair_chart,st.bokeh_chart,st.graphviz_chart,st.plotly_chart,st.pydeck_chart,st.pyplot,st.vega_lite_chart) based
    on the data and user questions. If it returns an error, you must retry with the corrected code using this tool. only generate charts using this tool 
    when specifically asked by the user otherwise do not.
    """ 
# Create a Tool instance
repl_tool = Tool(
    name="Python_REPL",
    description= description_repl,
    func=run_python_code,
    handle_tool_error=True,
    handle_validation_error=True
)

# Tool 1: Store Info
class StoreTopDataSearchInput(BaseModel):
    store_id: int = Field(..., description="Unique identifier for the store")
description = '''
    Retrieves detailed information for a specific store based on the provided store_id.
    To use this function, format your input as:
    {"store_id": <store_id>}
    Example Input:
    {"store_id": 1023}
'''
store_top_data_search = StructuredTool.from_function(
    func=get_top_key_information_of_store,
    name="Get_Top_Key_Information_For_Store",
    description=description,
    # args_schema=StoreTopDataSearchInput,
    handle_tool_error=True,
    handle_validation_error=True
)

# Tool 2: Anomaly Detection
class FindAnomalyInput(BaseModel):
    question: str = Field(..., description="Natural language question to identify anomalies")
description = '''
    Analyzes the input question to identify and retrieve rows containing anomalies from the database.
    The generated query will always include DISTINCT to avoid duplicate results.
    To use this function, format your input as:
    {"question": "<your_question_here>"}
    Example Input:
    {"question": "Which transactions are flagged as anomalous in the last 7 days?"}
'''
find_anomaly = StructuredTool.from_function(
    func=get_anomaly_agent,
    name="Get_Anomaly",
    description=description,
    # args_schema=FindAnomalyInput,
    handle_tool_error=True,
    handle_validation_error=True
    )

# Tool 3: Forecast Retrieval
class FindForecastInput(BaseModel):
    question: str = Field(..., description="Natural language question to retrieve forecast data")
description = '''
    Processes the input question to generate a query that retrieves forecast-related information from the database.
    The query will always include DISTINCT to ensure results are unique.
    To use this function, format your input as:
    {"question": "<your_question_here>"}
    Example Input:
    {"question": "What are the forecasted sales for the next quarter?"}
    '''
find_forecast = StructuredTool.from_function(
    func=get_forecast_details,
    name="Get_Forecast",
    description=description,
    # args_schema=FindForecastInput,
    handle_tool_error=True,
    handle_validation_error=True
    )

# Tool 4: Recommendation Engine
class GetRecommendationInput(BaseModel):
    question: str = Field(..., description="Natural language question to retrieve product/customer recommendations")
description = '''
    Interprets the input question to generate a query that retrieves recommendation-related information from the data.
    The query will always include DISTINCT to ensure the results are unique.
    To use this function, format your input as:
    {"question": "<your_question_here>"}
    Example Input:
    {"question": "What products are recommended for first-time customers?"}
    '''
get_recommndation_agent = StructuredTool.from_function(
    func=get_recommendation_agent,
    name="Get_Recommendation",
    description=description,
    # args_schema=GetRecommendationInput,
    handle_tool_error=True,
    handle_validation_error=True
    )

# Tool 5: Weather Impact
class GetWeatherInput(BaseModel):
    question: str = Field(..., description="Natural language question related to the impact of weather")
description = '''
    Processes weather-related questions to generate a query that retrieves relevant weather data from the weather_sales_data table.
    The query will always include DISTINCT to ensure unique results. JOIN operations may be required to combine data from other tables 
    for accurate and comprehensive answers.
    This function is specifically designed to handle questions involving weather factors and their impact. 
    The response must clearly explain how weather conditions influence the results.
    To use this function, format your input as:
    {"question": "<your_question_here>"}
    Example Input:
    {"question": "How did rainfall impact pizza sales last month?"}
    '''
# get_weather_agent = StructuredTool.from_function(
#     func=get_weather_details,
#     name="Get_Weather_Data",
#     description=description,
#     # args_schema=GetWeatherInput,
#     handle_tool_error=True,
#     handle_validation_error=True
#     )


# Tool 6: Executive Analytics
class ExecutiveAnalyticsInput(BaseModel):
    question: str = Field(..., description="Executive-level analytical question about store performance")

description = '''
    Provides executive-level business intelligence and analytical insights for IKEA store performance.
    This tool handles complex analytical queries including store comparisons, channel analysis, 
    conversion funnels, growth trends, and anomaly detection. Use this for management-level 
    reporting and strategic analysis questions.
    To use this function, format your input as:
    {"question": "<your_executive_question_here>"}
    Example Input:
    {"question": "Which store is performing better - DJA or YAS - and by what margin?"}
'''
executive_analytics_tool = StructuredTool.from_function(
    func=get_executive_dashboard,
    name="Get_Executive_Analytics",
    description=description,
    # args_schema=ExecutiveAnalyticsInput,
    handle_tool_error=True,
    handle_validation_error=True
)

# bind the tools
tools = []
tools.append(retriever_tool)
tools.append(repl_tool)
tools.append(store_top_data_search)
tools.append(executive_analytics_tool)
# tools.append(find_anomaly)
# tools.append(get_recommndation_agent)
# tools.append(find_forecast)
# tools.append(get_weather_agent)


# Get the directory of the current script
base_dir = os.getcwd()
# Construct path to the prompt file
# prompt_path = os.path.join(base_dir, "configuration", "prompts.yaml")
prompt_path = os.path.join(base_dir, "configuration", "ikea_store_prompts.yaml")
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
fallback_agent = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=20,
        max_execution_time=240,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        return_source_documents=True,
        early_stopping_method="force"
    )