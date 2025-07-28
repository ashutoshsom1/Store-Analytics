from PIL import Image
import base64
from io import BytesIO
import pandas as pd
from logger import logger
from constants.constant import *
from tools.tools_agents import *
from tools.fallback_agents import fallback_agent
from configuration.config import *
import copy
from langchain_core.messages import HumanMessage


def should_use_fallback(response_output:str)->bool:
    """
    Check if the main agent response indicates a fallback is needed.
    Args:
        response_output (str): The output from the main agent
    Returns:
        bool: True if fallback should be used, False otherwise
    """
    fallback_indicators = [
        "agent stopped due to max iterations",
        "maximum iterations reached",
        "max iterations exceeded",
        "I am currently unable to answer this question, please reframe and retry again",
        "unable to answer this question",
        "please reframe and retry",
        "cannot provide an answer at this time"
    ]
    response_lower = response_output.lower()
    return any(indicator in response_lower for indicator in fallback_indicators)

def call_fallback_agent(question:str,chat_history:str,reason:str):
        """Helper function to call fallback agent with consistent logging."""
        try:
            logger.warning(f"⚠️ {reason}; Calling Fallback Agent")
            fallback_answer = fallback_agent.invoke(
                {
                    "input": HumanMessage(question),
                    "chat_history": chat_history
                }
            )
            logger.info(f"✅ Fallback Agent Response: {fallback_answer['output']}")
            return fallback_answer
        except Exception as fallback_e:
            logger.critical(f"❌ Fallback Agent Job Failure: {fallback_e}")

def process_query(
        prompt:str,
        chat_history:str
)->dict:
    """process a single user query based on chat history"""
    answer = {} #create an empty instance of the answer dict
    logger.info(f"📑 Chat History Received: {chat_history}")
    try:
        reformed_question = llm.invoke(
        create_reform_prompt(
            question=prompt,
            chat_history=chat_history
            )
        ).content
        logger.info(f"✅ Question Reformed: {reformed_question}")
        pattern = r"No task — user is only greeting\."
        match = re.search(pattern,reformed_question)
        if match:
            logger.warning(f"⚠️ Reform Agent Hallucinated; passing down the source question")
            reformed_question = copy.deepcopy(prompt) 
    except Exception as e:
        logger.error(f"❌ Error: Question Reforming Failed: {e}")
        reformed_question = copy.deepcopy(prompt)
    logger.info(f"🤖 Reformed Question By Agent: {reformed_question}")

    """
        Agent Invocation Step with the Reformed Question with Action Plan
    """
    try:
        logger.info(f"📑 Chat History: {chat_history}")
        answer = agent_executor.invoke(  
            {
                "input": HumanMessage(reformed_question),
                "chat_history": chat_history
            }
        )
        logger.info(f"✅ Main Agent Response: {answer.get('output','None')}")
        if should_use_fallback(answer['output']):
            return call_fallback_agent(
                question=reformed_question,
                chat_history=chat_history,
                reason=answer.get('output','None')
            )
    except Exception as e:
            return call_fallback_agent(
                question=reformed_question,
                chat_history=chat_history,
                reason=answer.get('output',e)
            )
    return answer
