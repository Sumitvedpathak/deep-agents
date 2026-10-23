import os
import datetime
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_core.tools import tool

load_dotenv()

@tool
def search_doc():
    return null

@tool
def get_doc(id:str):
    return null