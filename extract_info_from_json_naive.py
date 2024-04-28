from dotenv import load_dotenv
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, ChatMessage


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
selected_model = "gpt-3.5-turbo-0125"
