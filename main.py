import os
import json
import autogen
import pandas as pd
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions

os.environ["AUTOGEN_USE_DOCKER"] = "False"

config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list , "timeout": 60, "temperature": 0}

recur_spliter = RecursiveCharacterTextSplitter(separators=[",","\n", "\r", "\t"])


assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant who answers the user question from the data provided in database adn returns answer only from the database not llm",
    llm_config=llm_config
) 

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key="sk-proj-ShR4UI3S04oz856T635gT3BlbkFJ5elLi7N1PQrQtzVLXgcZ",
                model_name="text-embedding-3-large"
            )

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "qa",
        "docs_path": "C:/Users/AmitAshokraoShelke/OneDrive - Nimble Accounting/D Drive/AutogenSamplecode/20_csv/sample_resumee.csv",
        "embedding_function": openai_ef,
        "custom_text_split_function": recur_spliter.split_text,
        "get_or_create" : True,
        "overwrite" : False,
        "vector_db": "chroma",
        "collection_name" : "abcd",
        "embedding_model" : "text-embedding-3-large",
        "model" : "gpt-4o-mini"
    },
    
 )

# assistant.reset()
# userproxyagent = autogen.UserProxyAgent(name="userproxyagent")
# userproxyagent.initiate_chat(assistant, message="Find the first name of the person with the job title 'Project Manager'.")

# import pandas as pd

# df = pd.read_csv("C:/Users/AmitAshokraoShelke/OneDrive - Nimble Accounting/D Drive/AutogenSamplecode/20_csv/sample_resumee.csv")
# print(df.head())

assistant.reset()
# code_problem = "Find the first name of the person with the job title 'Project Manager'."
chat_result = ragproxyagent.initiate_chat(
    assistant, message=ragproxyagent.message_generator, problem="Find the Job Description of the person with the job title 'Project Manager'.",
)