import config
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


#--------------------------------------------------------
# Step1: 環境設定
config.config_env()

#---------------------------------------
# Step2: 使用system_message_prompt和human_message_prompt建立LLMChain
#   system_message_prompt: 指定LLM要扮演的角色
#   human_message_prompt: 使用者的提問
#
system_prompt_template = "以下內容若不知道, 請說不知道"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)

human_prompt_template = "請介紹{person}的這個人?"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)

llm = ChatOpenAI(temperature=0)
prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

llm_chain = LLMChain(
    llm=llm,
    prompt= prompt
)

#---------------------------------------
# Step3: 問答
person = "王淳恆"
response_1 = llm_chain({'person': person})
print(response_1)
print('========================')


#--------------------------------------------------------
# 備註:
#    這功能幾乎是必備的!