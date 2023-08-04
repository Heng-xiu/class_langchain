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
# Step2: 建立LLMChain
#    Chain: 結合提示樣板(prompt_template)和大型語言模型(LLM)
prompt_template = "請介紹{person}的這個人?"

llm = ChatOpenAI(temperature=0)
prompt = PromptTemplate.from_template(prompt_template)
llm_chain = LLMChain(
    llm=llm,
    prompt= prompt
)

#---------------------------------------
# Step3: 瞎掰的回答
person = "王淳恆"
response_1 = llm_chain(person)
print(response_1['text'])

