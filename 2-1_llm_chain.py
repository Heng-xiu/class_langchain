import config
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

#--------------------------------------------------------
# Step1: 環境設定
config.config_env()

#---------------------------------------
# Step2: 建立LLMChain
#    Chain: 結合提示樣板(prompt_template)和大型語言模型(LLM)
prompt_template = "請提供一項關於{topic}的文章標題?"

llm = ChatOpenAI(temperature=0)
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)



#---------------------------------------
# Step3: 不同的主題問答
topic_1 = "人工智慧"
response_1 = llm_chain(topic_1)
print(response_1)

topic_2 = "超導體"
response_2 = llm_chain(topic_2)
print(response_2)


#--------------------------------------------------------
# 參考資料
# 1. https://python.langchain.com/docs/modules/chains/foundational/llm_chain

#--------------------------------------------------------
# 應用:
#   方面使用在不同的產品, 主題等
#
