# 參考資料
#   https://python.langchain.com/docs/modules/chains/foundational/sequential_chains

import config
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import SimpleSequentialChain


#--------------------------------------------------------
# Step1: 環境設定
config.config_env()

#---------------------------------------
# Step2: 建立SequentialChain
#    結合連續的Chain
prompt_template1 = "請提供一項關於{topic}的文章標題?"
prompt_template2 = "請條列五項標題{title}的文章章節"

llm = ChatOpenAI(temperature=0)
llm_chain1 = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(template = prompt_template1),
    output_key='title'
)
llm_chain2 = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(template = prompt_template2),
    output_key='chapter'
)

sequential_chain = SimpleSequentialChain(chains=[llm_chain1, llm_chain2], verbose=True)

#--------------------------
# Step3: 問答
topic_1 = "人工智慧"
response_1 = sequential_chain.run(topic_1)
print('Anwser: ')
print(response_1)



#--------------------------------------------------------
# 參考資料
# 1. https://python.langchain.com/docs/modules/chains/foundational/llm_chain

#--------------------------------------------------------
# SimpleSequentialChain限制:
#   (1) 只能一個輸入
#   (2) response只輸出最後結果

