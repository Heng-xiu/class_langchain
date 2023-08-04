# 參考資料
#   https://python.langchain.com/docs/modules/chains/foundational/sequential_chains

import config
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import SequentialChain


#--------------------------------------------------------
# Step1: 環境設定
config.config_env()

#---------------------------------------
# Step2: 建立SequentialChain
#    結合連續的Chain
prompt_template1 = "請提供一項關於{topic}的文章標題?"
prompt_template2 = "請條列{num_title}項標題{title}的文章章節"

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

sequential_chain = SequentialChain(chains=[llm_chain1, llm_chain2],
                                   input_variables=['topic', 'num_title'],
                                   output_variables=['title', 'chapter'],
                                   verbose=True)

#--------------------------
# Step3: 問答
topic_1 = "人工智慧"
num_title = 8
response_1 = sequential_chain({'topic':topic_1, 'num_title':  num_title})
print('[Title]: ')
print(response_1['title'])

print('[Chapter]: ')
print(response_1['chapter'])


#--------------------------------------------------------
# 參考資料
# 1. https://python.langchain.com/docs/modules/chains/foundational/llm_chain

#--------------------------------------------------------
# 應用:
#   LLM分段性工作效果較佳, sequential_chain適合生成一篇長文等工作
#

