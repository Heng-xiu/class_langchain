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

# 嘗試改變model name
#   https://platform.openai.com/docs/models/gpt-3-5
#   https://platform.openai.com/docs/models/gpt-4
#   https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4
model_name = 'gpt-3.5-turbo'             # 內定, 基本
#model_name = "gpt-3.5-turbo-16k"
#model_name = "gpt-3.5-turbo-0613"
#model_name = "gpt-3.5-turbo-16k-0613"   # 首選, Token長, 理解力更佳
#model_name = "gpt-4"
#model_name= "gpt-4-0613"
#model_name= "gpt-4-32k"  # (目前不支援)
#model_name= "gpt-4-32k-0613"  # (目前不支援)


# 嘗試改變temperature
temperature = 0   # 0: 沒有溫度, 改變少
#temperature = 1  # 1: 有溫度, 多變化
llm = ChatOpenAI(temperature=temperature, model=model_name)
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

#---------------------------------------
# Step3: 不同溫度的結果
topic_1 = "人工智慧"
response_1 = llm_chain(topic_1)
print(response_1)

topic_2 = "人工智慧"
response_2 = llm_chain(topic_2)
print(response_2)



#--------------------------------------------------------
# 參考資料
# 1. https://python.langchain.com/docs/modules/chains/foundational/llm_chain

#--------------------------------------------------------
# 補充:
#   1. 還有不同公司的模型使用
#   2. 中文還是OpenAI效果較佳
#   3. 注意資安問題
#   4. 考慮資安問題可以選擇Azure OpenAI的企業版
#      https://python.langchain.com/docs/integrations/chat/azure_chat_openai
#
