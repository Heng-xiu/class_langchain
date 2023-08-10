import config
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

#--------------------------------------------------------
# Step1: 環境設定 (若使用.env本地環境)
config.config_env()

#---------------------------------------
# Step2: 建立LLMChain
#    Chain: 結合提示樣板(prompt_template)和大型語言模型(LLM)
prompt_template = "請提供一項關於{topic}的文章標題?"

llm = ChatOpenAI(temperature=0)
prompt = PromptTemplate.from_template(prompt_template)
llm_chain = LLMChain(
    llm=llm,
    prompt= prompt
)

#---------------------------------------
# Step3: 主題問答
topic_1 = "人工智慧"
response_1 = llm_chain(topic_1)
print(response_1)

#---------------------------------------
# Step4: 了解Prompt細節
prompt_string = prompt.format(topic=topic_1)
print(f'prompt string: {prompt_string}')

prompt_messages = prompt.format_prompt(topic=topic_1).to_messages()
print(f'prompt messages: {prompt_messages}')






#--------------------------------------------------------
# 參考資料
# 1. https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/
