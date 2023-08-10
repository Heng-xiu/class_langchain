import config
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

#--------------------------------------------------------
# Step1: 環境設定 (若使用.env本地環境)
config.config_env()

#---------------------------------------
# Step2: 使用system_message_prompt和human_message_prompt建立LLMChain
#   system_message_prompt: 指定LLM要扮演的角色
#   human_message_prompt: 使用者的提問
#
system_prompt_template = "請將{input_language}翻譯成{output_language}."
human_prompt_template = "{text}"

system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)
prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

llm = ChatOpenAI(temperature=0)
llm_chain = LLMChain(
    llm=llm,
    prompt= prompt
)

#---------------------------------------
# Step3: 主題問答
language_1 = "中文"
language_2 = "英文"
text_1 = "人工智慧"
response_1 = llm_chain({'input_language': language_1, 'output_language': language_2, 'text': text_1})
print(response_1)
print('========================')

language_3 = "英文"
language_4 = "中文"
text_2 = "Artificial intelligence"
response_1 = llm_chain({'input_language': language_3, 'output_language': language_4, 'text': text_2})
print(response_1)
print('========================')

#---------------------------------------
# Step4: 了解Prompt細節
prompt_messages = prompt.format_prompt(input_language=language_1, output_language=language_2, text=text_1).to_messages()
print(f'prompt messages: {prompt_messages}')


#--------------------------------------------------------
# 參考資料
# 1. https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/
