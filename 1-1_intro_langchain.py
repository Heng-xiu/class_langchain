# 設定 OPENAI_API_KEY 在 .env(本地環境) or Secrets(replit)

import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

#--------------------------------------------------------
# Step1: 環境設定
load_dotenv()  # take environment variables from .env, replit不需要
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

#--------------------------------------------------------
# Step2: 簡單對話 (常識)
chat_model = ChatOpenAI()
response = chat_model.predict("台積電股票代碼多少?")
print(response)




#------------------------------------------------------
# 參考資料
# 1. https://python.langchain.com/docs/get_started/introduction.html


#--------------------------------------------------------
# 筆記以下兩個模型有什麼不同?
#   from langchain.llms import OpenAI
#   from langchain.chat_models import ChatOpenAI
#
# 參閱:
#   https://platform.openai.com/docs/api-reference/chat
#   from langchain.llms import OpenAI: 是文字接龍的模型
#   from langchain.chat_models import ChatOpenAI: 是對話的模型
#   使用上看應用, 本課程應用大多是都是用對話的模型
#
