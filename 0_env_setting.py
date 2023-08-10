# 1. pip install -r requirements
# 2. 設定環境變數
#   (1) 本地環境: 新增 .env 的檔案, 內容為OPENAI_API_KEY=sk-xxxxxxxxxxxxxxx
#   (2) CodeSpace: settings->secrets->codespaces中設定OPENAI_API_KEY=sk-xxxxxxxxxxxxxxx
#   (2) replit:  在Secrets設定OPENAI_API_KEY=sk-xxxxxxxxxxxxxxx
#   (3) 其他: 須按照依照不同環境做設定

import os
import pandas as pd
from dotenv import load_dotenv

#-------------------------------------
# 1. 測試基本指令是否成功
data = [1, 2, 3, 4]
data_df = pd.DataFrame({'data': data})
print(data_df)

# 2. 環境變數是否設定成功
load_dotenv()  # take environment variables from .env, replit不需要
my_secret = os.environ['OPENAI_API_KEY']
print(my_secret)
