# 1. pip install -r requirements
# 2. 設定環境變數
#   (1) 本地環境: 新增 .env 的檔案, 內容為OPENAI_API_KEY=sk-xxxxxxxxxxxxxxx
#   (2) CodeSpace: settings->secrets->codespaces中設定OPENAI_API_KEY=sk-xxxxxxxxxxxxxxx
#   (2) replit:  在Secrets設定OPENAI_API_KEY=sk-xxxxxxxxxxxxxxx
#   (3) 其他: 須按照依照不同環境做設定

import os
from dotenv import load_dotenv


def config_env():
    load_dotenv()  # take environment variables from .env, replit不需要
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
