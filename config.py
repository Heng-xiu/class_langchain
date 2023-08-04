# 設定 OPENAI_API_KEY 在 .env(本地環境) or Secrets(replit)

import os
from dotenv import load_dotenv


def config_env():
    load_dotenv()  # take environment variables from .env, replit不需要
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
