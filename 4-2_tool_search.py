import config
from langchain.tools import DuckDuckGoSearchRun
#from langchain.utilities import SerpAPIWrapper  # 使用Google的搜尋, 需另外申請API

#--------------------------------------------------------
# Step1: 環境設定 (若使用.env本地環境)
config.config_env()


#--------------------------------------------------------
# Step2: 使用工具  (使用DuckDuckGo搜尋)
# https://duckduckgo.com/
# pip install duckduckgo-search==3.8.4
search = DuckDuckGoSearchRun()
result = search.run("台積電")    # 工具幫忙找資料
print(result)



#---------------------------------------
# 參考資料
# 1. https://python.langchain.com/docs/integrations/tools/
# 2. https://python.langchain.com/docs/integrations/tools/ddg
# 3. https://python.langchain.com/docs/integrations/tools/google_serper
# 4. https://python.langchain.com/docs/integrations/tools/gradio_tools

#---------------------------------------


