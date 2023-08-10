import config
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun

#--------------------------------------------------------
# Step1: 環境設定 (若使用.env本地環境)
config.config_env()


#--------------------------------------------------------
# Step2: 使用工具  (使用維基百科的工具)
# pip install wikipedia==1.4.1
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
result = wikipedia.run("台積電")
print(result)



#---------------------------------------
# 參考資料
# 1. https://python.langchain.com/docs/integrations/tools/
# 2. https://python.langchain.com/docs/integrations/tools/wikipedia
#


#---------------------------------------


