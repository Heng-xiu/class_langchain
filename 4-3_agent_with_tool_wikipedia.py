import config
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

#--------------------------------------------------------
# Step1: 環境設定 (若使用.env本地環境)
config.config_env()


#---------------------------------------
# Step2: 建立LLMChain
llm = ChatOpenAI(temperature=0)

#--------------------------------------------------------
# Step3: Agent結合工具
#tools = load_tools(['wikipedia'], llm=llm)
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [wikipedia]
agent = initialize_agent(
            tools=tools,
            llm=llm,
            verbose=True)


result = agent.run("台積電的摘要, 請用中文回答")
print(result)




#---------------------------------------
# 參考資料
# 1. https://docs.langchain.com/docs/components/agents/
# 2. https://python.langchain.com/docs/integrations/tools/wikipedia
#


#---------------------------------------


