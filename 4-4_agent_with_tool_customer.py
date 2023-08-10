import config
from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.document_loaders import UnstructuredURLLoader, SeleniumURLLoader

#--------------------------------------------------------
# Step1: 環境設定 (若使用.env本地環境)
config.config_env()


#---------------------------------------
# Step2: 建立LLMChain
llm = ChatOpenAI(temperature=0)

#---------------------------------------
# Step3: 建立自己的工具
#
# ACS的Yahoo拍賣
#    https://tw.mall.yahoo.com/store/ACS%20%E8%B7%A8%E9%81%8B%E5%8B%95:pump306
#
class ACSSearchTool(BaseTool):
    name = "ACS產品搜尋工具"
    description = "ACS是一家運動鞋電商, 可尋找各廠牌各式的鞋子, 請用中文搜尋"   # 越精準越好

    def _run(self, query):
        print(f'[ACSSearchTool (debug)] 搜尋關鍵字: {query}')
        url = f'https://tw.mall.yahoo.com/search?m=search&sid=pump306&q={query}&search_type=product_name'
        urls = [url]
        # loader = UnstructuredURLLoader(urls=urls, continue_on_failure=False)
        loader = SeleniumURLLoader(urls, continue_on_failure=False)
        documents = loader.load()
        return documents[0].page_content[:2400]  # model token limit!

    def _arun(self):
        raise NotImplementedError("This tool does not support async")



#--------------------------------------------------------
# Step4: Agent結合工具
tools = [ACSSearchTool()]

agent = initialize_agent(
            tools=tools,
            llm=llm,
            verbose=True)

result = agent.run("請推薦五雙白鞋")
print('==================================')
print(result)




#---------------------------------------
# 參考資料
# 1. https://docs.langchain.com/docs/components/agents/




#---------------------------------------


