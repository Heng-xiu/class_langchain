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
model_name = "gpt-3.5-turbo-16k"
llm = ChatOpenAI(temperature=0, model=model_name)

#---------------------------------------
# Step3: 建立自己的工具
#
class BooksSearchTool(BaseTool):
    name = "博客來搜尋工具"
    description = "博客來是一家電商, 可以搜尋書籍及日常用品等"  # 越精準越好

    def _run(self, query):
        print(f'[BooksSearchTool (debug)] 搜尋關鍵字: {query}')
        url = f'https://search.books.com.tw/search/query/key/{query}'
        urls = [url]
        loader = UnstructuredURLLoader(urls=urls, continue_on_failure=False)
        # loader = SeleniumURLLoader(urls, continue_on_failure=False)
        documents = loader.load()

        if len(documents[0].page_content)>4096:
            return documents[0].page_content[:4096]
        else:
            return documents[0].page_content


    def _arun(self):
        raise NotImplementedError("This tool does not support async")



#--------------------------------------------------------
# Step4: Agent結合工具
tools = [BooksSearchTool()]

agent = initialize_agent(
            tools=tools,
            llm=llm,
            verbose=True)

result = agent.run("請推薦五本關於python的書籍")
print('==================================')
print(result)




#---------------------------------------
# 參考資料
# 1. https://docs.langchain.com/docs/components/agents/




#---------------------------------------


