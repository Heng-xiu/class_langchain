import config
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.document_loaders import UnstructuredURLLoader, SeleniumURLLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores import Chroma

#--------------------------------------------------------
# Step1: 環境設定 (若使用.env本地環境)
config.config_env()


#---------------------------------------
# Step2: 載入文件
#url = f'https://zh.wikipedia.org/zh-tw/%E8%B6%85%E5%B0%8E%E9%AB%94'
url1 = 'https://www.books.com.tw/web/sys_qalist/qa_36_87'   # 博客來的退貨規定
url2 = 'https://www.books.com.tw/web/sys_qalist/qa_36_40/'  # 博客來的換貨規定
url3 = 'https://www.books.com.tw/web/sys_qacontent/qa_36_43'  # 博客來的維修與保固
urls = [url1, url2, url3]   # 更多文建會遇到Token不足的問題, 下一章節將介紹文件分割

loader = UnstructuredURLLoader(urls, continue_on_failure=False)  # 方法1: 需安裝python-magic-bin, urls一定要是陣列
#loader = SeleniumURLLoader(urls, continue_on_failure=False)     # 方法2: 需安裝selenium, urls一定要是陣列,

docs = loader.load()

# 註解 1:
#   UnstructuredURLLoader 若出現以下錯誤, ValueError: Invalid file. The FileType.UNK file type is not supported in partition.
#   請安裝 pip install python-magic-bin

#-------------------------------------
# Step3: 儲存至Vector DataBase (儲存到硬碟)
#    需 pip install chromadb
llm_embedding = OpenAIEmbeddings()
db = Chroma.from_documents(docs, llm_embedding)

#---------------------------------------
# Step4: 建立LLMChain
model_name = "gpt-3.5-turbo-16k"
llm = ChatOpenAI(temperature=0, model=model_name)
#chain = load_qa_chain(llm=llm)  # , chain_type="stuff"
chain = load_qa_with_sources_chain(llm=llm)   # chain_type="stuff"


#---------------------------------------
# Step5: 搜尋與回答
query1 = "博客來的退貨規定?"
docs_search = db.similarity_search(query1, k=1)
response1 = chain.run(input_documents=docs_search, question=query1)
print(response1)
print('========================')



#---------------------------------------
# 參考資料
# 1. https://api.python.langchain.com/en/latest/chains/langchain.chains.qa_with_sources.loading.load_qa_with_sources_chain.html



#---------------------------------------


