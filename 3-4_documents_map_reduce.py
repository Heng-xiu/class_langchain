import config
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader, SeleniumURLLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

#--------------------------------------------------------
# Step1: 環境設定
config.config_env()


#---------------------------------------
# Step2: 載入文件 (更多文件)
#url = f'https://zh.wikipedia.org/zh-tw/%E8%B6%85%E5%B0%8E%E9%AB%94'
url1 = 'https://www.books.com.tw/web/sys_qalist/qa_36_87'   # 博客來的退貨規定
url2 = 'https://www.books.com.tw/web/sys_qalist/qa_36_40/'  # 博客來的換貨規定
url3 = 'https://www.books.com.tw/web/sys_qacontent/qa_36_43'  # 博客來的維修與保固
#urls = [url3]   # 短文件
#urls = [url1]    # 長文件
urls = [url1, url2, url3]   # 更多文件

loader = UnstructuredURLLoader(urls, continue_on_failure=False)  # 方法1: 需先pip install python-magic-bin, urls一定要是陣列
#loader = SeleniumURLLoader(urls, continue_on_failure=False)     # 方法2: 需先pip install selenium, urls一定要是陣列

docs = loader.load()


#---------------------------------------
# Step3: 建立LLMChain
#model_name = "gpt-3.5-turbo"        # not work, tokens數不夠
model_name = "gpt-3.5-turbo-16k"     # use this
llm = ChatOpenAI(temperature=0, model=model_name)

#chain = load_qa_chain(llm=llm)   # not work, 17248 tokens數
chain = load_qa_chain(llm=llm, chain_type="map_reduce")  # 會先做整合, 7007 tokens數, "map_reduce" 需要先 pip install tiktoken


#---------------------------------------
# Step4: 問答
query1 = "商品保固如何處理?"
response1 = chain.run(input_documents=docs, question=query1)
#qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
print(response1)
print('========================')


query2 = "如何換貨?"
response2 = chain.run(input_documents=docs, question=query2)
print(response2)
print('========================')


#---------------------------------------
# 參考資料
# 1. https://python.langchain.com/docs/modules/chains/document/map_reduce


