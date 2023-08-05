import config
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader, SeleniumURLLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

#--------------------------------------------------------
# Step1: 環境設定
config.config_env()


#---------------------------------------
# Step2: 載入文件
#url = f'https://zh.wikipedia.org/zh-tw/%E8%B6%85%E5%B0%8E%E9%AB%94'
url1 = 'https://www.books.com.tw/web/sys_qalist/qa_36_87'   # 博客來的退貨規定
url2 = 'https://www.books.com.tw/web/sys_qalist/qa_36_40/'  # 博客來的換貨規定
url3 = 'https://www.books.com.tw/web/sys_qacontent/qa_36_43'  # 博客來的維修與保固
urls = [url3]
#urls = [url1, url2, url3]   # 更多文建會遇到Token不足的問題, 下一章節將介紹文件分割

loader = UnstructuredURLLoader(urls, continue_on_failure=False)  # 方法1: 需安裝python-magic-bin, urls一定要是陣列
#loader = SeleniumURLLoader(urls, continue_on_failure=False)     # 方法2: 需安裝selenium, urls一定要是陣列,

docs = loader.load()

# 註解 1:
#   UnstructuredURLLoader 若出現以下錯誤, ValueError: Invalid file. The FileType.UNK file type is not supported in partition.
#   請安裝 pip install python-magic-bin

#---------------------------------------
# Step3: 建立LLMChain
llm = ChatOpenAI(temperature=0)
chain = load_qa_chain(llm=llm)  # , chain_type="stuff"
#chain = load_qa_with_sources_chain(llm=llm, chain_type="stuff")   # chain_type="stuff"

#---------------------------------------
# Step4: 問答
query1 = "商品保固如何處理?"
response1 = chain.run(input_documents=docs, question=query1)
print(response1)
print('========================')


query2 = "如何換貨?"
response2 = chain.run(input_documents=docs, question=query2)
print(response2)
print('========================')

query3 = "介紹一下王淳恆簡歷?"
response3 = chain.run(input_documents=docs, question=query3)
print(response3)
print('========================')


#-------------------------------------


# 註解2:
#   若出現 openai.error.InvalidRequestError: This model's maximum context length is 4097 tokens. However, your messages resulted in 8116 tokens. Please reduce the length of the messages.
#   表示文件所需的token太大
#     1. 將文件分割, 下一個章節會介紹
#     2. 使用token較大的LLM, 例如gpt-3.5-turbo-16k

#---------------------------------------
# 參考資料
# 1. https://python.langchain.com/docs/integrations/document_loaders/url
# 2. https://python.langchain.com/docs/use_cases/question_answering/how_to/question_answering


#---------------------------------------