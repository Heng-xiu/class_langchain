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
# Step2: 載入文件
#url = f'https://zh.wikipedia.org/zh-tw/%E8%B6%85%E5%B0%8E%E9%AB%94'
url1 = 'https://www.books.com.tw/web/sys_qalist/qa_36_87'   # 博客來的退貨規定
url2 = 'https://www.books.com.tw/web/sys_qalist/qa_36_40/'  # 博客來的換貨規定
url3 = 'https://www.books.com.tw/web/sys_qacontent/qa_36_43'  # 博客來的維修與保固

urls = [url1, url2, url3]   # 使用此

loader = UnstructuredURLLoader(urls, continue_on_failure=False)  # 方法1: 需安裝python-magic-bin, urls一定要是陣列
#loader = SeleniumURLLoader(urls, continue_on_failure=False)     # 方法2: 需安裝selenium, urls一定要是陣列,

docs = loader.load()

# 註解 1:
#   UnstructuredURLLoader 若出現以下錯誤, ValueError: Invalid file. The FileType.UNK file type is not supported in partition.
#   請安裝 pip install python-magic-bin

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size
    chunk_size=2048, #4096,  # 全
    chunk_overlap=200,
    length_function=len,
)

docs_splitter = text_splitter.split_documents(docs)

print(f'原本文件數量: {len(docs)}')
print(f'切割後文件數量: {len(docs_splitter)}' )

#---------------------------------------
# Step3: 建立LLMChain
model_name = "gpt-3.5-turbo-16k-0613"
llm = ChatOpenAI(temperature=0, model=model_name)
chain = load_qa_chain(llm=llm, chain_type="map_reduce")  # "map_reduce" need pip install tiktoken

#---------------------------------------
# Step4: 問答
query1 = "商品保固如何處理?"
response1 = chain.run(input_documents=docs, question=query1)
#qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
print(response1)
print('========================')


query2 = "如何換貨?"
response2 = chain.run(input_documents=docs_splitter, question=query2)
print(response2)
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