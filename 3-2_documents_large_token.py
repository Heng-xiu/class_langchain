import config
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader, SeleniumURLLoader
from langchain.chains.question_answering import load_qa_chain

#--------------------------------------------------------
# Step1: 環境設定 (若使用.env本地環境)
config.config_env()


#---------------------------------------
# Step2: 載入文件
#url = f'https://zh.wikipedia.org/zh-tw/%E8%B6%85%E5%B0%8E%E9%AB%94'
url1 = 'https://www.books.com.tw/web/sys_qalist/qa_36_87'   # 博客來的退貨規定
url2 = 'https://www.books.com.tw/web/sys_qalist/qa_36_40/'  # 博客來的換貨規定
url3 = 'https://www.books.com.tw/web/sys_qacontent/qa_36_43'  # 博客來的維修與保固
#urls = [url3]   # 短文件
urls = [url1]    # 長文件

loader = UnstructuredURLLoader(urls, continue_on_failure=False)  # 方法1: 需先pip install python-magic-bin, urls一定要是陣列
#loader = SeleniumURLLoader(urls, continue_on_failure=False)     # 方法2: 需先pip install selenium, urls一定要是陣列

docs = loader.load()



#---------------------------------------
# Step3: 建立LLMChain
model_name = "gpt-3.5-turbo-16k"
llm = ChatOpenAI(temperature=0, model=model_name)
chain = load_qa_chain(llm=llm)  # , chain_type="stuff"

#---------------------------------------
# Step4: 問答
query1 = "如何退貨?"
response1 = chain.run(input_documents=docs, question=query1)
print(response1)
print('========================')




