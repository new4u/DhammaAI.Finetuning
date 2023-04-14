# %%
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import DirectoryLoader
import jieba as jb
from langchain.chat_models import ChatOpenAI

# %%
files = ['Dhammatest.txt']
# files=['四念住.txt']

for file in files:
    # 读取data文件夹中的中文文档
    my_file = f"./data/{file}"
    with open(my_file, "r", encoding='utf-8') as f:
        data = f.read()

    # 对中文文档进行分词处理
    cut_data = " ".join([w for w in list(jb.cut(data))])
    # 分词处理后的文档保存到data文件夹中的cut子文件夹中
    cut_file = f"./data/cut/cut_{file}"
    with open(cut_file, 'w') as f:
        f.write(cut_data)
        f.close()

# %%
# 加载文档
loader = DirectoryLoader('./data/cut', glob='**/*.txt')
docs = loader.load()
# 文档切块
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
doc_texts = text_splitter.split_documents(docs)
# 调用openai Embeddings
os.environ["OPENAI_API_KEY"] = "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"
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
# 向量化
vectordb = Chroma.from_documents(
    doc_texts, embeddings, persist_directory="./data/cut")
vectordb.persist()
# 创建聊天机器人对象chain,并且返回引用了哪个文档
chain = ChatVectorDBChain.from_llm(OpenAI(
    temperature=0, model_name="gpt-3.5-turbo"), vectordb, return_source_documents=True)


# %%
def get_answer(question):
  chat_history = []
 # 把问题和答案放到chat_history中用于下一次的回答
  # chat_history.append(question)
  # chat_history.append(answer)
  # 调用chain对象，返回答案

  result = chain({"question": question, "chat_history": chat_history});
  # 把答案

  return result["answer"]


# %%
question = '隆波帕默是谁列出所有有关他的生平，用中文回复'
print(get_answer(question)
