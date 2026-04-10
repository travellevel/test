from vector_store import VectorStore
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
import config
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from file_history import get_file_history

embedding=OllamaEmbeddings(model=config.embeddings_model_name)
llm=ChatOllama(model=config.chat_model_name)

class RAG(object):
    def __init__(self):
        self.vector_store=VectorStore(
            embedding=embedding
        )
        
        self.promt_template=ChatPromptTemplate.from_messages(
            [
                ("system","你是我的人工智能助手，协助我解答问题。"),
                ("system","并且我提供一些历史记录，你可以根据历史记录进行回答。历史记录如下："),
                MessagesPlaceholder("history"),
                ("human","根据以下检索到的相关信息，结合你的知识，回答用户的问题。如果你无法从以下信息中找到答案，可以直接根据你的知识进行回答。{context}\n\n用户问题：{input}")
            ]
        )

        self.chatmodel=llm

        self.chain=self.get_chain()

    def get_chain(self):
        retriever =self.vector_store.get_retriever()

        def format_documents(docs:list[Document]):
            if not docs:
                return "没有检索到相关信息。"
            
            formatted_str=""
            for doc in docs:
                formatted_str+=f"文档片段:{doc.page_content}\n文档源数据:{doc.metadata}\n"
            return formatted_str
            

        def temp1(value:dict)->str:
            return value["input"]
        def temp2(value):
            new_value={}
            new_value["input"]=value["input"]["input"]
            new_value["context"]=value["context"]
            new_value["history"]=value["input"]["history"]
            return new_value



        chain=(
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(temp1) | retriever | format_documents
            }|RunnableLambda(temp2) | self.promt_template | self.chatmodel |StrOutputParser()
        )

        conversation_chain=RunnableWithMessageHistory(
            chain,
            get_file_history,
            input_messages_key="input",
            history_messages_key="history",
        )



        return conversation_chain
    

if __name__=="__main__":
    session_config = {"configurable": {"session_id": "user1"}}
    
    rag=RAG()
    question="这些sql语句是什么意思？"
    res = rag.chain.invoke({"input": question}, config=session_config)
    print(res)