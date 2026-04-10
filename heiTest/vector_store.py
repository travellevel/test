from langchain_chroma import Chroma
import config

class VectorStore(object):
    def __init__(self,embedding):
        self.embedding=embedding
        self.vector_store=Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )



    def get_retriever(self):
        retriver=self.vector_store.as_retriever(
            search_kwargs={"k":config.search_kwargs}
        )
        return retriver

