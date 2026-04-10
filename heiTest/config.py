md5_path="./md5.text"
collection_name="rag"
persist_directory="./chroma_db"
chunk_size=1000
chunk_overlap=100
separators=["\n\n","!",".","?","\n","。","！","？"," ",""]
max_split_characters=1000
search_kwargs=2

embeddings_model_name="nomic-embed-text"
chat_model_name="deepseek-r1:7b"
