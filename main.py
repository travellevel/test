from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriver

model=OllamaLLM(model="deepseek-r1:7b")

template="""
你是一个点评美食餐厅的专家

这里是一些近期的汇报:{reviews}

这是需要回答的问题:{question}
"""

prompt=ChatPromptTemplate.from_template(template)

chain=prompt | model

while True:
    print("\n\n-----------------------------")
    question=input("提出你的问题(按q退出):")
    print("\n\n")
    if (question=="q"):
        break

    reviews=retriver.invoke(question)
    res=chain.invoke({"reviews":reviews,"question":question})
    print(res)


