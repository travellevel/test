import streamlit as st
from knowledge_base import KnowledgeBase
import time

st.title("知识库更新")

upload_file=st.file_uploader(
    "请上传TXT文件",
    type=['txt'],
    accept_multiple_files=False,

)

if "service" not in st.session_state:   
    st.session_state.service=KnowledgeBase()


if upload_file is not None:
    file_name=upload_file.name
    file_type=upload_file.type
    file_size=upload_file.size/1024


    st.subheader(f"文件名:{file_name}")
    st.write(f"大小:{file_size:.2f} KB|格式:{file_type}")

    text=upload_file.getvalue().decode("gbk")
    with st.spinner("正在上传文件..."):
        time.sleep(1)
        res=st.session_state.service.upload_by_str(text,file_name)
        st.text(res)
