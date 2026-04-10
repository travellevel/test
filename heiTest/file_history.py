import os
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.messages import message_to_dict
from langchain_core.messages import messages_from_dict
import json

from typing import Sequence


def get_file_history(session_id):
    return FileHistory(session_id,storage_path="./chat_history")


class FileHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id=session_id
        self.storage_path=storage_path
        self.file_path=os.path.join(self.storage_path,self.session_id)
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)

    def add_message(self,messages: Sequence[BaseMessage]) -> None:
        all_messages=list(self.messages)
        all_messages.extend(messages)


        new_messages=[message_to_dict(message) for message in all_messages]
        with open(self.file_path,'w',encoding='utf-8') as f:
            json.dump(new_messages,f)

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path,'r',encoding='utf-8') as f:
                messages_data=json.load(f)
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []
        

    def clear(self) -> None:
        with open(self.file_path,'w',encoding='utf-8') as f:
            json.dump([],f)