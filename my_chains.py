# 一些langchain模板
from langchain.chains.base import Chain
import json
import re


class Text2JsonChain(Chain):
    llm:object=None
    @property
    def input_keys(self):
        return ['text']
    @property
    def output_keys(self):
        return ['content', 'error', 'response']
    prompt='''请总结一下文本到json格式。
输出格式：{{
    "中心思想": "本文的中心思想",
    "主要人物": ["人物1", "人物2"]
}}
请不要加入自己的主观推测。
文本：
{text}
json：'''
    def _call(self,inputs:dict)->dict:
        query=self.prompt.format(**inputs)
        response=self.llm(query)
        # delete the last comma before ]
        response=re.sub(r',\s*\]', r']', response)
        try:
            json_object=json.loads(response)
        except Exception as e:
            print(e,response[:100])
            return {'content':None, 'error':e, 'response':response}
        return {'content':json_object, 'error':None, 'response':response}

