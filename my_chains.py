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
        return ['json', 'error', 'response']
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
            return {'json':None, 'error':e, 'response':response}
        return {'json':json_object, 'error':None, 'response':response}


class Text2TextChain(Chain):
    llm:object=None
    @property
    def input_keys(self):
        return ['query']
    @property
    def output_keys(self):
        return ['response']
    prompt='''{query}'''
    def _call(self,inputs:dict)->dict:
        query=self.prompt.format(**inputs)
        response=self.llm(query)
        return {'response':response}


class ChunkSummaryChain(Chain):
    llm:object=None
    max_background_len:int=300
    max_retrospect_len:int=100
    @property
    def input_keys(self):
        return ['text']
    @property
    def output_keys(self):
        return ['summary','characters_summary']
    prompt='''可能有关的背景信息（不要加入到回复中）：
{background}
前情提要（不要加入到回复中）：{retrospect}
请阅读下面的小说片段，请列举其中出现的所有人名和他们的身份，然后概括小说的情节。
不要加入自己的想法和推测，请忠实于原著。不要加入没有意思的套话，只要总结出小说中的主要内容即可。
不要重复前情提要。
格式如下。
人物：A（反派），B（帮助主角，又叫C，原名D）
概要：主角在A的帮助下，打败了B，然后和C在一起了
--------------------
{text}
'''
    def _call(self,inputs:dict)->dict:
        inputs={k:v for k,v in inputs.items() if v is not None and v!=''}
        inputs['retrospect']=inputs.get('retrospect','（无）')[:self.max_retrospect_len]
        inputs['background']=inputs.get('background','')[:self.max_background_len]
        query=self.prompt.format(**inputs)
        print('requesting summary')
        response=self.llm(query)
        try:
            characters_summary=response.split('人物：')[1].split('概要：')[0].replace('/n','').strip()
            summary=response.split('概要：')[1].replace('/n','').strip()
            return {'summary':summary, 'characters_summary':characters_summary}
        except Exception as e:
            print(e)
            print(repr(response))
            return {'summary':None, 'characters_summary':None}
        

class RefineChunkSummaryChain(Text2JsonChain):
    llm:object=None
    max_background_len:int=300
    max_retrospect_len:int=100
    max_characters_summary_len:int=100
    max_summary_len:int=200
    @property
    def input_keys(self):
        return ['text','summary','characters_summary']
    @property
    def output_keys(self):
        return ['summary','characters']+super().output_keys
    prompt:str='''可能有关的背景信息（不要加入到回复中）：
{background}
前情提要（不要加入到回复中）：{retrospect}
--------------------
{text}
概要：{summary}
人物：{characters_summary}
--------------------
上面的概要大概率有误，请修改上面的概要，并输出为json格式：
{{
    "summary": "主角在A的帮助下，打败了B，然后和C在一起了",
    "characters":[
        {{"name":"A","description":"反派，过去和A有纠葛","action":"帮助主角准备物资"}}
    ]
}}
'''
    def _call(self,inputs:dict)->dict:
        inputs={k:v for k,v in inputs.items() if v is not None and v!=''}
        inputs['retrospect']=inputs.get('retrospect','（无）')[:self.max_retrospect_len]
        inputs['background']=inputs.get('background','（无）')[:self.max_background_len]
        inputs['characters_summary']=inputs.get('characters_summary','（无）')[:self.max_characters_summary_len]
        inputs['summary']=inputs.get('summary','（无）')[:self.max_summary_len]
        
        print('requesting RefineChunkSummaryChain')
        outputs=super()._call(inputs)
        if outputs['json'] is not None:
            outputs['summary']=outputs['json']['summary']
            outputs['characters']=outputs['json']['characters']
        return outputs
        

        
        
class Story2JsonChain(Text2JsonChain):
    max_background_len:int=350
    max_summary_len:int=150
    max_characters_summary_len:int=50
    max_retrospect_len:int=100
    prompt:str='''
提取对话say，内心独白think，行为act，标注主语char，推测表情expression，标注背景信息或事件info
输出格式：[
    {{"info":"小明是个中学生，很笨拙"}},
    {{"info":"这是一个晴朗的早上，小明看到了一只小猫，小猫有着一双美丽的大眼睛"}},
    {{"char":"小明", "expression":"惊讶", "say":"我看到了什么？！"}},
    {{"char":"小明", "think":"我打不过这只猫，我需要保护自己"}},
    {{"char":"小明", "act":"慌张地逃跑"}},
    {{"info":"结果小明没站稳，摔倒了"}}
]
say，think，act，要求照抄原文，并标注清楚char。不要遗漏任何一条对话和行为。
只能有say,think,act,info,expression,char这几种标注，不要加入其他标注。
info不要进行总结，而是要原汁原味保留原文。
确保json格式正确。
有些字被打错成了发音相似的字，请修正。
不加自己的推测想法，请忠实于原著。不要加入没有意思的套话。请不要把角色名搞混淆。请不要把顺序弄乱
可能有关的背景信息：（不要加入到回复中）
{background}
可能有关的上下文情节：{retrospect} {summary}
可能出现的角色：{characters_summary}
--------------------
{text}
json：'''
    def _call(self,inputs:dict)->dict:
        inputs={k:v for k,v in inputs.items() if v is not None and v!=''}
        inputs['summary']=inputs.get('summary','（无）')[:self.max_summary_len]
        inputs['retrospect']=inputs.get('retrospect','（无）')[:self.max_retrospect_len]
        inputs['background']=inputs.get('background','')[:self.max_background_len]
        inputs['characters_summary']=inputs.get('characters_summary','（无）')[:self.max_characters_summary_len]

        print('requesting Story2JsonChain')
        outputs=super()._call(inputs)
        return outputs



















# class Novel2ScriptChain(Chain):
#     llm:object=None
#     max_bg_hint_len:int=400
#     @property
#     def input_keys(self):
#         return ['text']
#     @property
#     def output_keys(self):
#         return ['script']
#     prompt='''可能有关的背景信息：（不要加入到回复中）
# {background}
# 前情提要：（不要加入到回复中）{retrospect}
# 本章提要：{summary}
# 角色概述：{characters_summary}
# 请将下面的小说片段转换为剧本。
# 不要加入自己的想法，请忠实于原著。请严格照抄原文，不要遗漏任何一行字！
# 在你输出的每一行，请遵循如下格式：
# 角色：对话内容
# 角色：\\me 行为描述
# 角色：（心理活动）
# 【旁白，注释，环境描写等等】
# ----------
# {text}
# 下面是你的回答：
# '''
#     def _call(self,inputs:dict)->dict:
#         inputs={k:v for k,v in inputs.items() if v is not None and v!=''}
#         inputs['retrospect']=inputs.get('retrospect','（无）')
#         inputs['summary']=inputs.get('summary','（无）')
#         inputs['characters_summary']=inputs.get('characters_summary','（无）')
#         inputs['background']=inputs.get('background','')[:self.max_bg_hint_len]
#         query=self.prompt.format(**inputs)
#         print('requesting summary')
#         response=self.llm(query)
#         return {'script':response}
