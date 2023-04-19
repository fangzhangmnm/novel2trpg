chatgpt=None
chatglm_model=None
chatglm_tokenizer=None


def llm_chatgpt(query):
    if not llm_chatgpt.disabled and chatgpt is None:
        load_chatgpt()
    if llm_chatgpt.show_query:
        print('\033[91m'+query+'\033[0m',end='')
    if llm_chatgpt.disabled:
        return ''
    else:

        if llm_chatgpt.max_output_tokens is None:
            # print(llm_chatgpt.max_tokens,chatgpt.get_num_tokens(query))
            max_output_tokens=llm_chatgpt.max_all_tokens-chatgpt.get_num_tokens(query)-8 # magic number of tokens for the internal prompt
        # The maximum number of tokens to generate in the completion. see https://python.langchain.com/en/latest/reference/modules/llms.html
        chatgpt.model_kwargs['max_tokens']=max_output_tokens # doc of langchain is very bad but it works
        chatgpt.model_kwargs['temperature']=llm_chatgpt.temperature 
        response=chatgpt(query)

    if llm_chatgpt.show_response:
        print('\033[94m'+response+'\033[0m')
    return response
llm_chatgpt.show_query=True
llm_chatgpt.show_response=True
llm_chatgpt.disabled=False
llm_chatgpt.temperature=0.01
llm_chatgpt.max_all_tokens=4096
llm_chatgpt.max_output_tokens=None

def llm_chatglm(query):
    if not llm_chatglm.disabled and chatglm_model is None:
        load_chatglm()
    count = 0
    old_length=0
    if llm_chatglm.show_query:
        print('\033[91m'+query+'\033[0m',end='')
    if llm_chatglm.disabled:
        return ''
    else:
        if llm_chatglm.max_output_tokens is None:
            max_output_tokens=llm_chatglm.max_all_tokens-len(chatglm_tokenizer.encode(query))
        else:
            max_output_tokens=llm_chatglm.max_output_tokens
        if llm_chatglm.show_response:
            print('\033[94m',end='')
        for response, history in chatglm_model.stream_chat(chatglm_tokenizer, query, history=[],temperature=llm_chatglm.temperature,max_length=llm_chatglm.max_length):
            if llm_chatglm.show_response:
                print(response[old_length:],end='')
                old_length=len(response)
            count += 1
            if count >= max_output_tokens:
                break
        if llm_chatglm.show_response:
            print('\033[0m')
    return response
llm_chatglm.show_query=True
llm_chatglm.show_response=True
llm_chatglm.disabled=False
llm_chatglm.temperature=0.01
llm_chatglm.max_all_tokens=2048
llm_chatglm.max_output_tokens=None



def load_chatgpt():
    from langchain.llms import OpenAI
    global chatgpt
    chatgpt = OpenAI(openai_api_key=open("C:/Users/15617/.ssh/openai.txt").read().strip(),
                model_name='gpt-3.5-turbo',
                temperature=.01, max_tokens=2048)

def load_chatglm():
    from transformers import AutoTokenizer, AutoModel
    from tqdm.auto import tqdm
    global chatglm_model, chatglm_tokenizer
    chatglm_tokenizer = AutoTokenizer.from_pretrained("D:\ml\chatglm-6b-int4-qe", trust_remote_code=True)
    chatglm_model = AutoModel.from_pretrained("D:\ml\chatglm-6b-int4-qe", trust_remote_code=True).half().cuda()
    chatglm_model = chatglm_model.eval()

    

# class LLM:
#     def __init__(self, llm):
#         self.llm = llm
#         self.show_query=True
#         self.show_response=True
#         self.disabled=False
#     def __call__(self, query):
#         if self.show_query:
#             print('\033[91m'+query+'\033[0m',end='')
#         if self.disabled:
#             return ''
#         else:
#             response=self.llm(query)
#         if self.show_response:
#             print('\033[94m'+response+'\033[0m')
#         return response

# export the function llm

# import openai
# openai.my_api_key=open("C:/Users/15617/.ssh/openai.txt").read().strip()
# def llm(query):
#     response=openai.ChatCompletion.create(model='gpt-3.5-turbo',messages=[{'role':'user','content':query}])['choices'][0]['message']['content']
#     return response