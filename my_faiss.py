from langchain.embeddings import HuggingFaceEmbeddings
import sentence_transformers
from langchain.vectorstores import FAISS

from sentence_transformers import SentenceTransformer
class SentenceTransformerNormalized(SentenceTransformer):
    def encode(self, *args, **kwargs):
        kwargs['normalize_embeddings']=True
        return super().encode(*args, **kwargs)
embeddings = HuggingFaceEmbeddings(model_name=r'D:\ml\text2vec-large-chinese')
embeddings.client=SentenceTransformerNormalized(embeddings.model_name,device="cuda")

def get_bg_hint(context,max_text_len=500,k=25):
    results=get_bg_hint.vector_store.similarity_search_with_relevance_scores(context,k=k)
    all_result_text=''
    for result,score in results:
        knowledge_item='【{0}（{1:.2f}）】'.format(result.page_content,score)
        if len(all_result_text)+len(knowledge_item)+1>max_text_len:
            break
        all_result_text=all_result_text+knowledge_item+'\n'
    return all_result_text
get_bg_hint.vector_store=None