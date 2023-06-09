from langchain.embeddings import HuggingFaceEmbeddings
import sentence_transformers
from langchain.vectorstores import FAISS
import math

from sentence_transformers import SentenceTransformer
class SentenceTransformerNormalized(SentenceTransformer):
    def encode(self, *args, **kwargs):
        kwargs['normalize_embeddings']=True
        return super().encode(*args, **kwargs)
embeddings = HuggingFaceEmbeddings(model_name=r'D:\ml\text2vec-large-chinese')
embeddings.client=SentenceTransformerNormalized(embeddings.model_name,device="cuda")

def relevance_score_fn_fixed(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # The 'correct' relevance function
    # may differ depending on a few things, including:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
    # - embedding dimensionality
    # - etc.
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return min(1,max(0,1.0 - score / math.sqrt(2)))

def get_bg_hint(context,max_text_len=500,k=25,show_score=False):
    get_bg_hint.vector_store.relevance_score_fn=relevance_score_fn_fixed
    results=get_bg_hint.vector_store.similarity_search_with_relevance_scores(context,k=k)
    all_result_text=''
    for result,score in results:
        if show_score:
            knowledge_item='【{0}（{1:.2f}）】'.format(result.page_content,score)
        else:
            knowledge_item='【{0}】'.format(result.page_content)
        if len(all_result_text)+len(knowledge_item)+1>max_text_len:
            break
        all_result_text=all_result_text+knowledge_item+'\n'
    if all_result_text[-1]=='\n':
        all_result_text=all_result_text[:-1]
    return all_result_text
get_bg_hint.vector_store=None