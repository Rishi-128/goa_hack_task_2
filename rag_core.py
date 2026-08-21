from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os
from groq import Groq
from langchain_groq import ChatGroq
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from ragas import evaluate
try:
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


print("Packages imported successfully!!")
load_dotenv()
api = os.getenv('GROQ_API_KEY') 
class RagPipeline():
  def __init__(self):
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    self.reranker = CrossEncoder(
        'BAAI/bge-reranker-base',
        max_length = 512)
    self.api = api
    self.client = Groq(
        api_key= self.api
    )
    self.chunks = None
    self.texts= None

    self.index = None
    self.bm25 = None
    self.history=[]
    self.eval_data =[]

  def loading(self):
    loader = DirectoryLoader(
          "./agent/data",
          glob= '*.txt',
          loader_cls = TextLoader,
          loader_kwargs = {"encoding": "utf-8"}
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
    chunk_size = 650,
    chunk_overlap = 50
    )

    self.chunks = splitter.split_documents(docs)
    self.texts = [i.page_content for i in self.chunks]
    emb = self.model.encode(self.texts)
    emb=np.array(emb).astype('float32')
    dim = emb.shape[1]
    self.index = faiss.IndexFlatL2(dim)
    self.index.add(emb)

    tokenize = [
        text.split() for text in self.texts
    ]
    self.bm25 = BM25Okapi(tokenize)

  def query_embedding(self,query):
    query_emb = self.model.encode([query])
    query_emb = np.array(query_emb).astype('float32')
    return query_emb

  def search_index(self,original_query,query_emb,top_k):
    distances,indices = self.index.search(query_emb,top_k)
    indices = indices[0]
    bm_score = self.bm25.get_scores(original_query.split())
    bm_indices = np.argsort(bm_score)[-top_k:][::-1]
    hybrid_indices = list(set(list(indices)+list(bm_indices)))

    return hybrid_indices

  def reranking_content(self,original_query,hybrid_indices):
    pairs = [
        [original_query,self.chunks[i].page_content] for i in hybrid_indices
    ]

    reranking = self.reranker.predict(pairs,batch_size=1)
    reranking= sorted(
        zip(reranking,hybrid_indices),
        key = lambda x: x[0],
        reverse = True
    )

    return reranking

  def retrieval(self,reranking,original_query,ground_truth):

    top_3 = reranking[:3]
    retrieved_chunks_list = [
    self.chunks[idx].page_content
    for score, idx in top_3
]
    # retrieved_chunks_list=[
    #     self.chunks[idx].page_content
    #     for score,idx in top_3
    # ]
    retrived_chunks = "\n\n".join(
    retrieved_chunks_list
)

    prompt = f"""
      Answer ONLY using the provided context.

      Context:
      {retrived_chunks}

      Question:
      {original_query}

      If answer is not found in context,
      say:
      "Content not found"

      Generate:
      Answer:
      <answer>

      Summary:
      <summary>
      (Summary should be max 30 tokens whereas answer should be max 170 tokens)
      """

    response = self.client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role":'user',
                "content":prompt
            }
        ],
        temperature = 0,
        max_tokens=200
    )
    response_text = response.choices[0].message.content

    parts = response_text.split("Summary:")

    answer = parts[0].replace(
        "Answer:",
        ""
    ).strip()

    summary = parts[1].strip()

    print(answer)
    self.history.append({
    "query": original_query,
    "summary": summary,
    "answer": answer
    })

    self.eval_data.append({
    "question": original_query,
    "answer": answer,
    "contexts": retrieved_chunks_list,
    "ground_truth": ground_truth
})
    return answer

  def hyde_retriving(self,query):
    prompt = f"""
    Answer the following question :
    question : {query}
    """

    response = self.client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role":'user',
                "content":prompt
            }
        ],
        temperature = 0,
        max_tokens = 200
    )

    return response.choices[0].message.content

  def regenerate_query(self,original_query):
   prompt = f"""
      Conversation History:
      {self.history}

      Current User Question:
      {original_query}

      Rewrite the question into a standalone,
      fully self-contained query.

      Do NOT answer the question.

      If no rewrite is needed,
      return the original query.
      """

   response = self.client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role":'user',
                "content":prompt
            }
        ],
      temperature = 0,
      max_tokens = 30
      )
   print(response.choices[0].message.content)
   return response.choices[0].message.content

  def ragas_eval(self):
    evaluator_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=self.api,
        temperature=0
    )
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    data = {
     "question": [],
     "answer": [],
     "contexts": [],
     "ground_truth": []
    }

    for i in self.eval_data:
       data['question'].append(i['question'])
       data['answer'].append(i['answer'])
       data['contexts'].append(i['contexts'])
       data['ground_truth'].append(i['ground_truth'])

    dataset = Dataset.from_dict(data)

    ragas_results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    print(ragas_results)

  def llm(self,query):
    prompt = f"""
    Answer the query in most consise manner and total amount of tokens is 50 so keep in mind and answer the query
    Query : {query}
    """
    response = self.client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role":'user',
                "content":prompt
            }
        ],
      temperature = 0.2,
      max_tokens = 50
      )
    return response.choices[0].message.content
  def run(self,original_query,ground_truth , hyde_re=False):
    # self.history.append(original_query)
    rewritten_query = self.regenerate_query(original_query)
    if(hyde_re):
      altered_query = self.hyde_retriving(rewritten_query)
    else:
      altered_query = rewritten_query
    query_emb = self.query_embedding(altered_query)
    hybrid_indices = self.search_index(rewritten_query,query_emb,5)
    reranking = self.reranking_content(rewritten_query,hybrid_indices)


    answer = self.retrieval(reranking,original_query,ground_truth)
    # print("Answer:",answer)
    # self.ragas_eval()
    return answer