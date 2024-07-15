from pprint import pprint

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader('./state of union.txt')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embedding_function = SentenceTransformerEmbeddings(
    model_name='all-MiniLM-L6-v2')

db = Chroma.from_documents(docs, embedding_function)

query = 'Cancer problems'
docs = db.similarity_search_with_relevance_scores(query, k=10)

pprint(docs)
