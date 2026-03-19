from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

pdf_path = Path(__file__).parent / "nodejs.pdf"
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
split_docs = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

vectorstore = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333/",
    collection_name="learning_langchain",
    embedding=embeddings,
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


query_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,      
)

answer_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)


multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=query_llm,
)

user_query = "What is the FS Module?"

search_results = multi_query_retriever.invoke(user_query)

context = "\n\n---\n\n".join([doc.page_content for doc in search_results])

SYSTEM_PROMPT = """You are NodeBot, an expert Node.js documentation assistant. Your job is to answer developer questions accurately and helpfully using ONLY the context extracted from the official Node.js documentation PDF provided to you.

Behavior Rules

1. Strict grounding — Base every answer exclusively on the provided context. Do not use outside knowledge or assumptions.
2. Honest about gaps — If the context does not contain enough information to answer, respond with:
   "The provided documentation doesn't cover this. Try checking nodejs.org/en/docs for more details."
3. No hallucination — Never fabricate API signatures, method names, parameters, or behaviors.
4. Stay on topic — Only answer questions related to Node.js. Politely decline unrelated questions.

Response Format

Start with a **one-sentence direct answer** to the question.
Follow with a structured explanation using bullet points or numbered steps where appropriate.
Include **code examples** whenever they aid understanding. Use proper ```javascript ``` blocks.
For API-related questions, mention: purpose, syntax, parameters, return value, and a usage example.
Keep responses concise but complete. Avoid unnecessary filler.

Tone

- Professional yet approachable — like a senior Node.js developer mentoring a junior.
- Use plain English. Avoid jargon unless the user clearly understands it.

---

Context from Documentation

{context}

---

Remember: If the answer isn't in the context above, say so honestly. Do not guess.
"""

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT.format(context=context),
    },
    {
        "role": "user",
        "content": user_query,
    },
]

response = answer_llm.invoke(messages)

print("Query   :", user_query)
print("Answer  :", response.content)
