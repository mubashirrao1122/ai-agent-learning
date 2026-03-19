from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore

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


query_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,      # deterministic reformulations
)

answer_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)


def generate_query_variants(user_query: str, n_queries: int = 3) -> list[str]:
    """Use LLM to create multiple related queries (fan‑out)."""
    system_prompt = (
        "You are a search query rewriter for a vector database.\n"
        f"Given a user question, generate {n_queries} different short search "
        "queries that capture different ways of asking the same thing.\n"
        "Return them as a numbered list, one query per line."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    resp = query_llm.invoke(messages)
    lines = resp.content.splitlines()

    queries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip leading numbers like "1. ", "2) " etc.
        if line[0].isdigit():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                line = parts[1]
        queries.append(line)

    if not queries:
        queries = [user_query]

    return queries[:n_queries]


def reciprocal_rank_fusion(result_lists, k: float = 60.0):
    """
    Apply Reciprocal Rank Fusion over multiple ranked lists.
    result_lists: list of lists; each inner list is [(Document, score), ...]
                  sorted best‑first.[web:2][web:41]
    Returns: list of (Document, fused_score) sorted by fused_score desc.
    """
    fused_scores = defaultdict(float)
    doc_by_key = {}

    for results in result_lists:
        for rank, (doc, _score) in enumerate(results, start=1):
            key = (
                str(doc.metadata.get("id"))
                or f"{doc.metadata.get('source', '')}-{doc.metadata.get('page', '')}-{hash(doc.page_content)}"
            )
            fused_scores[key] += 1.0 / (k + rank)
            if key not in doc_by_key:
                doc_by_key[key] = doc

    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_by_key[key], score) for key, score in sorted_items]



user_query = "What is the FS Module?"


query_variants = generate_query_variants(user_query, n_queries=3)

top_k_per_query = 5
result_lists = []

for q in query_variants:
   
    results = vectorstore.similarity_search_with_score(
        q,
        k=top_k_per_query,
    )
    result_lists.append(results)

fused_results = reciprocal_rank_fusion(result_lists, k=60.0)


final_k = 6
top_docs = [doc for doc, _ in fused_results[:final_k]]

context = "\n\n---\n\n".join(doc.page_content for doc in top_docs)

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

print("Query        :", user_query)
print("Variants     :", query_variants)
print("Answer       :", response.content)
