from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from sentence_transformers import CrossEncoder

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10
    }
)


model = ChatOllama(
    model="qwen2.5-coder:3b",
    temperature=0
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Make the question standalone using conversation history
    if chat_history:
        messages = [
            SystemMessage(
                content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."
            ),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Retrieve relevant documents with similarity scores
    results = db.similarity_search_with_relevance_scores(
        search_question,
        k=10
    )

    # Create (question, document) pairs
    pairs = [
        (search_question, doc.page_content)
        for doc, _ in results
    ]

    # Compute reranker scores
    scores = reranker.predict(pairs)

    # Combine documents with reranker scores
    reranked = list(zip(results, scores))

    # Sort by reranker score (highest first)
    reranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Keep only the best 3
    reranked = reranked[:3]

    docs = []

    print("\nTop Documents After Reranking:\n")

    for i, ((doc, vector_score), rerank_score) in enumerate(reranked, 1):

        docs.append(doc)

        print("=" * 80)
        print(f"Document {i}")
        print("=" * 80)

        print(f"Vector Score   : {vector_score:.4f}")
        print(f"Rerank Score   : {rerank_score:.4f}")
        print(f"Chunk ID       : {doc.metadata.get('chunk_id')}")
        print(f"Source         : {doc.metadata.get('source')}")

        print("\nPreview:")
        print(doc.page_content[:300])

        print("\n")

    

    # Step 3: Create prompt
    document_text = "\n".join(
        f"- {doc.page_content}" for doc in docs
    )

    combined_input = f"""
You are a question-answering assistant.

Answer the user's question ONLY using the retrieved documents.

If the answer is not present in the documents, reply exactly:

"I don't have enough information to answer that question based on the provided documents."

Question:
{user_question}

Retrieved Documents:
{document_text}

Answer:
"""

    # Step 4: Generate answer
    messages = [
        SystemMessage(
            content="You answer questions only from the retrieved documents."
        ),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    # Step 5: Save conversation history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    # Optional: keep only the last 10 messages
    chat_history[:] = chat_history[-10:]

    # Step 6: Print answer
    print("\n==============================")
    print("Answer:")
    print(answer)

    print("\nSources:")
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")

        if source not in seen:
            seen.add(source)
            print(f"- {source}")

    print("==============================\n")

    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()

# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"