import os

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



def load_documents(docs_path="docs"):
    """Load all text files from the docs directory."""
    print(f"Loading documents from {docs_path}...")

    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"Directory '{docs_path}' does not exist. "
            "Please create the directory and add text files to it."
        )

    # Load all .txt files using UTF-8 encoding
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8",
            "autodetect_encoding": True
        }
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No text files found in '{docs_path}'. "
            "Please add text files to the directory."
        )

    print(f"\nLoaded {len(documents)} document(s).\n")

    # Display preview of the first two documents
    for i, doc in enumerate(documents[:2]):
        print("=" * 80)
        print(f"Document {i + 1}")
        print("=" * 80)
        print(f"Source : {doc.metadata.get('source')}")
        print(f"Length : {len(doc.page_content)} characters\n")

        print("Preview:")
        print(doc.page_content[:300])
        print("\nMetadata:")
        print(doc.metadata)
        print()

    return documents

def split_documents(documents, chunk_size=500, chunk_overlap=100):
    """Split documents into smaller chunks with overlap."""
    print(f"Splitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",   # Split by paragraphs first
            "\n",     # Then by new lines
            ". ",     # Then by sentences
            "? ",
            "! ",
            " ",      # Then by spaces
            ""        # Finally by characters
        ]
    )

    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    print(f"\nTotal Chunks Created: {len(chunks)}\n")

    # Display first 5 chunks
    for i, chunk in enumerate(chunks[:5]):
        print("=" * 80)
        print(f"Chunk {i + 1}")
        print("=" * 80)
        print(f"Source    : {chunk.metadata.get('source')}")
        print(f"Chunk ID  : {chunk.metadata.get('chunk_id')}")
        print(f"Length    : {len(chunk.page_content)} characters\n")

        print(chunk.page_content)
        print()

    if len(chunks) > 5:
        print(f"...and {len(chunks) - 5} more chunks.\n")

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
        
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    print("Main function started.")

    documents = load_documents("docs")
    
    chunks = split_documents(documents)

    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()