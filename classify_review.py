"""
CLI tool: Assign a topic to an app review of a puzzle game
Usage:
    python classify_review.py "The puzzles in this game are too easy!"
"""

import argparse
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

# -------- CONFIG --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "labeled_reviews"
CHROMA_PATH = "./chroma_db"
DATA_PATH = "data/puzzle_app_reviews.csv"

# -------- BUILD VECTOR STORE (only runs once) --------
def build_vector_store():
    print("Initializing vector store...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    if len(vector_store.get()["ids"]) == 0:
        df = pd.read_csv(DATA_PATH)
        docs = [
            Document(page_content=row["review"], metadata={"category": row["category"]})
            for _, row in df.iterrows()
        ]
        vector_store.add_documents(docs)
        print("vector store built")
    else:
        print("vector store already populated")

    return vector_store


# -------- RETRIEVAL --------
def get_similar_reviews(vector_store, query, k=5):
    docs = vector_store.similarity_search(query, k=k)
    return " | ".join(
        [f"\"{d.page_content}\" : {d.metadata['category']}" for d in docs]
    )


# -------- CLASSIFICATION --------
def classify_review(review_text, vector_store):
    similar = get_similar_reviews(vector_store, review_text)

    prompt = f"""
    You are tasked with deciding what topic an app review falls into, out of the following categories:
    gameplay, ads, performance, design, features

    Here are some similar reviews and their topic assignmentss:
    {similar}

    Now, assign one of those topics (only one word) to the following review:
    "{review_text}"
    """

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=OPENAI_API_KEY)
    response = llm.invoke(prompt)
    return response.content.strip()


# -------- CLI ENTRY POINT --------
def main():
    parser = argparse.ArgumentParser(description="Classify a puzzle app review.")
    parser.add_argument("review", type=str, help="The app review text.")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        raise ValueError("Please set your OpenAI API key in the OPENAI_API_KEY environment variable.")

    vector_store = build_vector_store()
    topic = classify_review(args.review, vector_store)
    print(f"Category: {topic}")


if __name__ == "__main__":
    main()
