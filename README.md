# app-review-rag-demo

---

## Overview

This is a quick, simple impelementation of a RAG pipeline that classifies app reviews into one of a set of predefined categories. 

This simple executable takes a string, representing the text of an app review, and returns a one-word topic as a response. It uses a ChromaDB vector store lookup to find similar reviews and an augmented OpenAI LLM query to make the classification.

Usage:
`python classify_review.py "The puzzles in this game are too easy!"`

---

## Project Structure
```
├── data/
│
├── classify_review.py
│
└── README.md
```

---

## Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language Processing** | langchain, chromadb, openai |
| **Data Handling** | pandas |

---

## Author

Jennifer Podracky

Data Scientist | Product & Program Analytics
