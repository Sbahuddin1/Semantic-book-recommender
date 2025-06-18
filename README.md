# 📚 Semantic Emotion-Based Book Recommender

An AI-powered book recommendation system that understands your **natural language prompt**, filters results by **category and emotional tone**, and returns personalized book suggestions. Built with **LLMs**, **vector embeddings**, and an interactive **Gradio dashboard**.

---

## 🚀 Overview

This project is a complete pipeline that takes raw book descriptions and builds an intelligent system capable of recommending books based on how you **feel** and what you're **interested** in.

Users can input prompts like:

> “A mysterious and sad story of a lost kingdom.”

And get recommendations that:
- Semantically match the prompt using vector search
- Belong to a chosen **category** (e.g., Romance, Mystery)
- Reflect a desired **tone** (e.g., Happy, Sad, Surprising)

---

## 🧭 Project Workflow

The system was developed in logical stages over time. Below is the high-level flow, also reflected in our development video:

---

## 🛠️ Tools & Technologies Used

| Area | Tools |
|------|-------|
| **Data Processing** | `pandas`, `numpy` |
| **Vector Search** | `LangChain`, `ChromaDB`, `Google Generative AI Embeddings` |
| **Zero-shot Classification** | `Hugging Face Transformers` |
| **Sentiment Analysis** | LLM-based (fine-tuned classification models) |
| **Interface** | `Gradio` |
| **Notebook Work** | Jupyter (`.ipynb`) files for prototyping |
| **Environment** | Python 3.10+, virtualenv |

---

## 📁 Project Structure

```text
Semantic-book-recommender/
├── books_cleaned.csv                  # Cleaned descriptions
├── books_descriptions.txt            # Raw book texts
├── books_with_categories.csv         # With LLM-inferred categories
├── books_with_emotions.csv           # With LLM-inferred emotions
├── dashboard.py                      # Gradio UI and recommender logic
├── data-exploration.ipynb            # Data cleaning and filtering
├── text_classification.ipynb         # Zero-shot category classification
├── sentiment_analysis.ipynb          # Emotion labeling of books
├── vector_search.ipynb               # Embedding + semantic similarity
├── requirements.txt                  # Project dependencies
├── LICENSE
└── README.md
