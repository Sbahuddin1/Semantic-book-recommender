# Semantic Book Recommender

This project develops a **semantic book recommender system** powered by large language models (LLMs), vector search, and natural language processing (NLP). It processes book descriptions to recommend books based on semantic similarity, classify genres, and analyze sentiments and emotions, all accessible via an interactive **Gradio dashboard**.

## Project Overview

The Semantic Book Recommender uses advanced NLP techniques to:
- Clean and preprocess book description data.
- Create a vector database for semantic search with LangChain.
- Classify books into genres using zero-shot text classification.
- Analyze sentiments and emotions in book descriptions.
- Deliver personalized book recommendations through a Gradio interface.

## Workflow and Process

The project follows a structured pipeline:

1. **Data Preparation**:
   - Import and clean book descriptions (`books_descriptions.txt`).
   - Address missing data and filter short descriptions (`books_cleaned.csv`).
   - Explore data patterns (`data-exploration.ipynb`).

2. **Vector Search**:
   - Split text with LangChain’s `CharacterTextSplitter` (`vector_search.ipynb`).
   - Build a vector database for semantic similarity.
   - Enable book recommendations based on user queries.

3. **Text Classification**:
   - Apply zero-shot classification using Hugging Face LLMs (`text_classification.ipynb`).
   - Assign genres to books (`books_with_categories.csv`).
   - Assess classification accuracy.

4. **Sentiment and Emotion Analysis**:
   - Perform sentiment analysis with fine-tuned LLMs (`sentiment_analysis.ipynb`).
   - Extract emotions from descriptions (`books_with_emotions.csv`).

5. **Gradio Dashboard**:
   - Develop an interactive UI (`dashboard.py`) for book recommendations.
   - Allow users to input queries and receive tailored suggestions.

## Output

The primary output is a **semantic book recommender system** that:
- Recommends books based on semantic similarity to user queries.
- Displays book genres and emotional insights derived from descriptions.
- Provides an intuitive Gradio dashboard for seamless interaction.

## Installation and Setup

To run the project locally:

### Prerequisites
- Python 3.8+
- Git
- Virtual environment tool (e.g., `venv`)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/semantic-book-recommender.git
   cd semantic-book-recommender
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**:
   - Place `books_descriptions.txt` in the project root or adjust script paths.
   - Generated datasets (`books_cleaned.csv`, etc.) are created during execution.

5. **Launch the Gradio Dashboard**:
   ```bash
   python dashboard.py
   ```
   - Access the dashboard via the local URL (e.g., `http://127.0.0.1:7860`).

6. **Run Notebooks**:
   - Use Jupyter Notebook to explore `data-exploration.ipynb`, `vector_search.ipynb`, `text_classification.ipynb`, or `sentiment_analysis.ipynb`.

### Notes
- Hugging Face LLMs may require API keys. Set them as environment variables.
- Ensure adequate memory for vector database and LLM inference.

## Project Structure

```
semantic-book-recommender/
├── books_cleaned.csv           # Cleaned book data
├── books_descriptions.txt      # Raw book descriptions
├── books_with_categories.csv   # Books with genres
├── books_with_emotions.csv     # Books with emotions
├── dashboard.py                # Gradio dashboard script
├── data-exploration.ipynb      # Data preprocessing
├── sentiment_analysis.ipynb    # Sentiment and emotion analysis
├── text_classification.ipynb   # Genre classification
├── vector_search.ipynb         # Vector search implementation
├── requirements.txt            # Dependencies
├── LICENSE                     # License file
└── README.md                   # This file
```

## Screenshots

[Insert screenshot of Gradio dashboard here]

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Powered by [LangChain](https://www.langchain.com/), [Hugging Face](https://huggingface.co/), and [Gradio](https://www.gradio.app/).
- Inspired by advancements in NLP and semantic search.