import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

import gradio as gr
import os

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                          google_api_key=os.getenv("api_key"))
load_dotenv()

books = pd.read_csv("./books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["authors"] = books["authors"].fillna("Unknown Author").astype(str)
books["large_thumbnail"] = np.where(books["large_thumbnail"].isna(), "cover_not_found.jpg", books["large_thumbnail"])

raw_documents = TextLoader('./books_descriptions.txt', encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, embedding= embeddings)



def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_recommendations: int = 50,
    final_recommendations: int = 16
) -> pd.DataFrame:
    
    recs = db_books.similarity_search(query, k=initial_recommendations)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)]

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    tone_column_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Sad": "sadness",
        "Angry": "anger",
        "Suspenseful": "fear"
    }

    if tone in tone_column_map:
        book_recs = book_recs.sort_values(by=tone_column_map[tone], ascending=False)

    return book_recs



def recommend_books(query: str, category: str,
             tone: str) -> list:
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    
    for _, row in recommendations.iterrows():
        try:
            description = row["description"]
            truncated_desc_split = description.split()
            truncated_description = " ".join(truncated_desc_split[:30]) + "..."
            
            authors = row["authors"]
            if pd.isna(authors) or not isinstance(authors, str):
                authors_str = "Unknown Author"
            else:
                authors_split = authors.split(";")
                if len(authors_split) > 2:
                    authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
                elif len(authors_split) == 2:
                    authors_str = f"{authors_split[0]} and {authors_split[1]}"
                else:
                    authors_str = authors
            
            caption = f"{row['title']} by {authors_str}: {truncated_description}"
            results.append((row["large_thumbnail"], caption))
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
                              
tones = ["All"] + [ "Happy", "Surprising", "Sad", "Angry", "Suspenseful"]
                              
    
import gradio as gr

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown(
        """
        # 📚 Semantic Book Recommender
        _Find the perfect book based on your description, category, and mood._
        """,
        elem_id="title"
    )

    with gr.Column():
        user_query = gr.Textbox(
            label="📖 Enter a book description",
            placeholder="e.g., A journey of friendship and adventure",
            lines=2
        )

        with gr.Row():
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="📂 Category",
                value="All"
            )
            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="🎭 Tone",
                value="All"
            )

        submit_button = gr.Button("🔍 Search Recommendations")

    gr.Markdown("## 📘 Recommended Books")

    output = gr.Gallery(
        label="",
        columns=4,
        rows=2,
        height=500,
        object_fit="contain",
        show_label=False
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

        
    
if __name__ == "__main__":
    dashboard.launch()