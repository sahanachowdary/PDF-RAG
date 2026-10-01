# PDF-RAG

PDF-RAG is a Retrieval-Augmented Generation (RAG) application built using Python, Streamlit, PyPDF, Sentence Transformers, ChromaDB, and Ollama. The main purpose of this project is to allow users to upload a PDF document and ask questions based on its content.

The application first extracts text from the uploaded PDF using PyPDF. The extracted text is divided into smaller chunks, and each chunk is converted into vector embeddings using the `all-MiniLM-L6-v2` Sentence Transformer model. These embeddings are stored in ChromaDB.

When the user asks a question, the question is also converted into an embedding. ChromaDB performs similarity search and retrieves the most relevant chunks from the uploaded document. These chunks are then combined as context and sent to the Llama 3.2 model through Ollama to generate the final answer.

## Features

- Upload text-based PDF documents
- Extract text from PDFs
- Split large document content into smaller chunks
- Generate embeddings using Sentence Transformers
- Store document chunks and embeddings in ChromaDB
- Perform semantic similarity search
- Retrieve the most relevant chunks for a question
- Generate context-based answers using Ollama
- View the retrieved document chunks used for generating the answer
- Adjust chunk size and number of retrieved chunks using the Streamlit sidebar

## Tech Stack

Python, Streamlit, PyPDF, Sentence Transformers, ChromaDB, Ollama, and Llama 3.2.

## RAG Workflow

PDF Upload → Text Extraction → Text Chunking → Embedding Generation → ChromaDB Storage → User Question → Question Embedding → Similarity Search → Relevant Chunks → Context → Ollama → Generated Answer

## Running the Project

Install the required packages using `py -m pip install -r requirements.txt`, download the Llama 3.2 model using `ollama pull llama3.2`, and start the application using `py -m streamlit run app.py`.

Make sure Ollama is installed and running locally before using the application.

## How to Use

Upload a text-based PDF in the Streamlit application and click **Process & Store PDF**. The application extracts the PDF content, divides it into chunks, creates embeddings, and stores them in ChromaDB.

After the PDF is processed, enter a question related to the document and click **Ask AI**. The application searches ChromaDB for the most relevant chunks and sends them along with the question to Ollama. The generated answer is then displayed along with an option to view the retrieved context.

## Example

If the uploaded PDF contains information such as:

`Students must maintain at least 75% attendance to be eligible for semester examinations.`

The user can ask:

`How much attendance is required?`

The RAG application retrieves the relevant part of the document and generates an answer such as:

`Students must maintain at least 75% attendance.`

## Note

This project currently uses Ollama as a locally running LLM service. Therefore, Ollama and the required `llama3.2` model must be available on the system while running the application.

The application works best with text-based PDFs. Scanned PDFs that contain only images may require OCR before their text can be processed.

## Architecture

PDF → PyPDF → Chunks → Sentence Transformer Embeddings → ChromaDB → Similarity Search → Retrieved Context → Ollama Llama 3.2 → Answer
