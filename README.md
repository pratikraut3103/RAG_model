
# RAG Model for German Contracts

This project involves creating a Retrieval-Augmented Generation (RAG) model that reads German contracts and extracts data from them. The model uses various natural language processing (NLP) techniques to translate, process, and retrieve information from PDF documents.

## Prerequisites

- Python 3.6 or higher
- Necessary API keys for Google Gemini and Groq

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/pratikraut3103/RAG_model.git
    cd RAG_model
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `creds.json` file in the root directory of your project and add your API keys:

    ```json
    {
        "GOOGLE_API_KEY": "your_gemini_api_key",
        "groq_api_key": "your_groq_api_key"
    }
    ```

## Usage

### Main Script

The main script provides functionality to either generate embeddings from a PDF and store them or retrieve answers from already generated embeddings. 

