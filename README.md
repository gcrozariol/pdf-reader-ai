# PDF Reader with Retrieval-Augmented Generation (RAG)

This project is a PDF Reader with Retrieval-Augmented Generation (RAG) capabilities. It leverages LangChain, OpenAI, and Chroma to load, process, and query PDF documents. The solution includes text splitting, embedding generation, and a question-answering chain to provide context-aware answers from the PDF content.

## Features

- Load and process PDF documents
- Split text into manageable chunks
- Generate embeddings using OpenAI
- Store and retrieve text chunks using Chroma
- Answer questions based on the PDF content

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/pdf-reader-rag.git
    cd pdf-reader-rag
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set your OpenAI API key:
    ```python
    import os
    os.environ["OPENAI_API_KEY"] = "YOUR_OPEN_AI_API_KEY"
    ```

## Usage

1. Load and process a PDF document:
    ```python
    from langchain_community.document_loaders.pdf import PyPDFLoader

    pdf_link = "example.pdf"
    loader = PyPDFLoader(pdf_link, extract_images=False)
    pages = loader.load_and_split()
    ```

2. Split the document into chunks:
    ```python
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(pages)
    ```

3. Save chunks to a vector store:
    ```python
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    embeddings_model = OpenAIEmbeddings()
    db = Chroma.from_documents(
        chunks, 
        embedding=embeddings_model, 
        persist_directory="text_index"
    )
    ```

4. Load the database and create a retriever:
    ```python
    vector_db = Chroma(
        persist_directory="text_index", 
        embedding_function=embeddings_model
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    ```

5. Create a question-answering chain:
    ```python
    from langchain.chains.question_answering.chain import load_qa_chain
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=200)
    chain = load_qa_chain(llm, chain_type="stuff")
    ```

6. Ask questions based on the PDF content:
    ```python
    def ask(question):
        context = retriever.invoke(question)
        answer = (chain({"input_documents": context, "question": question}, return_only_outputs=True))['output_text']
        return answer, context

    user_question = input("User: ")
    answer, context = ask(user_question)
    print("Answer: ", answer)
    ```

## License

This project is licensed under the MIT License.
