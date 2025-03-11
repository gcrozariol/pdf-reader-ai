#!/usr/bin/env python
# coding: utf-8
import os
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders.pdf import PyPDFLoader

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Model Loading (Embeddings & LLM)

embeddings_model = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=200)

def loadData():
  # Load PDF
  pdf_link = "limiters.pdf"
  loader = PyPDFLoader(pdf_link, extract_images=False)
  pages = loader.load_and_split()

  # Separate document in chunks
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 4000,
    chunk_overlap = 20,
    length_function = len,
    add_start_index = True,
  )

  chunks = text_splitter.split_documents(pages)
  vector_db = Chroma.from_documents(chunks, embedding=embeddings_model)

  # Load retriever
  retriever = vector_db.as_retriever(search_kwargs={"k": 3})

  return retriever

def getRelevantDocs(question):
  retriever = loadData()
  context = retriever.invoke(question)

  return context

def ask(question, llm):
  TEMPLATE = """
  You are an expert in the audio engineering field. You are asked the following question: "{question}". Please provide a detailed answer to the question with the following context: "{context}".
  """

  prompt = PromptTemplate(input_variables = ['context', 'question'], template = TEMPLATE)
  sequence = RunnableSequence(prompt | llm)
  context = getRelevantDocs(question)

  response = sequence.invoke({ 'context': context, 'question': question })

  return response

def handler(event, context):
  body = json.loads(event.get('body', {}))
  question = body.get('question')
  response = ask(question, llm).content

  return {
    "statusCode": 200,
    "headers": {
      "Content-Type": "application/json",
    },
    "body": json.dumps({
      "message": "Success",
      "data": response
    }),
  }