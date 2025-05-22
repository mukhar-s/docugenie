import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# 1. Load and extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# 2. Split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)

# 3. Create vector store (embedding + FAISS index)
def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embeddings)

# 4. Build a QA chain
def build_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# 5. Full pipeline
def pdf_qa_pipeline(pdf_path, question):
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)
    vector_store = create_vector_store(chunks)
    qa_chain = build_qa_chain(vector_store)
    return qa_chain.run(question)

# Example usage
if __name__ == "__main__":
    pdf_path = "your_pdf_file.pdf"
    question = "What is the main topic of the document?"
    answer = pdf_qa_pipeline(pdf_path, question)
    print("Answer:", answer)
