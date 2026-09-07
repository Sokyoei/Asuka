import os

import pandas as pd
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FileLoadFactory:
    @staticmethod
    def get_loader(filename: str):
        ext = get_file_extension(filename)
        if ext == "pdf":
            return PyPDFLoader(filename)
        elif ext == "docx" or ext == "doc":
            return UnstructuredWordDocumentLoader(filename)
        else:
            raise NotImplementedError(f"file extension {ext} not supported.")


def get_file_extension(filename: str):
    return filename.split(".")[-1]


def load_docs(filename: str) -> list[Document]:
    file_loader = FileLoadFactory.get_loader(filename)
    pages = file_loader.load_and_split()
    return pages


def ask_document(filename: str, query: str) -> str:
    raw_docs = load_docs(filename)
    if len(raw_docs) == 0:
        return "抱歉，文档为空"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=200, length_function=len, add_start_index=True
    )
    documents = text_splitter.split_documents(raw_docs)
    if documents is None or len(documents) == 0:
        return "无法读取文档内容"
    db = Chroma.from_documents(documents, OpenAIEmbeddings(model="text-embedding-ada-002"))
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0, model_kwargs={"seed": 18}), chain_type="stuff", retriever=db.as_retriever()
    )
    response = qa_chain.run(query + "(请用中文回答)")
    return response


def list_files_in_directory(path: str) -> str:
    file_names = os.listdir(path)
    return "\n".join(file_names)


def finish(the_final_answer: str) -> str:
    return the_final_answer


def get_sheet_names(filename: str) -> str:
    excel_file = pd.ExcelFile(filename)
    sheet_names = excel_file.sheet_names
    return f"这是 {filename} 文件的工作表名称：\n\n{sheet_names}"


def get_column_names(filename: str) -> str:
    df = pd.read_excel(filename, sheet_name=0)
    column_names = "\n".join(df.columns.to_list())
    return f"这是 {filename} 文件的第一个工作表列名：\n\n{column_names}"


def get_first_n_rows(filename: str, n: int) -> str:
    result = get_sheet_names(filename) + "\n\n"
    result += get_column_names(filename) + "\n\n"

    df = pd.read_excel(filename, sheet_name=0)
    n_lines = "\n".join(df.head(n).to_string(index=False, header=True).split("\n"))
    result += f"这是 {filename} 文件第一个工作表的前 {n} 行样例：\n\n{n_lines}"
    return result
