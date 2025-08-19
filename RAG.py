import os
import PyPDF2
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from config import EMBEDDING_MODEL_ID

load_dotenv()
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")


def file_to_langchain_doc(pdf_path: str) -> list[Document]:
    """
    Converts a physics textbook PDF file to a list of langchain Document objects.

    Args:
        pdf_path (str): The file path to the physics textbook PDF.

    Returns:
        list[Document]: A list of langchain Document objects, each representing a page in the PDF.
    """
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        pages = []

        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            doc = Document(
                page_content=page_text,
                metadata={
                    "source": pdf_path,
                    "page": page_num,
                    "total_pages": len(pdf_reader.pages),
                },
            )
            pages.append(doc)

    return pages


def chunk_langchain_pages(
    pages: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 500,
    add_start_index: bool = True,
) -> list[Document]:
    """
    Splits a list of langchain Document objects into smaller chunks.

    Args:
        pages (list[Document]): List of Document objects to chunk
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Number of characters to overlap between chunks
        add_start_index (bool): Whether to add start index to metadata

    Returns:
        list[Document]: List of chunked Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=add_start_index,
    )
    chunks = splitter.split_documents(pages)
    return chunks


class PhysicsTextbookRAG:
    """Manages physics textbook processing, vector database, and physics concept retrieval."""

    def __init__(
        self,
        num_return_chunks=20,
        chunk_size=1500,
    ):
        self.db: Chroma | None = None
        self.embedding_function = None
        self._total_pages = 0

        self.chunk_size = chunk_size
        self.num_return_chunks = num_return_chunks

        self.client = InferenceClient(api_key=HF_API_TOKEN)
        self.embedding_function = HuggingFaceEndpointEmbeddings(
            model=EMBEDDING_MODEL_ID,
            huggingfacehub_api_token=HF_API_TOKEN,
        )

    def set_total_pages(self, total_pages: int):
        self._total_pages = total_pages

    def load_textbook(self, pdf_path: str) -> dict:
        """Load and embed a physics textbook from the file system."""
        try:
            pages = file_to_langchain_doc(pdf_path)
            return self.embed_pdf(pages)
        except FileNotFoundError:
            return {"success": False, "error": f"File not found: {pdf_path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def embed_pdf(self, pages: list[Document]) -> dict:
        """Embed a list of langchain Document objects into a vector database."""
        try:
            # Chunk the content of the pdf
            chunks = chunk_langchain_pages(pages)

            # Embed the chunks to create the vector database
            self.db = Chroma.from_documents(chunks, self.embedding_function)
            self.set_total_pages(len(pages))

            return {
                "success": True,
                "pages": len(pages),
                "message": "Physics textbook processed successfully",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def semantic_search(
        self, physics_query: str, k: int = None
    ) -> str:
        """Search for physics concepts and relevant context in the textbook database."""

        if self.db is None:
            raise ValueError("No physics textbook has been processed yet")

        if k is None:
            k = self.num_return_chunks

        # Search the textbook for relevant physics content
        results = self.db.similarity_search_with_relevance_scores(physics_query, k=k)

        retrieval = "\n\n---\n\n".join(
            [
                f"[Page {page.metadata.get('page', 'N/A') + 1}]\n{page.page_content}"
                for page, _ in results
            ]
        )

        return retrieval

    def answer_physics_question(
        self, question: str, model_id: str, return_retrieval: bool = False
    ) -> str:
        """Generate an answer to a physics question using relevant textbook content."""
        context = self.semantic_search(question, k=self.num_return_chunks)

        PROMPT_TEMPLATE = """
        You are a helpful and knowledgeable assistant, specialising in Physics. Below is a multiple-choice question and relevant textbook context. Provide a chain of thought as you solve the problem, stating any relevant principles, concepts or equations from the provided context. After this thinking, state the correct answer to the question based on the index of the correct option.

        It is essential that your final answer index is formatted as "Answer: <index>", with no additional text or punctuation, and that your total response is less than 512 characters.

        Textbook Context:
        {context}

        Question:
        {question}

        """

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, question=question)

        response = self.client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            seed=123
        )

        if return_retrieval:
            return response.choices[0].message.content or "", context
        else:
            return response.choices[0].message.content or ""

   

