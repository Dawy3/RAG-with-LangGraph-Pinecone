from typing import Annotated, TypedDict, Sequence

from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph  import StateGraph , START, END
from langgraph.graph.message import add_messages
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os 
import uuid
import tempfile

from dotenv import load_dotenv 
load_dotenv()

app = FastAPI(title="RAG API")
# Openrouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = os.getenv("MODEL_NAME")

# Initiazlize pinecone 
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "modern-rag"

# Initialize sentence-transformers embedding
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
    encode_kwargs={'normalize_embeddings': True}
)

# Get embedding dimension dynamically
sample_embedding = embeddings.embed_query("test")
EMBEDDING_DIMENSION= len(sample_embedding)


# Create index if it doens't exists 
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name= INDEX_NAME,
        dimension= EMBEDDING_DIMENSION,
        metric= "cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
   
# Initialize vector store
vectorstore= PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding= embeddings
)
    

# Define RAG Agent State
class State(TypedDict):
    # add_messages : Handle duplicated and append automatically
    messages : Annotated[Sequence[BaseMessage], add_messages] 
    documents : list[Document]
    query : str
    answer : str
    
def create_rag_graph():
    """RAD with LangGraph - multi-step reasoning"""
    
    llm = ChatOpenAI(
        model= MODEL_NAME,
        api_key= OPENROUTER_API_KEY,
        base_url= OPENROUTER_BASE_URL,
        temperature=0.3
    )
    
    # Node 1 : Query Rewriter
    async def rewrite_query(state:State):
        """Rewrite user query for better retrival"""
        query = state["messages"][-1].content
        
        rewrite_prompt = f"""Given the user query: "{query}"
Rewrite this query to be more specific and suitable for semantic search.
Return only the rewritten querry, nothing else."""

        response = await llm.ainvoke([HumanMessage(content=rewrite_prompt)])
        rewritten= response.content
        
        return {"query": rewritten}
    
    # Node 2: Retriever 
    async def retrieve_document(state:State):
        """Retrieve relevant document"""
        query = state.get("query", state["messages"][-1].content) 
        
        # pinecone retrieval
        docs = vectorstore.similarity_search(
            query= query,
            k=4,
            filter=None # Add metadata filter if needed 
        )
        
        return {"documents": docs}
    
    # Node 3 : Grader
    async def grade_documents(state:State):
        """Grade document relevance"""
        query = state["query"]
        documents = state["documents"]
        
        relevant_docs= []
        
        for doc in documents:
            grade_prompt = f"""Query: {query}
Document: {doc.page_content[:500]}
Is this document relevant to the query? Answer only 'yes' or 'no'."""

            response = await llm.ainvoke([HumanMessage(content=grade_prompt)])
            grade = response.content.strip().lower() # yes/ no

            if grade == 'yes':
                relevant_docs.append(doc)
                
        return {"documents": relevant_docs}
        
    # Node 4: Generator
    async def generate_answer(state:State):
        """Generate Final Answer"""
        query = state["query"]
        documents = state["documents"]
        
        # format context
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ])
        
        generation_prompt = f"""Answer the question based on the following context.
Context:
{context}

Question: {query}

Answer:"""
        
        response = await llm.ainvoke([HumanMessage(content=generation_prompt)])
        answer = response.content
        
        # Add to messages
        return{
            "answer": answer,
            "messages": [AIMessage(content=answer)]
        }        
        
        
    # Build the graph
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("retrieve", retrieve_document)
    graph.add_node("grade", grade_documents)
    graph.add_node("generate", generate_answer)
    
    # add edges
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("grade", "generate")
    graph.add_edge("generate", END)
    
    return graph.compile()


# Initialize the RAG graph
rag_graph = create_rag_graph()


# ---- Pydantic models 
class QueryRequest(BaseModel):
    query : str
    session_id : str  = "default"
    
class QueryResponse(BaseModel):
    query : str
    rewritten_query : str
    answer : str
    source: list[dict]
    num_source_used : int
    
@app.post("/documents/upload")
async def upload_documents(file:UploadFile = File(...)):
    """Upload and process document"""
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False , suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        # Load Document
        loader = PyPDFLoader(tmp_path)
        documents= loader.load()
        
        # split 
        text_splitter=  RecursiveCharacterTextSplitter(
            chunk_size= 1000,
            chunk_overlap = 200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata
        doc_id = str(uuid.uuid4())
        for chunk in chunks:
            chunk.metadata.update({
                "doc_id": doc_id,
                "filename": file.filename
            })
            
            
        # Store in Pinecone 
        vectorstore.add_documents(chunks)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return{
            "doc_id": doc_id,
            "filename": file.filename,
            "chunk_created": len(chunks),
            "status": "sucess"
            }
        
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/query", response_model=QueryResponse)
async def query_rag(request:QueryRequest):
    """Query the RAG System"""
    
    # Run the graph
    result = await rag_graph.ainvoke({
        "messages": [HumanMessage(content=request.query)],
        "documents": [],
        "query": "",
        "answer":""
    })
    
    return QueryResponse(
        query= request.query,
        rewritten_query=result["query"],
        answer= result["answer"],
        source=[
            {
                "content": doc.page_content[:200],
                "filename": doc.metadata.get("filename", "Unknown")
            } 
            for doc in result["documents"]
        ],
        num_source_used= len(result["documents"])
    )
    
    
@app.get("/index/state")
async def get_index_stats():
    """Get Pinecone index statistics"""
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    
    return {
        "total_vectorstore": stats.total_vector_count,
        "dimension": stats.dimension,
        "index_fullness": stats.index_fullness,
        "namespace": stats.namespaces
    }
