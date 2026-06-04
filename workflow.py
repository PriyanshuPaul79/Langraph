"""
Real-World LangGraph Workflow Examples
======================================

Practical, production-ready workflows for common use cases:
1. RAG-based Question Answering
2. Multi-turn Conversation
3. Document Processing Pipeline
4. Error Handling & Retry Logic
5. Streaming Responses
"""

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from datetime import datetime
import traceback

# ============================================================================
# 1. RAG-BASED Q&A WORKFLOW (Like NiDaan Diagnostic Assistant)
# ============================================================================

class RAGQAState(TypedDict):
    """State for RAG-based Q&A system"""
    query: str
    context_documents: list[str]
    messages: Annotated[list[BaseMessage], add_messages]
    answer: str
    confidence: float
    sources_used: list[str]


def create_rag_qa_workflow():
    """
    RAG (Retrieval Augmented Generation) workflow:
    
    User Query
        ↓
    Retrieve Relevant Documents
        ↓
    Generate Answer with Context
        ↓
    Evaluate Confidence
        ↓
    Return Answer + Sources
    
    Useful for: NiDaan diagnostic system, medical QA, document-based QA
    """
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    # Mock knowledge base (in production, use FAISS/Pinecone)
    knowledge_base = {
        "fever": [
            "High fever (>102°F) in infants <3 months requires immediate medical attention",
            "For children 3-36 months: fever >103°F or lasting >3 days needs evaluation",
            "Check for meningitis signs: neck stiffness, rash, sensitivity to light",
            "Use paracetamol for fever >101°F, max 4 times daily"
        ],
        "cough": [
            "Persistent cough >3 weeks could indicate TB - refer to health facility",
            "Cough with difficulty breathing needs emergency care",
            "Whooping cough: look for paroxysmal cough, use antibiotics if confirmed"
        ],
        "respiratory": [
            "Breathing >50 breaths/min in children needs urgent referral",
            "Check for chest indrawing - sign of pneumonia",
            "Use oxygen if SpO2 <90%"
        ]
    }
    
    def retrieve_documents(state: RAGQAState) -> RAGQAState:
        """Retrieve relevant documents from knowledge base"""
        
        query = state["query"].lower()
        retrieved_docs = []
        
        # Simple keyword matching (replace with FAISS in production)
        for keyword, docs in knowledge_base.items():
            if keyword in query:
                retrieved_docs.extend(docs)
        
        # If no docs found, return general message
        if not retrieved_docs:
            retrieved_docs = ["No specific information found in knowledge base"]
        
        return {
            "context_documents": retrieved_docs[:3],  # Top 3 documents
            "sources_used": [f"KB-{i}" for i in range(len(retrieved_docs[:3]))]
        }
    
    def generate_answer(state: RAGQAState) -> RAGQAState:
        """Generate answer using retrieved context"""
        
        context = "\n".join([f"- {doc}" for doc in state["context_documents"]])
        
        prompt = f"""
        You are a medical assistant helping ASHA health workers.
        
        Question: {state['query']}
        
        Relevant Information:
        {context}
        
        Provide a clear, actionable response. If urgent care is needed, emphasize it.
        Keep language simple for health workers.
        """
        
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        
        return {
            "messages": messages + [response],
            "answer": response.content
        }
    
    def evaluate_confidence(state: RAGQAState) -> RAGQAState:
        """Evaluate confidence in the answer"""
        
        # Count how many documents were used
        num_sources = len(state["context_documents"])
        
        # Check if documents are empty or generic
        has_specific_info = not any("No specific" in doc for doc in state["context_documents"])
        
        # Confidence: 0.0 to 1.0
        if num_sources >= 2 and has_specific_info:
            confidence = 0.9
        elif num_sources == 1 and has_specific_info:
            confidence = 0.7
        else:
            confidence = 0.4
        
        return {
            "confidence": confidence
        }
    
    # Build graph
    workflow = StateGraph(RAGQAState)
    
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("evaluate", evaluate_confidence)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "evaluate")
    workflow.add_edge("evaluate", END)
    
    return workflow.compile()


# ============================================================================
# 2. MULTI-TURN CONVERSATION WORKFLOW
# ============================================================================

class ConversationState(TypedDict):
    """State for multi-turn conversation"""
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    conversation_turn: int
    conversation_summary: str
    should_summarize: bool


def create_conversation_workflow():
    """
    Multi-turn conversation with context preservation:
    
    User Input
        ↓
    Update Messages (preserves history)
        ↓
    Generate Response
        ↓
    (Every 5 turns) Summarize conversation
        ↓
    Response
    
    Benefits:
    - Maintains full conversation history
    - Periodic summarization prevents token overload
    - LLM can reference previous context
    """
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    def process_input(state: ConversationState) -> ConversationState:
        """Add user message to conversation"""
        
        messages = state.get("messages", [])
        user_msg = HumanMessage(content=state["user_input"])
        turn = state.get("conversation_turn", 0) + 1
        
        return {
            "messages": messages + [user_msg],
            "conversation_turn": turn,
            "should_summarize": turn % 5 == 0  # Summarize every 5 turns
        }
    
    def generate_response(state: ConversationState) -> ConversationState:
        """Generate response using all conversation history"""
        
        messages = state.get("messages", [])
        
        # If we have a summary, prepend it for context
        system_message = "You are a helpful assistant."
        if state.get("conversation_summary"):
            system_message = f"""You are a helpful assistant. 
            
Summary of previous conversation:
{state['conversation_summary']}

Use this context when responding."""
        
        # Get response from LLM (has full message history)
        response = llm.invoke(messages)
        
        return {
            "messages": messages + [response]
        }
    
    def should_summarize(state: ConversationState) -> bool:
        """Check if we need to summarize"""
        return state.get("should_summarize", False)
    
    def summarize_conversation(state: ConversationState) -> ConversationState:
        """Create a summary of the conversation so far"""
        
        messages = state.get("messages", [])
        
        # Get last 10 messages for summary
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        
        summary_prompt = f"""
        Summarize this conversation in 2-3 sentences:
        
        {chr(10).join([f'{m.type}: {m.content}' for m in recent_messages])}
        """
        
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        
        return {
            "conversation_summary": response.content
        }
    
    # Build graph
    workflow = StateGraph(ConversationState)
    
    workflow.add_node("process_input", process_input)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("summarize", summarize_conversation)
    
    workflow.set_entry_point("process_input")
    workflow.add_edge("process_input", "generate_response")
    
    # Conditional summarization
    def route_after_response(state: ConversationState) -> str:
        return "summarize" if should_summarize(state) else END
    
    workflow.add_conditional_edges(
        "generate_response",
        route_after_response,
        {"summarize": "summarize", END: END}
    )
    
    workflow.add_edge("summarize", END)
    
    return workflow.compile()


# ============================================================================
# 3. DOCUMENT PROCESSING PIPELINE
# ============================================================================

class DocumentProcessingState(TypedDict):
    """State for document processing"""
    document_path: str
    document_content: str
    chunks: list[str]
    extracted_info: dict
    messages: Annotated[list[BaseMessage], add_messages]
    processing_status: str
    errors: list[str]


def create_document_processing_workflow():
    """
    Document processing pipeline:
    
    Load Document
        ↓
    Chunk Text
        ↓
    Extract Key Information
        ↓
    Store/Index
        ↓
    Return Summary
    
    Useful for: Medical records, reports, knowledge base building
    """
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    def load_document(state: DocumentProcessingState) -> DocumentProcessingState:
        """Load and read document"""
        try:
            # Mock loading (replace with actual file reading)
            content = f"Document from {state['document_path']}"
            return {
                "document_content": content,
                "processing_status": "loaded"
            }
        except Exception as e:
            return {
                "errors": [f"Load error: {str(e)}"],
                "processing_status": "failed"
            }
    
    def chunk_text(state: DocumentProcessingState) -> DocumentProcessingState:
        """Split document into chunks"""
        try:
            content = state.get("document_content", "")
            
            # Simple chunking (use RecursiveCharacterTextSplitter in production)
            chunk_size = 200
            chunks = [
                content[i:i+chunk_size] 
                for i in range(0, len(content), chunk_size)
            ]
            
            return {
                "chunks": chunks,
                "processing_status": "chunked"
            }
        except Exception as e:
            return {
                "errors": state.get("errors", []) + [f"Chunking error: {str(e)}"],
                "processing_status": "failed"
            }
    
    def extract_information(state: DocumentProcessingState) -> DocumentProcessingState:
        """Extract structured information from chunks"""
        try:
            chunks = state.get("chunks", [])
            
            if not chunks:
                return {
                    "extracted_info": {},
                    "processing_status": "no_chunks"
                }
            
            extraction_prompt = f"""
            Extract key information from this document chunk:
            
            {chunks[0]}
            
            Return as JSON with keys: title, key_concepts, entities, summary
            """
            
            response = llm.invoke([HumanMessage(content=extraction_prompt)])
            
            # Parse response (handle potential JSON errors)
            try:
                extracted = json.loads(response.content)
            except:
                extracted = {"raw_extraction": response.content}
            
            return {
                "extracted_info": extracted,
                "processing_status": "extracted",
                "messages": state.get("messages", []) + [
                    HumanMessage(content=extraction_prompt),
                    response
                ]
            }
        except Exception as e:
            return {
                "errors": state.get("errors", []) + [f"Extraction error: {str(e)}"],
                "processing_status": "failed"
            }
    
    # Build graph
    workflow = StateGraph(DocumentProcessingState)
    
    workflow.add_node("load", load_document)
    workflow.add_node("chunk", chunk_text)
    workflow.add_node("extract", extract_information)
    
    workflow.set_entry_point("load")
    workflow.add_edge("load", "chunk")
    workflow.add_edge("chunk", "extract")
    workflow.add_edge("extract", END)
    
    return workflow.compile()


# ============================================================================
# 4. ERROR HANDLING & RETRY LOGIC
# ============================================================================

class RobustWorkflowState(TypedDict):
    """State with error handling"""
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    attempt_count: int
    max_retries: int
    error_log: list[dict]
    response: str
    success: bool


def create_robust_workflow():
    """
    Workflow with error handling and retry logic:
    
    User Input
        ↓
    Try Generate Response
        ↓
    (on error) Check Retry Count
        ↓ (retries remaining)
    Retry with Different Params
        ↓
    (out of retries)
    Return Error Response
    
    Features:
    - Automatic retries
    - Error logging
    - Fallback responses
    - Detailed error information
    """
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    def attempt_generation(state: RobustWorkflowState) -> RobustWorkflowState:
        """Try to generate response with error handling"""
        
        try:
            attempt = state.get("attempt_count", 0) + 1
            temperature = 0.7 - (attempt * 0.1)  # Lower temp on retries
            
            prompt = f"""
            User request: {state['user_input']}
            
            Provide a helpful response.
            """
            
            messages = [HumanMessage(content=prompt)]
            response = llm.invoke(messages)
            
            return {
                "messages": messages + [response],
                "response": response.content,
                "success": True,
                "attempt_count": attempt
            }
        
        except Exception as e:
            error_info = {
                "attempt": state.get("attempt_count", 0) + 1,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
            
            return {
                "error_log": state.get("error_log", []) + [error_info],
                "attempt_count": state.get("attempt_count", 0) + 1,
                "success": False
            }
    
    def should_retry(state: RobustWorkflowState) -> bool:
        """Check if we should retry"""
        attempt = state.get("attempt_count", 0)
        max_retries = state.get("max_retries", 2)
        return attempt < max_retries and not state.get("success", False)
    
    def fallback_response(state: RobustWorkflowState) -> RobustWorkflowState:
        """Provide fallback response"""
        
        fallback_msg = f"""
        I apologize, but I encountered an error processing your request.
        
        Details:
        - Attempts: {state.get('attempt_count', 0)}
        - Last error: {state.get('error_log', [{}])[-1].get('error_type', 'Unknown')}
        
        Your request was: {state['user_input']}
        
        Please try again or contact support.
        """
        
        return {
            "response": fallback_msg,
            "success": False
        }
    
    # Build graph
    workflow = StateGraph(RobustWorkflowState)
    
    workflow.add_node("attempt", attempt_generation)
    workflow.add_node("fallback", fallback_response)
    
    workflow.set_entry_point("attempt")
    
    # Conditional routing for retries
    def route_after_attempt(state: RobustWorkflowState) -> str:
        if state.get("success"):
            return END
        elif should_retry(state):
            return "attempt"  # Retry
        else:
            return "fallback"
    
    workflow.add_conditional_edges(
        "attempt",
        route_after_attempt,
        {"attempt": "attempt", "fallback": "fallback", END: END}
    )
    
    workflow.add_edge("fallback", END)
    
    return workflow.compile()


# ============================================================================
# 5. STREAMING WORKFLOW
# ============================================================================

class StreamingState(TypedDict):
    """State for streaming responses"""
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    response_stream: str


def create_streaming_workflow():
    """
    Workflow with token-level streaming:
    
    Useful for:
    - Real-time UI updates
    - Long responses
    - Chat interfaces
    
    Note: Streaming works best with WebSocket or Server-Sent Events
    """
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    def stream_response(state: StreamingState) -> StreamingState:
        """Generate and stream response"""
        
        messages = [HumanMessage(content=state["user_input"])]
        
        # Collect streamed tokens
        full_response = ""
        
        # Use stream() for token-by-token streaming
        for chunk in llm.stream(messages):
            if hasattr(chunk, 'content'):
                full_response += chunk.content
                # In a real app, you'd send this to a websocket/SSE
                print(chunk.content, end="", flush=True)
        
        return {
            "messages": messages + [AIMessage(content=full_response)],
            "response_stream": full_response
        }
    
    # Build graph
    workflow = StateGraph(StreamingState)
    workflow.add_node("stream", stream_response)
    workflow.set_entry_point("stream")
    workflow.add_edge("stream", END)
    
    return workflow.compile()


# ============================================================================
# HELPER FUNCTIONS FOR TESTING
# ============================================================================

def test_rag_workflow():
    """Test the RAG workflow"""
    print("\n" + "="*70)
    print("RAG Q&A Workflow Test")
    print("="*70)
    
    graph = create_rag_qa_workflow()
    result = graph.invoke({
        "query": "How should I handle high fever in an infant?",
        "context_documents": [],
        "messages": [],
        "answer": "",
        "confidence": 0.0,
        "sources_used": []
    })
    
    print(f"Query: {result.get('query')}")
    print(f"Answer: {result.get('answer')}")
    print(f"Confidence: {result.get('confidence'):.2%}")
    print(f"Sources: {result.get('sources_used')}")


def test_conversation_workflow():
    """Test multi-turn conversation"""
    print("\n" + "="*70)
    print("Multi-Turn Conversation Workflow Test")
    print("="*70)
    
    graph = create_conversation_workflow()
    
    # Simulate multiple turns
    state = {
        "messages": [],
        "user_input": "Hello! What is machine learning?",
        "conversation_turn": 0,
        "conversation_summary": ""
    }
    
    result = graph.invoke(state)
    print(f"Turn {result.get('conversation_turn')}")
    print(f"Messages count: {len(result.get('messages', []))}")


def test_robust_workflow():
    """Test error handling"""
    print("\n" + "="*70)
    print("Robust Workflow with Error Handling Test")
    print("="*70)
    
    graph = create_robust_workflow()
    result = graph.invoke({
        "messages": [],
        "user_input": "Tell me about AI",
        "attempt_count": 0,
        "max_retries": 2,
        "error_log": [],
        "response": "",
        "success": False
    })
    
    print(f"Success: {result.get('success')}")
    print(f"Attempts: {result.get('attempt_count')}")
    print(f"Response length: {len(result.get('response', ''))}")


if __name__ == "__main__":
    print(__doc__)
    
    print("\n" + "="*70)
    print("REAL-WORLD LANGGRAPH WORKFLOW EXAMPLES")
    print("="*70)
    
    print("\nTo test workflows, ensure OPENAI_API_KEY is set, then:")
    print("- test_rag_workflow()")
    print("- test_conversation_workflow()")
    print("- test_robust_workflow()")
    
    # Uncomment to test (requires API key)
    # test_rag_workflow()
    # test_conversation_workflow()
    # test_robust_workflow()