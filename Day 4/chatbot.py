from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

class ChatState(TypedDict):
    conversation: Annotated[list[BaseMessage], add_messages]
    # basemessage is the one which is inerited by all the message types in humanmessage, ai message and system message
    # here we need a reducer also which will take the conversation and the new message and add it to the conversation and return the updated conversation

def chatNode(state:ChatState):
    conversation = state["conversation"]
    
    result = model.invoke(conversation)
    return {"conversation": [result]}


cp = MemorySaver()
graph = StateGraph(ChatState)
graph.add_node("generate_response", chatNode)
graph.add_edge(START, "generate_response")
graph.add_edge("generate_response", END)

chatbot = graph.compile(checkpointer=cp)
intial_state = {"conversation":[HumanMessage(content="hello")]}
thread_id = '1'
result = chatbot.invoke(intial_state,
                        config={"configurable": {"thread_id": thread_id}})['conversation'][-1].content
print(result)


while True:
    user_input = input("User: ")
    normalized_input = user_input.strip().lower()
    if normalized_input in ['exit', 'quit']:
        print("Exiting the chatbot. Goodbye!")
        break
    
    
    
    bot = chatbot.invoke({"conversation":[HumanMessage(content=user_input)]}, config={"configurable": {"thread_id": thread_id}})
    print("Chatbot:", bot['conversation'][-1].content)


