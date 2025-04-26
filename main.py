# Import necessary libraries
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
from langchain_core.tools import tool
from tavily import TavilyClient
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict
from dotenv import load_dotenv
from tavily import TavilyClient

# Set up your API keys
load_dotenv()
GOOGLE_API_KEY =""
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Initialize LLM and Tavily search
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
tavily_client = TavilyClient(api_key=tavily_api_key)
search_tool = TavilySearchResults()

# Define the state
class ResearchState(TypedDict):
    question: str
    research_output: str
    final_draft: str

# Define the research node
def research_node(state):
    question = state["question"]

    # Create the Research agent
    research_agent_prompt = hub.pull("hwchase17/openai-functions-agent")
    research_agent = create_openai_functions_agent(llm, [search_tool], research_agent_prompt)
    research_executor = AgentExecutor(agent=research_agent, tools=[search_tool], verbose=True)

    research_result = research_executor.invoke({"input": question})
    return {"research_output": research_result["output"]}

# Define the drafting node
def drafting_node(state):
    info = state["research_output"]
    prompt = PromptTemplate.from_template(
        "You are an expert assistant. Write a clear and readable paragraph or bullet points from the following information, without mentioning sources:\n\n{info}"
    )
    chain = prompt | llm
    response = chain.invoke({"info": info})

    # Extract clean text
    if hasattr(response, 'content'):
        final_text = response.content
    else:
        final_text = str(response)

    return {"final_draft": final_text}

# Define the graph
graph = StateGraph(ResearchState)
graph.add_node("research", research_node)
graph.add_node("draft", drafting_node)
graph.add_edge("research", "draft")
graph.add_edge("draft", END)
graph.set_entry_point("research")

# Compile the graph
chain = graph.compile()

# Streamlit UI
st.title("🔎 Deep Research AI Agent")

user_question = st.text_input("Enter your research question:")

if st.button("Run Research"):
    if user_question:
        with st.spinner("Researching... Please wait..."):
            output = chain.invoke({"question": user_question})
            st.success("Answer:")

            # Display the final draft nicely
            st.write(output["final_draft"])
    else:
        st.warning("Please enter a question first.")
