from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from .rag import search_knowledge_base, llm

import json

from typing import Literal
from pydantic import BaseModel, Field



class TriageResult(BaseModel):

    category: Literal[
        "Network",
        "Hardware",
        "Software",
        "Database",
        "Security",
        "Access Management",
        "Email",
        "Other"
    ]

    priority: Literal["P1", "P2", "P3", "P4"]

    assigned_team: str

    retrieved_knowledge: list[str]

    recommended_resolution: str

    requires_human_intervention: bool

    reason: str




@tool
def search_knowledge_base_tool(
    title: str,
    description: str,
    requester_department: str
) -> str:
    """
    Search the IT support knowledge base for relevant troubleshooting
    information based on the ticket details.
    """

    query = f"""
    Title: {title}
    Description: {description}
    Department: {requester_department}
    """

    results = search_knowledge_base(query)

    context = "\n\n".join(
        [
            f"Source: {doc.metadata['source']}\n"
            f"Category: {doc.metadata['category']}\n"
            f"Content: {doc.page_content}"
            for doc in results
        ]
    )

    return context


@tool
def classify_ticket(
    title: str,
    description: str
) -> str:
    """
    Classify an IT support ticket into one of the supported categories.
    """

    prompt = f"""
    Classify the following IT support ticket.

    Title: {title}
    Description: {description}

    Choose exactly one category from:

    Network
    Hardware
    Software
    Database
    Security
    Access Management
    Email
    Other

    Return only the category name.
    """

    response = llm.invoke(prompt)

    return response.content.strip()


@tool
def recommend_priority(
    title: str,
    description: str
) -> str:
    """
    Recommend the priority of an IT support ticket.
    """

    prompt = f"""
    Determine the priority of this IT support ticket.

    Title: {title}
    Description: {description}

    Use these priority levels:

    P1 - Critical:
    Major business disruption, critical systems unavailable,
    or serious security incidents.

    P2 - High:
    Significant impact on work or multiple users affected.

    P3 - Medium:
    Normal operational issue affecting an individual user
    with a reasonable workaround.

    P4 - Low:
    Minor issue, general request, or low-impact inconvenience.

    Consider the urgency, business impact, number of affected users,
    and security risks.

    Return the priority code and a short explanation.
    """

    response = llm.invoke(prompt)

    return response.content.strip()


@tool
def recommend_team(
    category: str,
    title: str,
    description: str
) -> str:
    """
    Recommend the appropriate IT support team for a ticket.
    """

    prompt = f"""
    Recommend the appropriate support team for this IT ticket.

    Category: {category}
    Title: {title}
    Description: {description}

    Consider these support teams:

    Network Team
    Hardware Support Team
    Software Support Team
    Database Team
    Security Team
    Access Management Team
    Email Support Team
    General IT Support Team

    Select the most appropriate team.

    Return the team name and a short explanation.
    """

    response = llm.invoke(prompt)

    return response.content.strip()


tools = [
    search_knowledge_base_tool,
    classify_ticket,
    recommend_priority,
    recommend_team
]

llm_with_tools = llm.bind_tools(tools)

def call_model(state: MessagesState):

    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


graph_builder = StateGraph(MessagesState)

graph_builder.add_node("llm", call_model)

graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "llm")

graph_builder.add_conditional_edges(
    "llm",
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)

graph_builder.add_edge("tools", "llm")

agent = graph_builder.compile()



def generate_structured_result(ticket, messages):

    # Collect tool execution results
    tool_results = []

    for message in messages:

        if message.type == "tool":

            tool_results.append(
                f"Tool: {message.name}\n"
                f"Result: {message.content}"
            )

    # Combine tool results
    context = "\n\n".join(tool_results)

    # Initialize structured LLM
    structured_llm = llm.with_structured_output(TriageResult)

    prompt = f"""
    You are an intelligent IT support ticket triage system.

    Analyze the ticket using the results returned by the agent tools.

    TICKET DETAILS:

    Title: {ticket['title']}

    Description: {ticket['description']}

    Requester Department: {ticket['requester_department']}


    AGENT TOOL RESULTS:

    {context}


    INSTRUCTIONS:

    1. Select the appropriate category from the permitted categories.

    2. Select the priority as P1, P2, P3 or P4.

    3. Recommend the appropriate support team.

    4. Include the relevant retrieved knowledge from the knowledge base.

    5. Provide clear troubleshooting steps based on the retrieved knowledge.

    6. Determine whether human intervention is required.

    7. Explain your decisions in the reason field.

    IMPORTANT:

    - Use the tool results and knowledge-base information as your reference.
    - Do not invent company-specific troubleshooting procedures.
    - Security incidents involving suspected compromise, malware,
      ransomware or data exposure require human security review.
    - If the retrieved information is insufficient, mention this.
    - Do not automatically assign P1 simply because a ticket is urgent.
    """

    result = structured_llm.invoke(prompt)

    return result

def run_ticket_triage(ticket):

    messages = [
        SystemMessage(
            content="""
            You are an intelligent IT support ticket triage agent.

            Analyze the ticket using all four available tools:

            1. Search the knowledge base.
            2. Classify the ticket.
            3. Recommend its priority.
            4. Recommend the appropriate support team.

            Use the tool results to prepare the final analysis.

            Do not invent troubleshooting procedures.
            """
        ),

        HumanMessage(
            content=f"""
            Analyze this IT support ticket:

            Title: {ticket['title']}
            Description: {ticket['description']}
            Department: {ticket['requester_department']}
            """
        )
    ]

    # Execute LangGraph
    response = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 15}
    )

    # Generate structured result
    result = generate_structured_result(
        ticket,
        response["messages"]
    )

    return result.model_dump()



if __name__ == "__main__":

    ticket = {
        "title": "Unable to connect to VPN",
        "description": "VPN authentication failed even though my password is correct. I have an urgent client meeting.",
        "requester_department": "Sales"
    }

    messages = [
        SystemMessage(
            content="""
            You are an intelligent IT support ticket triage agent.

            Use all four available tools:

            1. Search the knowledge base.
            2. Classify the ticket.
            3. Recommend its priority.
            4. Recommend the appropriate support team.

            Use the tool results to analyze the ticket.
            Do not invent troubleshooting procedures.
            """
        ),

        HumanMessage(
            content=f"""
            Analyze this IT support ticket:

            Title: {ticket['title']}
            Description: {ticket['description']}
            Department: {ticket['requester_department']}
            """
        )
    ]

    # Execute LangGraph agent
    response = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 15}
    )

    # Generate structured result
    result = generate_structured_result(
        ticket,
        response["messages"]
    )

    # Print JSON output
    print("\nFINAL TRIAGE RESULT:\n")

    print(
        json.dumps(
            result.model_dump(),
            indent=4
        )
    )


