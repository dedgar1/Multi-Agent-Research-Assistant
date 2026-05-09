import asyncio
import os
import requests
import xml.etree.ElementTree as ET
from typing import List, Sequence

from dotenv import load_dotenv
load_dotenv()

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(
    model="gpt-5-nano",
    api_key=os.getenv("OPENAI_API_KEY")
)

def arxiv_search_tool(query: str, max_results: int = 3) -> str:
    """
    Search papers from arXiv related to the given query.
    Returns a human-readable summary of results.
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }

    # Send a request to the arXiv API
    response = requests.get(base_url, params=params, timeout=10)
    
    if response.status_code != 200:
        return f"Error: Unable to fetch data from arXiv (status {response.status_code})"
    
    # Parse the XML response
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    
    if not entries:
        return f"No papers found for topic '{query}'."
    
    output = [f"Papers for '{query}':"]
    for i, entry in enumerate(entries, start=1):
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        link = entry.find("atom:id", ns).text.strip()
        output.append(f"{i}. {title}\n   Link: {link}\n   Abstract: {summary[:300]}...")
    
    return "\n".join(output)

# Test the function
print(arxiv_search_tool("AI in healthcare"))


# Create the topic refinement agent
topic_agent = AssistantAgent(
    "TopicAgent",
    description="Refines a broad research idea into a specific topic.",
    model_client=model_client,
    system_message="""
    ROLE: You are a Research Topic Refiner.
    TASK: Given a general idea, output one refined research topic.

    OUTPUT FORMAT:
    Refined Research Topic: <Topic>
    End with: TASK_COMPLETED_TOPIC_AGENT
    """
)

paper_agent = AssistantAgent(
    "PaperAgent",
    description="Finds research papers for the given topic.",
    tools=[arxiv_search_tool],  # Already created in Task 2
    model_client=model_client,
    system_message="""
    ROLE: You are a Paper Discovery Agent.
    TASK: Find 2–3 recent research papers related to the topic.

    OUTPUT FORMAT:
    ### Topic: <Topic>
    - Paper 1: <Title> — <Summary> - <URL>
    - Paper 2: <Title> — <Summary> - <URL>
    """
)

insight_agent = AssistantAgent(
    "InsightAgent",
    description="Summarizes key insights from discovered papers.",
    model_client=model_client,
    system_message="""
    ROLE: You are an Insight Synthesizer.
    TASK: Provide 3–5 bullet points summarizing common trends or findings.

    OUTPUT FORMAT:
    Key Insights:
    - <Insight 1>
    - <Insight 2>
    """
)

report_agent = AssistantAgent(
    "ReportAgent",
    description="Compiles a concise research report.",
    model_client=model_client,
    system_message="""
    ROLE: You are a Report Compiler.
    TASK: Create a mini research report.

    OUTPUT FORMAT:
    ## Research Report
    ### Introduction
    ...
    ### Key Findings
    ...
    ### Discussion
    ...
    ### Conclusion
    ...
    """
)
gap_agent = AssistantAgent(
    "GapAnalysisAgent",
    description="Performs research gap analysis and suggests future directions.",
    model_client=model_client,
    system_message="""
    ROLE: You are a Research Gap Analyst.
    TASK: Identify 2–3 research gaps and 2–3 future directions.

    OUTPUT FORMAT:
    Research Gaps:
    - <Gap 1>
    - <Gap 2>

    Future Directions:
    - <Direction 1>
    - <Direction 2>

    FINAL REPORT
    """
)

async def main():
    topic_result = await topic_agent.run(task="AI in healthcare")
    print(topic_result.messages[-1].content)

    paper_result = await paper_agent.run(task=topic_result.messages[-1].content)
    print(paper_result.messages[-1].content)

    insight_result = await insight_agent.run(task=paper_result.messages[-1].content)
    print(insight_result.messages[-1].content)

    report_result = await report_agent.run(task=insight_result.messages[-1].content)
    print(report_result.messages[-1].content)

    gap_result = await gap_agent.run(task=report_result.messages[-1].content)
    print(gap_result.messages[-1].content)

asyncio.run(main())