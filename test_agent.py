import os
from openai import OpenAI
from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv
load_dotenv()

MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")

kimi = LLM(
    model="openai/kimi-k2.6",
    api_key=MOONSHOT_API_KEY,
    base_url="https://api.moonshot.ai/v1",
    # temperature=0.1,
)

researcher = Agent(
    role="Research Analyst",
    goal="Research the requested topic and extract the key facts.",
    backstory="You are a precise technical researcher.",
    llm=kimi,
    verbose=True,
)

task = Task(
    description="""
    Explain why Rust is increasingly used in blockchain infrastructure.
    Focus on performance, memory safety, and concurrency.
    """,
    expected_output="A concise technical explanation.",
    agent=researcher,
)

crew = Crew(
    agents=[researcher],
    tasks=[task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()

print(result)