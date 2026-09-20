import os
from openai import OpenAI
from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv
load_dotenv()

MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")


client = OpenAI(
    api_key=MOONSHOT_API_KEY,
    base_url="https://api.moonshot.ai/v1",
)

completion = client.chat.completions.create(
    model="kimi-k3",
    messages=[
        {"role": "system", "content": "You are an expert at history, especially pertaining to Asian history."},
        {"role": "user", "content": "What was India's capital city prior to New Delhi? Why was the capital city moved to New Delhi?"}
    ]
)

print(completion.choices[0].message.content)