import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai.flow.flow import Flow, listen, start
from dotenv import load_dotenv


load_dotenv()

MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")

if not MOONSHOT_API_KEY:
    raise RuntimeError("MOONSHOT_API_KEY is not set.")


llm = LLM(
    model="openai/kimi-k2.6",
    api_key=MOONSHOT_API_KEY,
    base_url="https://api.moonshot.ai/v1",
)


# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

interviewer = input("Enter the name of the interviewer: ")
company = input("Enter the name of the company: ")
job_position = input("Enter the job position: ")
job_description = input("Enter the job description: ")


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

interviewer_agent = Agent(
    role=f"You are {interviewer}, who is the CEO of the company and is hiring for a seasoned AI Expert",
    goal=(
        f"Conduct a rigorous interview for the {job_position} role. "
        "Ask questions that test technical depth, system design, engineering "
        "judgment, and first-principles reasoning. "
        f"The job description is: {job_description}"
    ),
    backstory=(
        f"{interviewer} is an AI Expert and a serial entrepreneur. "
        "They are a systems and first-principles thinker. They previously also "
        "held senior management positions at large MNCs. They are also an "
        "expert at Python and Rust programming. You are simulating an interview "
        "being conducted by them."
    ),
    llm=llm,
)


coach_agent = Agent(
    role="interview_coach",
    goal=(
        f"Help the candidate prepare for {job_position} with description: "
        f"'{job_description}' by grading the relevance of the candidate's "
        "answer and offering suggestions for improvement."
    ),
    backstory="An expert at interview prep for tech roles",
    llm=llm,
)


# ---------------------------------------------------------------------------
# CrewAI Flow
#
# The Flow owns the deterministic human-in-the-loop sequence:
#
# interviewer agent -> question -> human answer -> coach agent -> feedback
# ---------------------------------------------------------------------------


class InterviewFlow(Flow):

    @start()
    def ask_interview_question(self):
        interview_prep_task = Task(
            description=(
                f"Interview the candidate for the job {job_position} "
                f"with the job description: {job_description}"
            ),
            expected_output=(
                f"Ask only one question to the candidate that is relevant "
                f"for the job {job_position}"
            ),
            agent=interviewer_agent,
        )

        interview_crew = Crew(
            agents=[interviewer_agent],
            tasks=[interview_prep_task],
            verbose=True,
            process=Process.sequential,
        )

        result = interview_crew.kickoff()
        question = result.raw

        # Persist the interviewer output in Flow state.
        self.state["question"] = question

        print("\n" + "=" * 80)
        print("INTERVIEW QUESTION")
        print("=" * 80)
        print(question)

        return question

    @listen(ask_interview_question)
    def collect_candidate_answer(self, question):
        print("\n" + "=" * 80)
        candidate_answer = input("ENTER YOUR ANSWER:\n> ").strip()

        self.state["candidate_answer"] = candidate_answer

        return {
            "question": question,
            "candidate_answer": candidate_answer,
        }

    @listen(collect_candidate_answer)
    def coach_candidate(self, interview_response):
        question = interview_response["question"]
        candidate_answer = interview_response["candidate_answer"]

        coaching_task = Task(
            description=(
                "Give the candidate feedback on the relevance and quality of "
                "their answer, with suggestions for improvement.\n\n"
                f"ROLE: {job_position}\n"
                f"JOB DESCRIPTION: {job_description}\n\n"
                f"INTERVIEW QUESTION:\n{question}\n\n"
                f"CANDIDATE ANSWER:\n{candidate_answer}"
            ),
            expected_output=(
                "A structured and concise response highlighting what they got "
                "right, where they can improve, and any other suggestions for "
                "improvement. The response should be in extremely concise "
                "bullet points (not exceeding 5 bullets)."
            ),
            agent=coach_agent,
        )

        coaching_crew = Crew(
            agents=[coach_agent],
            tasks=[coaching_task],
            verbose=True,
            process=Process.sequential,
        )

        result = coaching_crew.kickoff()
        feedback = result.raw

        self.state["feedback"] = feedback

        print("\n" + "=" * 80)
        print("COACHING FEEDBACK")

if __name__ == "__main__":
    print("Starting interview flow...")
    flow = InterviewFlow()
    flow.kickoff()