## input_analyzer.py: module A of skill_analyzer_agent - technical skill keywords extraction from inputs

# from openai import OpenAI
from scripts.langchain_llm import invoke_json

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

TEMPLATE_A = load_prompt("prompts/input_analyzer.txt")

class InputAnalyzer:
    """
    Module A: Extract raw skill phrases from JD + User description.
    This class encapsulates keyword extraction logic.
    """

    def __init__(self, client, template=TEMPLATE_A):
        self.client = client
        self.template = template

    def extract_keywords(self, jd_text: str = "", user_desc: str = "") -> dict:
        """
        Extracts required_keywords, preferred_keywords, general_keywords based on JD + user description.
        At least one of jd_text or user_desc must be non-empty.
        """

        if not jd_text and not user_desc:
            raise ValueError("Both jd_text and user_desc are empty. Provide at least one input.")

        # Construct the prompt
        prompt = self.template

        if jd_text.strip():
            prompt += "\n\n### Job Description:\n" + jd_text.strip()

        if user_desc.strip():
            prompt += "\n\n### User Description:\n" + user_desc.strip()

        prompt += "\n\n### JSON Output:\n"

        try:
            data = invoke_json(
                prompt=prompt,
                system_message="You extract and categorize skill keywords with maximum accuracy.",
                temperature=0.0,
                client=self.client,
            )
        except Exception as e:
            print("[ERROR] Could not parse model output as JSON")
            raise e

        return data


# ## Example usage
# def main():

#     jd = """
#     Responsibilities:
#     - Conduct data analysis in LIVE related business, including watching experience, creator ecosystem, agency management, algorithm improvement and etc..
#     - Design metrics framework to measure product healthiness, keep tracking of core metrics and understand root causes of metric movements.
#     - Conduct scientific evaluation with statistical methods, including A/B testing and casual inference.
#     - Identify growth opportunities with data analytics, and drive business decisions. Work with PM/MLE/RD to deliver product and strategy improvement.
#     - Research data science theories and methodology, improve analysis efficiency and data product tools.

#     Minimum Qualifications:
#     - Currently pursuing an Undergraduate/Master in Math, Statistic, Data Science, Machine Learning, or a related technical discipline.
#     - Expertise in SQL and programming in Python or R.
#     - Strong analytical and causal reasoning mindset, and rigidity on statistical correctness. Strong communication and passion about product challenges.

#     Preferred Qualifications:
#     - Experience of LIVE related business.
#     - Knowledge of machine learning and recommendation systems.
#     - Experience with big data technologies.
#     """
#     user = """
#     I really want to enhance my SQL coding ability, especially with window functions and more complex analytical queries. I also need help strengthening my Pandas coding skills, random forest related machine learning knowledge as well.
#     """
#     client = OpenAI()
#     analyzer = InputAnalyzer(client=client, template=TEMPLATE_A)
#     extracted = analyzer.extract_keywords(jd_text=jd, user_desc=user)
#     print(json.dumps(extracted, indent=2))

# if __name__ == "__main__":
#     main()
