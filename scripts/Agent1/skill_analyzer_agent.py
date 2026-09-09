from scripts.Agent1.input_analyzer import InputAnalyzer
from scripts.Agent1.skill_mapper import SkillMapper
from scripts.Agent1.weight_allocator import WeightAllocator


class SkillAnalyzerAgent:
    """
    Agent1: integrates Module A (input analyzer),
            Module B (skill mapper),
            Module C (weight allocator).

    API:
        agent = SkillPlanningAgent(...)
        result = agent.run(jd_text, user_desc, total_questions)
    """

    def __init__(self, client):
        """
        client: OpenAI client instance
        taxonomy_path: path to taxonomy_skills.json
        template_A: keyword extraction prompt
        template_B: mapping prompt template
        """
        self.analyzer = InputAnalyzer(client)
        self.mapper = SkillMapper(client)
        self.allocator = WeightAllocator()

    def run(self, jd_text, user_desc):
        """
        Full pipeline execution:
        A → extract keywords
        B → map to taxonomy
        C → allocate question counts
        """
        # A: extract raw keywords
        extracted = self.analyzer.extract_keywords(jd_text, user_desc)

        # B: map natural keywords to taxonomy
        mapped = self.mapper.map(extracted)

        # C: allocate questions
        weights = self.allocator.allocate_weights(mapped)

        return {
            "extracted": extracted,
            "mapped": mapped,
            "weights": weights
        }
