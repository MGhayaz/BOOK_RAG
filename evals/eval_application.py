from core.config import settings
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
import json

from src.rag_pipeline import RagPipeline

GOLDEN_PATH = settings.APPLICATION_DATASET_PATH
JUDGE_MODEL = settings.OPENAI_JUDGEMENT_MODEL_NAME
THRESHOLD = settings.EVAL_THRESHOLD

with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


#  RUN THE FULL PIPELINE per query, build a test case from LIVE output
rag = RagPipeline()
test_cases = []
for g in goldens:
    result = rag.invoke(g["question"])          # retrieve → rerank → generate

    test_cases.append(
        LLMTestCase(
            input=g["question"],
            actual_output=result["answer"],
            expected_output=g["ideal_answer"],
        )
    )


# . APPLICATION-LEVEL QUALITY METRICS, correctness metrics is ignore bcz book content is an opinion of author and is not a concrete fact

# COMPLETENESS — reference-based, judges COVERAGE (not correctness)
completeness = GEval(
    name="Completeness",
    evaluation_steps=[
        "Identify the key points contained in the expected output.",
        "Check how many of those key points are addressed in the actual output.",
        "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
        "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly — factual correctness is judged separately.",
        "Do NOT penalize the actual output for adding extra information beyond the expected output.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
        Rubric(score_range=(5, 8),  expected_outcome="Covers the main key points but misses one or more."),
        Rubric(score_range=(0, 4),  expected_outcome="Misses several key points; only partially covers the expected output."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,
)

# STYLE — reference-free, judges TONE only (note: no EXPECTED_OUTPUT)
style = GEval(
    name="Style",
    evaluation_steps=[
        "Judge only the teaching style and tone of the actual output, not whether it is factually correct or complete.",
        "Reward an intuitive, explanatory tone: plain language, the idea explained before any formula or jargon, and technical terms briefly unpacked when used.",
        "Reward a direct, conversational register written in prose, as a CampusX lecture would explain it out loud, rather than a dry, formal, or bullet-list tone.",
        "An analogy or concrete example is a BONUS when the concept is abstract, but a clear, direct, well-explained answer is fully acceptable and must NOT be penalized for not having one.",
        "Penalize answers that are stiff, bureaucratic, structured as a bare list with no explanation, or that use unexplained jargon.",
        "Do NOT reward or penalize based on correctness, completeness, or length — only on style and tone.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="Clearly in a teaching voice: intuitive, conversational prose that explains before it formalizes."),
        Rubric(score_range=(7, 8),  expected_outcome="Clear, conversational, and well-explained in prose. Fully acceptable even without an analogy or example."),
        Rubric(score_range=(4, 6),  expected_outcome="Understandable but somewhat flat, formal, or list-heavy in places."),
        Rubric(score_range=(0, 3),  expected_outcome="Dry, stiff, bare-list, jargon-heavy, or robotic; does not read like a teaching explanation."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,
)




# EVALUATE — all two together
evaluate(test_cases=test_cases, metrics=[completeness, style])
