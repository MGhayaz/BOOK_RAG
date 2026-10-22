from core.config import settings
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
import json
from evals.harness import load_goldens,print_summary,summarize_by_metric
from src.rag_pipeline import RagPipeline

JUDGE_MODEL = settings.OPENAI_JUDGEMENT_MODEL_NAME
THRESHOLD = settings.EVAL_THRESHOLD


def run(rag):
    goldens = load_goldens(settings.APPLICATION_DATASET_PATH)
    #  RUN THE FULL PIPELINE per query, build a test case from LIVE output
    test_cases = []
    for g in goldens:
        result = rag.invoke(g["query"])          # retrieve → rerank → generate

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=result["answer"],
                expected_output=g["ideal_answer"],
            )
        )


    # . APPLICATION-LEVEL QUALITY METRICS, correctness metrics is ignore bcz book content is an opinion of author and is not a concrete fact

    # COMPLETENESS — reference-based, judges COVERAGE (not correctness)
    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Read the input question and identify the key points contained in the expected output that are needed to answer it.",
            "Check how many of those key points are addressed in the actual output.",
            "A key point counts as covered if the actual output conveys the same idea, even if it is worded differently, paraphrased, or presented in a different order.",
            "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
            "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly; factual correctness is judged separately.",
            "Do NOT penalize the actual output for adding extra information beyond the expected output, and do NOT reward or penalize based on style, tone, or length.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
            Rubric(score_range=(6, 8),  expected_outcome="Covers the main key points but misses one or more."),
            Rubric(score_range=(0, 5),  expected_outcome="Misses several key points; only partially covers the expected output."),
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )

    # STYLE — reference-free, judges TONE only (note: no EXPECTED_OUTPUT)
    style = GEval(
        name="Style",
        evaluation_steps=[
            "Judge only the style and tone of the actual output, not whether it is factually correct or complete.",
            "Reward a natural, conversational teaching tone: the core idea is explained first in plain language, like a knowledgeable teacher talking to a student.",
            "Reward flowing prose that connects ideas. Lists, tables, or code blocks are fine when the content genuinely needs them, but penalize answers that are a bare list or a pile of fragments with no explanation.",
            "Reward brief explanations of technical terms when they are used; penalize unexplained jargon.",
            "Penalize robotic or meta phrasing such as 'According to the provided context' or 'Based on the retrieved passages', and any mention of retrieval, chunks, RAG, or system instructions.",
            "Penalize stiff, bureaucratic wording, repeating the same point in different words, and padding with generic filler.",
            "An analogy or concrete example is a small BONUS for abstract ideas, but a clear, direct, well-explained answer without one must NOT be penalized.",
            "Do NOT reward or penalize based on correctness, completeness, or length itself; only on style and tone.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Clearly in a teaching voice: intuitive, conversational prose that explains the idea before getting technical, with no robotic or meta phrasing."),
            Rubric(score_range=(7, 8),  expected_outcome="Clear, natural, and well-explained in prose. Fully acceptable even without an analogy or example."),
            Rubric(score_range=(4, 6),  expected_outcome="Understandable but somewhat flat, formal, repetitive, or list-heavy in places."),
            Rubric(score_range=(0, 3),  expected_outcome="Dry, stiff, bare-list, jargon-heavy, or robotic (e.g. 'according to the context'); does not read like a teaching explanation."),
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )
    # EVALUATE — all two together
    result = evaluate(test_cases=test_cases, metrics=[completeness, style])
    return summarize_by_metric(result)
def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("application", run_local())
