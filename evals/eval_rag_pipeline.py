import json
from core.config import settings
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from evals.harness import print_summary,load_goldens,summarize_by_metric
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from src.rag_pipeline import RagPipeline
    # 1. LOAD queries (we only need the queries --- context comes from the pipeline now)
def run(rag):    
    goldens = load_goldens(settings.GENERATOR_DATASET_PATH)
    test_cases = []
    for g in goldens:
        result = rag.invoke(g["query"])          # retrieve → rerank → generate

        test_cases.append(
            LLMTestCase(
                input=result["query"],
                actual_output=result["answer"],       # what the generator produced
                retrieval_context=result["context"],  # what the RETRIEVER returned
            )
        )
        # 3. THE THREE TRIAD METRICS
        metrics = [
            ContextualRelevancyMetric(
            threshold=settings.EVAL_THRESHOLD,
            model=settings.OPENAI_JUDGEMENT_MODEL_NAME,
            include_reason=True,  
        ),
            FaithfulnessMetric(
            threshold=settings.EVAL_THRESHOLD,
            model=settings.OPENAI_JUDGEMENT_MODEL_NAME,
            include_reason=True, 
        ),
            AnswerRelevancyMetric(
            threshold=settings.EVAL_THRESHOLD,
            model=settings.OPENAI_JUDGEMENT_MODEL_NAME,
            include_reason=True,  
        ),
        ]

        # 4. EVALUATE
    result = evaluate(test_cases=test_cases, metrics=metrics)
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())
    