import json
from core.config import settings
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from src.rag_pipeline import RagPipeline
    # 1. LOAD queries (we only need the queries --- context comes from the pipeline now)
with open(settings.GENERATOR_DATASET_PATH) as f:
    goldens = json.load(f)

    # 2. RUN THE INJECTED PIPELINE per query, build a test case from LIVE output
rag = RagPipeline()
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
evaluate(test_cases=test_cases, metrics=metrics)
    