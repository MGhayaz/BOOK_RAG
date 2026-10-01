from core.config import settings
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric,ContextualPrecisionMetric
import json
import time
from src.retriever import build_retriever

with open(settings.DATASET_PATH) as v :
    dataset = json.load(v)
retriever = build_retriever()
test_cases = []
for case in dataset [:5] :
    retrieved = retriever.invoke(case["query"])
    retrieval_context = [doc.page_content for doc in retrieved]
    test_cases.append(
        LLMTestCase(
            input=case["query"],
            expected_output=case["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output=("generator not evaluated in this run"),
        )
    )
    time.sleep(settings.EMBEDDING_SLEEP_DELAY)
metrics = [
    ContextualRecallMetric(threshold=settings.EVAL_THRESHOLD,model=settings.JUDGEMENT_MODEL_NAME,include_reason=True),
    ContextualPrecisionMetric(threshold=settings.EVAL_THRESHOLD,model=settings.JUDGEMENT_MODEL_NAME,include_reason=True),
]    
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(
        max_concurrent=1,    # reduce parallel calls
        throttle_value=40 ,    # wait 3 seconds between test cases
    ),
    hyperparameters={
            #"retriever": "base_k5",          # vs "reranked" when you swap it in
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k": settings.TOP_K_CONSTANT,
            "judge_model": settings.JUDGEMENT_MODEL_NAME,
            "golden_set": settings.DATASET_PATH,
            
    },
    
)
    

