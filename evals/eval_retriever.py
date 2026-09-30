from core.config import settings
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric,ContextualPrecisionMetric
import json
from src.retriever import build_retriever

with open(settings.DATASET_PATH) as v :
    dataset = json.load(v)
retriever = build_retriever()
test_cases = []
for case in dataset :
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
metrics = [
    ContextualRecallMetric(threshold=settings.EVAL_THRESHOLD,model=settings.JUDGEMENT_MODEL_NAME,include_reason=True),
    ContextualPrecisionMetric(threshold=settings.EVAL_THRESHOLD,model=settings.JUDGEMENT_MODEL_NAME,include_reason=True),
]    
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
            #"retriever": "base_k5",          # vs "reranked" when you swap it in
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k": settings.TOP_K_CONSTANT,
            "judge_model": settings.JUDGEMENT_MODEL_NAME,
            "golden_set": settings.DATASET_PATH,
    }
    
)
    

