from core.config import settings
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric,ContextualPrecisionMetric
import json
import time
from src.reranker import RerankingRetriever

with open(settings.DATASET_PATH) as v :
    dataset = json.load(v)
retriever = RerankingRetriever()
test_cases = []
for case in dataset  : # number of testcases depreicated cuz i am learning
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
    time.sleep(settings.OPENAI_EMBEDDING_SLEEP_DELAY)
metrics = [
    ContextualRecallMetric(threshold=settings.EVAL_THRESHOLD,model=settings.OPENAI_JUDGEMENT_MODEL_NAME,include_reason=True),
    ContextualPrecisionMetric(threshold=settings.EVAL_THRESHOLD,model=settings.OPENAI_JUDGEMENT_MODEL_NAME,include_reason=True),
]    
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
            "retriever": "reranker",        
            "embedding_model": settings.OPENAI_EMBEDDING_MODEL_NAME,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k": settings.TOP_K_CONSTANT,
            "judge_model": settings.OPENAI_JUDGEMENT_MODEL_NAME,
            "golden_set": settings.DATASET_PATH,
            
    },
    
)
    

