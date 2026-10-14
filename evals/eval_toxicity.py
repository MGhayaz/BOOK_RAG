import json
from core.config import settings
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToxicityMetric
from src.rag_pipeline import RagPipeline
# 1. LOAD toxicity inputs
with open(settings.TOXICITY_GOLDEN_PATH) as f:
    goldens = json.load(f)


# 2. RUN THE FULL PIPELINE per input, build a test case from LIVE output
rag = RagPipeline()
test_cases = []

for g in goldens:
    result = rag.invoke(g["input"])             # retrieve → rerank → generate

    test_cases.append(
        LLMTestCase(
            input=g["input"],
            actual_output=result["answer"],
        )
    )


# 3. TOXICITY — built-in DeepEval metric
#    Lower score is better. A test passes when toxicity <= threshold.
toxicity = ToxicityMetric(
    threshold=settings.TOXICITY_THRESHOLD,
    model=settings.OPENAI_JUDGEMENT_MODEL_NAME,
    include_reason=True,
    strict_mode=False,
)


# 4. EVALUATE
evaluate(
    test_cases=test_cases,
    metrics=[toxicity],
)