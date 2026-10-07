import json
from core.config import settings
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from src.generator import generate
    
with open(settings.GENERATOR_DATASET_PATH)as f :
    goldens = json.loads(f)


test_cases = []
for g in goldens:
    context = g["ideal_context"]              # known-good context (list of chunk strings) taken from curated dataset
    answer = generate(g["query"], context)    # RUN the generator -> actual_output

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=answer,             # the generated answer we're judging
            retrieval_context=context,        # faithfulness checks the answer against THIS
            # no expected_output --- faithfulness never reads it
        )
    )

# 3. THE METRICS --- decompose actual_output into claims, attribute each to context
metrics = [
    FaithfulnessMetric(
        threshold=settings.EVAL_THRESHOLD,
        model=settings.OPENAI_JUDGEMENT_MODEL_NAME,
        include_reason=True,   # prints WHY each score --- shows which claims were unsupported
    ),
    AnswerRelevancyMetric(
        threshold=settings.EVAL_THRESHOLD,
        model=settings.OPENAI_JUDGEMENT_MODEL_NAME,
        include_reason=True,
    ),
]

# 4. EVALUATE --- runs the metrics on every case, prints a report
result = evaluate(test_cases=test_cases, metrics=metrics)

