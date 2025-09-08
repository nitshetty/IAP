import asyncio
from deepeval.evaluate import evaluate
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluate.configs import AsyncConfig

from rag_pipeline import build_rag_pipeline


# --------------------------
# Custom Metrics
# --------------------------
class CustomFaithfulnessMetric(BaseMetric):
    def __init__(self, model=None, async_mode=False, threshold=0.5):
        super().__init__()
        self.model = model
        self.async_mode = async_mode
        self.name = "CustomFaithfulness"
        self.threshold = threshold  # Properly initialize threshold

    def measure(self, test_case: LLMTestCase):
        answer = test_case.actual_output
        context = test_case.context
        expected = test_case.expected_output

        # Debugging inputs
        print("Debug: Actual Output:", answer)
        print("Debug: Context:", context)
        print("Debug: Expected Output:", expected)

        # Check if the expected output matches the actual output
        if expected and expected.lower() in answer.lower():
            self.score = 1.0
            print("Debug: Setting score to 1.0")
            return 1.0

        self.score = 0.0
        print("Debug: Setting score to 0.0")
        return 0.0  # Ensure a valid score is always returned

    async def a_measure(self, test_case: LLMTestCase):
        if not self.async_mode:
            # Force synchronous execution
            return self.measure(test_case)
        return self.measure(test_case)

    def evaluate(self, test_case: LLMTestCase): 
        return asyncio.get_event_loop().run_until_complete(self.a_evaluate(test_case))

    def is_successful(self) -> bool:
        return self.score is not None and self.score >= self.threshold


class CustomAnswerRelevancyMetric(BaseMetric):
    def __init__(self, model, async_mode=False, threshold=0.5):
        super().__init__()
        self.model = model
        self.async_mode = async_mode
        self.threshold = 0.5
        self.score = None

    async def a_evaluate(self, test_case: LLMTestCase):
        """
        Async evaluation logic for Relevancy.
        Replace the dummy logic with an actual model call if needed.
        """
        if len(test_case.actual_output.strip()) > 0:
            self.score = 0.9
        else:
            self.score = 0.2
        return self.score

    def is_successful(self) -> bool:
        return self.score is not None and self.score >= self.threshold


# --------------------------
# Main Evaluation
# --------------------------
def main():
    rag_chain, llm = build_rag_pipeline()

    query = "Who developed the theory of relativity?"
    result = rag_chain.invoke(query)  # Replace deprecated Chain.run with Chain.invoke

    test_case = LLMTestCase(
        input=query,
        actual_output=result["result"],
        expected_output="Albert Einstein",  # Set a valid expected output
        context=["Albert Einstein developed the theory of relativity."]
    )

    # Debugging the test case
    print("Debug: Test Case:", test_case)

    metrics = [
        CustomFaithfulnessMetric(model=llm, async_mode=False, threshold=0.5)
    ]

    # Call evaluate directly without asyncio.run
    results = evaluate(
        [test_case],
        metrics,
        async_config=AsyncConfig(run_async=False)  # Enforce synchronous execution
    )

    print("Evaluation Results ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()