from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase


from harness.config.settings import settings

judge = GeminiModel(
    model="gemini-2.5-flash",
    api_key=settings.gemini_api_key.get_secret_value()    
)

test_case = LLMTestCase(
    input= "How can i reset my password",
    actual_output= "To reset your password, go to the login page, click 'Forgot Password', "
        "enter your email, and follow the instructions sent to you"
)

metrics = AnswerRelevancyMetric(model=judge, threshold=0.7)
metrics.measure(test_case)

print(f"Score : {metrics.score}")
print(f"Reason: {metrics.reason}")
print(f"Passed: {metrics.is_successful()}")
