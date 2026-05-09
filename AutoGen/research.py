from dotenv import load_dotenv
import os
load_dotenv()


from autogen_ext.models.openai import OpenAIChatCompletionClient



model_client = OpenAIChatCompletionClient(
    model="gpt-5-nano",
    api_key=os.getenv("OPENAI_API_KEY")
)
