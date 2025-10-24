from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion

from semantic_kernel.agents import ChatCompletionAgent


from semantic_kernel.connectors.ai.google.google_ai.google_ai_prompt_execution_settings import (GoogleAIChatPromptExecutionSettings)
import os

import asyncio

from dotenv import load_dotenv
load_dotenv()

async def main():
    email_kernel = Kernel()
    email_chat_completion = GoogleAIChatCompletion(api_key=os.environ['google_api_key'],gemini_model_id="gemini-2.5-flash",service_id="google-email")
    email_kernel.add_service(email_chat_completion)

    email_agent = ChatCompletionAgent(
        name="email_agent",
        instructions="You are an email agent that writes a clear and professional one-page document.",
        kernel=email_kernel,
        description="An agent that drafts concise and well-structured one-page documents."
    )

    thread = None
    user_input = input("User>")


    while user_input!="exit":

        async for response in email_agent.invoke(user_input,thread=thread):
            print(f"Assistant> {response.content}")
            thread=response.thread

        user_input=input("User> ")



if __name__ == '__main__':
    asyncio.run(main())