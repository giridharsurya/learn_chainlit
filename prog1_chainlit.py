import os
import asyncio
from dotenv import load_dotenv

import chainlit as cl
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
from semantic_kernel.agents import ChatCompletionAgent

load_dotenv()

# --- Kernel & Model Setup ---
email_kernel = Kernel()
email_chat_completion = GoogleAIChatCompletion(
    api_key=os.environ["google_api_key"],
    gemini_model_id="gemini-2.5-flash",
    service_id="google-email"
)
email_kernel.add_service(email_chat_completion)

email_agent = ChatCompletionAgent(
    name="email_agent",
    instructions="You are an email agent that writes a clear and professional one-page document.",
    kernel=email_kernel,
    description="An agent that drafts concise and well-structured one-page documents."
)

# --- Helper Function ---
async def get_agent_response(user_input, thread):
    """Handles the async streaming of responses from the SK agent."""
    async for response in email_agent.invoke(user_input, thread=thread):
        # Display incremental responses (streaming)
        assist_msg = cl.Message(content=response.content)
        await assist_msg.send()
        thread = response.thread

    # Store updated thread in session
    cl.user_session.set("thread", thread)

# --- Chainlit Events ---
@cl.on_chat_start
async def setup_variables():
    """Initialize session variables."""
    cl.user_session.set("thread", None)
    await cl.Message(content="👋 Hi! I’m your Email Agent. What should I help you write today?").send()

@cl.on_message
async def main(message: cl.Message):
    """Handles user messages."""
    thread = cl.user_session.get("thread")
    await get_agent_response(message.content, thread)
