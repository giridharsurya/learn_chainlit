import os
import asyncio
from dotenv import load_dotenv
import uuid
import chainlit as cl
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.functions import kernel_function
from weather_plugin import Weather

load_dotenv()

class Human:
    @kernel_function(name='get_information',description="get further information regarding the task from user")
    async def get_information(self,query:str)-> str:

        # await cl.Message(author="assistant", content=query).send()
        query = "tool_request> " + query
        user_msg = await cl.AskUserMessage(content=query, timeout=300).send()
        return user_msg['output']
    
async def function_invocation(context,next):
    print(context.function.name)
    print(context.arguments)
    # if context.arguments['city']=='paris':
    #     count = 2
    # else:
    #     count = 10
    # for i in range(count):
    #     print(f"{context.arguments['city']}-{i}")
    #     await asyncio.sleep(2)
    await next(context)
    print(context.result.value)
    await cl.Message(content=context.result.value).send()

async def handle_intermediate_steps(message: ChatMessageContent) -> None:
    for item in message.items or []:
        if isinstance(item, FunctionCallContent):
            await cl.Message(content=f"Function Call:> {item.name} with arguments: {item.arguments}").send()
            # print(f"Function Call:> {item.name} with arguments: {item.arguments}")
        elif isinstance(item, FunctionResultContent):
            await cl.Message(content=f"Function Result:> {item.result} for function: {item.name}").send()
            # print(f"Function Result:> {item.result} for function: {item.name}")
        else:
            await cl.Message(content=f"{message.role}: {message.content}").send()
            # print(f"{message.role}: {message.content}")

# --- Kernel & Model Setup ---
email_kernel = Kernel()
email_chat_completion = GoogleAIChatCompletion(
    api_key=os.environ["google_api_key"],
    gemini_model_id="gemini-2.5-flash",
    service_id="google-email"
)
email_kernel.add_service(email_chat_completion)

email_kernel.add_filter("function_invocation",function_invocation)

# --- Helper Function ---
async def get_agent_response(user_input, thread):
    """Handles the async streaming of responses from the SK agent."""
    agent = cl.user_session.get("agent")
    async for response in agent.invoke(user_input, thread=thread,on_intermediate_message=handle_intermediate_steps):
        # Display incremental responses (streaming)
        thread = response.thread
    await cl.Message(content=str(response.content)).send()

    # Store updated thread in session
    cl.user_session.set("thread", thread)

# --- Chainlit Events ---
@cl.on_chat_start
async def setup_variables():
    """Initialize session variables."""
    session_id = str(uuid.uuid4())[:8]
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("thread", None)
    email_agent = ChatCompletionAgent(
        name="email_agent",
        instructions="You are an ai assistant to help users with weather",
        kernel=email_kernel,
        description="An agent that helps users with weather",
        plugins=[Weather()]
        )
    cl.user_session.set("agent",email_agent)
    await cl.Message(content=f"👋 Session started. ID: {session_id}").send()

@cl.on_message
async def main(message: cl.Message):
    """Handles user messages."""

    # Case 2: Normal conversation flow
    thread = cl.user_session.get("thread")
    await get_agent_response(message.content, thread)
