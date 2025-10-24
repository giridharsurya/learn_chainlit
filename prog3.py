import asyncio
import logging
from semantic_kernel import Kernel
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.mcp import MCPSsePlugin
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.agents import (
    ChatCompletionAgent,
    ChatHistoryAgentThread,
)
from semantic_kernel.agents import Agent, ChatCompletionAgent, HandoffOrchestration, OrchestrationHandoffs
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.filters import FilterTypes, PromptRenderContext
from typing import Awaitable, Callable
from semantic_kernel.filters import FunctionInvocationContext
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.google.google_ai.google_ai_prompt_execution_settings import (GoogleAIChatPromptExecutionSettings)
import os
from typing import TypedDict, Literal

import asyncio

from azure.identity import AzureCliCredential

from semantic_kernel.agents import Agent, ChatCompletionAgent, HandoffOrchestration, OrchestrationHandoffs
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import AuthorRole, ChatMessageContent, FunctionCallContent, FunctionResultContent
from semantic_kernel.functions import kernel_function

from lights_plugin import LightsPlugin
from weather_plugin import Weather
from dotenv import load_dotenv
load_dotenv()

def get_agents():

    manager_kernel  = Kernel()
    email_kernel = Kernel()
    coding_kernel = Kernel()
    driver_kernel = Kernel()
    weather_kernel = Kernel()

    manager_chat_completion = GoogleAIChatCompletion(api_key=os.environ['google_api_key'],gemini_model_id="gemini-2.5-flash",service_id="google-main")
    email_chat_completion = GoogleAIChatCompletion(api_key=os.environ['google_api_key'],gemini_model_id="gemini-2.5-flash",service_id="google-email")
    coding_chat_completion = GoogleAIChatCompletion(api_key=os.environ['google_api_key'],gemini_model_id="gemini-2.5-flash",service_id="google-coding")
    driver_chat_completion = GoogleAIChatCompletion(api_key=os.environ['google_api_key'],gemini_model_id="gemini-2.5-flash",service_id="google-driver")
    weather_chat_completion = GoogleAIChatCompletion(api_key=os.environ['google_api_key'],gemini_model_id="gemini-2.5-flash",service_id="google-weather")

    manager_kernel.add_service(manager_chat_completion) 
    email_kernel.add_service(email_chat_completion)
    coding_kernel.add_service(coding_chat_completion)
    driver_kernel.add_service(driver_chat_completion)
    weather_kernel.add_service(weather_chat_completion)

    manager_agent = ChatCompletionAgent(
    name="manager_agent",
    instructions="You are a manager agent that orchestrates and coordinates all other agents to complete the task.",
    kernel=manager_kernel,
    description="An orchestrator agent that is the first point of contact to the user and it manages and oversees all other agents."
    )

    weather_agent = ChatCompletionAgent(
    name="weather_agent",
    instructions="You are a weather agent that that provides the weather status",
    kernel=weather_kernel,
    description="A weather agent that provides the weather status",
    plugins=[Weather()]
    )

    email_agent = ChatCompletionAgent(
        name="email_agent",
        instructions="You are an email agent that writes a clear and professional one-page document.",
        kernel=email_kernel,
        description="An agent that drafts concise and well-structured one-page documents."
    )

    coding_agent = ChatCompletionAgent(
        name="coding_agent",
        instructions="You are a coding agent that reviews and improves written documents or code.",
        kernel=coding_kernel,
        description="An agent that performs detailed reviews and provides improvement suggestions."
    )

    driver_agent = ChatCompletionAgent(
        name="driver_agent",
        instructions="You are a driver agent that follows instructions and executes tasks like a car driver following routes.",
        kernel=driver_kernel,
        description="An execution agent that carries out tasks based on given directions."
    )

    handoffs = (
        OrchestrationHandoffs()
        .add_many(
            source_agent=manager_agent.name,
            target_agents={
                email_agent.name: "Transfer to this agent if the task is email related",
                coding_agent.name: "Transfer to this agent if the task is coding related",
                driver_agent.name: "Transfer to this agent if the task is driver related",
                weather_agent.name: "Transfer to this agent if the task is weather related"
            },
        )
        .add_many(
            source_agent=email_agent.name,
            target_agents={
                weather_agent.name: "Transfer to this agent if the task is weather related",
                manager_agent.name: "Transfer to this agent if the task is not email related"}
        )
        .add(
            source_agent=coding_agent.name,
            target_agent=manager_agent.name,
            description="Transfer to this agent if the task is not coding related",
        )
        .add(
            source_agent=driver_agent.name,
            target_agent=manager_agent.name,
            description="Transfer to this agent if the the task is not driving related",
        )
        .add(
            source_agent=weather_agent.name,
            target_agent=email_agent.name,
            description="Transfer to this agent if the the task is email related",
        )
    )

    return [manager_agent,email_agent,coding_agent,driver_agent,weather_agent], handoffs

def agent_response_callback(message: ChatMessageContent) -> None:
    """Observer function to print the messages from the agents.

    Please note that this function is called whenever the agent generates a response,
    including the internal processing messages (such as tool calls) that are not visible
    to other agents in the orchestration.
    """
    print(f"agent_response_callback> {message.name}: {message.content}")
    for item in message.items:
        if isinstance(item, FunctionCallContent):
            print(f"Calling '{item.name}' with arguments '{item.arguments}'")
        if isinstance(item, FunctionResultContent):
            print(f"Result from '{item.name}' is '{item.result}'")

def human_response_function() -> ChatMessageContent:
    """Observer function to print the messages from the agents."""
    user_input = input("User: ")
    return ChatMessageContent(role=AuthorRole.USER, content=user_input)

async def main():
    
    agents, handoffs = get_agents()

    handoff_orchestration = HandoffOrchestration(
        members=agents,
        handoffs=handoffs,
        agent_response_callback=agent_response_callback,
        human_response_function=human_response_function,
    )

    # 2. Create a runtime and start it
    runtime = InProcessRuntime()
    runtime.start()

    user_input = input("User>")

    while user_input!="exit":

        # 3. Invoke the orchestration with a task and the runtime
        orchestration_result = await handoff_orchestration.invoke(
            task=user_input,
            runtime=runtime,
        )

        # 4. Wait for the results
        value = await orchestration_result.get()
        print(f"orchestration_result> {value}")

        user_input = input("User>")


    # 5. Stop the runtime after the invocation is complete
    await runtime.stop_when_idle()

    
if __name__ == "__main__":
    asyncio.run(main())