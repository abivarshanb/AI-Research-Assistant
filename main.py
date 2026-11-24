
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import (
    create_tool_calling_agent,
    AgentExecutor
)

from tools import search_tool, wiki_tool, save_tool


# Load environment variables
load_dotenv()


# Define the research response structure
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)


# Initialize parser
parser = PydanticOutputParser(
    pydantic_object=ResearchResponse
)


# Create prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful research assistant.

        Research the user's topic using the available tools.
        Use tools when necessary.

        Return the final answer as valid JSON only.
        Do not include Markdown code fences.

        Follow this structure:

        {format_instructions}
        """
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}")
]).partial(
    format_instructions=parser.get_format_instructions()
)


# Register tools
tools = [
    search_tool,
    wiki_tool,
    save_tool
]


# Create agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)


# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)


# Get user input
query = input("\nWhat can I help you research? ")


# Execute the agent
try:
    response = agent_executor.invoke({
        "query": query,
        "chat_history": []
    })

    output = response["output"]

    # Convert the output into the Pydantic model
    structured_response = parser.parse(output)

    print("\n========== RESEARCH RESULT ==========")
    print(structured_response.model_dump_json(indent=4))

except Exception as error:
    print("\nSomething went wrong:")
    print(error)

