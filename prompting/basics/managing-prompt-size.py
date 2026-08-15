from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts.chat import ChatPromptValue
from langchain.tools import WikipediaQueryRun
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from operator import itemgetter


load_dotenv()

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=10_000),
)
tools = [wiki]



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)


model = ChatOpenAI()



def condense_prompt(prompt: ChatPromptValue) -> ChatPromptValue:
    messages = prompt.to_messages()
    num_tokens = model.get_num_tokens_from_messages(messages)
    ai_function_messages = messages[2:]
    while num_tokens > 4_000:
        ai_function_messages = ai_function_messages[2:]
        num_tokens = model.get_num_tokens_from_messages(
            messages[:2] + ai_function_messages
        )
    messages = messages[:2] + ai_function_messages
    return ChatPromptValue(messages=messages)
    
agent = (
    {
        "input": itemgetter("input"),
        "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"])
    } | prompt | condense_prompt | model.bind(functions=[format_tool_to_openai_function(t) for t in tools])
     | OpenAIFunctionsAgentOutputParser()
)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print(agent_executor.invoke({
    "input": "Who is the current US president? What's their home state? What's their home state's bird? What's that bird's scientific name?"
}))