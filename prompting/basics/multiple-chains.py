"""
This section covers working with multiple chains in langchain  
"""

from dotenv import load_dotenv
from operator import itemgetter
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

prompt1 = ChatPromptTemplate.from_template("what is the city {person} is from?")
prompt2 = ChatPromptTemplate.from_template("What country is the city {city} in? respond in {language}")

model = ChatOpenAI()
# chain1 = prompt1 | model | StrOutputParser()


# chain2 = (
#     {"city": chain1, "language": itemgetter("language")}
#     | prompt2 | model | StrOutputParser()
# )


# print(chain2.invoke({"person": "obama", "language": "english"}))

# prompt1 = ChatPromptTemplate.from_template("generate a {attribute} color. Returen the name of the color and nothing else.")
# prompt2 = ChatPromptTemplate.from_template("What is a fruit of color: {color}. Return the name of the fruit and nothing else.")
# prompt3 = ChatPromptTemplate.from_template("what is a coutnry with a flag that has the color: {color}. Return the name and nothing else.")
# prompt4 = ChatPromptTemplate.from_template("What is the color of {fruit} and the flag of country {country}?")


# model_parser = model | StrOutputParser()


# color_generator = {"attribute": RunnablePassthrough()} | prompt1 | {"color": model_parser}


# color_of_fruit = prompt2 | model_parser
# color_of_country = prompt3 | model_parser

# question_generator = color_generator | {"fruit": color_of_fruit, "country": color_of_country} | prompt4

# prompt = question_generator.invoke("warm")
# print(model.invoke(prompt))


# Runnable parallel chains

planner = (
    ChatPromptTemplate.from_template("Generate an argument aobut : {input}")
    | ChatOpenAI()
    | StrOutputParser()
    | {"base_response": RunnablePassthrough()}
)

arguments_for = (
    ChatPromptTemplate.from_template("List the pros or positive aspects of {base_response}")
    | ChatOpenAI()
    | StrOutputParser()
)

arguments_against = (
    ChatPromptTemplate.from_template("List all the arugments against {base_response}")
    | ChatOpenAI()
    | StrOutputParser()
)

final_responder = (
    ChatPromptTemplate.from_messages(
        [
            ("ai", "{original_response}"),
            ("human", "Pros: \n {result_1} \n\n Cons: \n {result_2}"),
            ("system", "Generate a final response given the critique")
        ]
    ) | ChatOpenAI() | StrOutputParser()
)

chain = (
    planner | {"result_1": arguments_for, "result_2": arguments_against, "original_response": itemgetter("base_response")} | final_responder
)

print(chain.invoke({"input": "scrum"}))