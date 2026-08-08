"""multi-phase langgraph pipeline for landing page generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, TypedDict

from deepagents import create_deep_agent
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from config import final_text, get_model_id
from pipeline import build_sequential_graph, make_agent_node
from prompts import AGENT_PROMPTS
from tasks import TaskPrompts
from tools.browser_tools import BrowserTools
from tools.file_tools import FileTools
from tools.search_tools import SearchTools
from tools.template_tools import TemplateTools


RESEARCH_TOOLS = [
    SearchTools.search_internet,
    BrowserTools.scrape_and_summarize_website,
]

DEVELOPER_TOOLS = [
    SearchTools.search_internet,
    BrowserTools.scrape_and_summarize_website,
    TemplateTools.learn_landing_page_options,
    TemplateTools.copy_landing_page_template_to_project_folder,
    FileTools.write_file,
]


class LandingState(TypedDict):
    messages: Annotated[list, add_messages]
    output: str
    idea: str
    expanded_idea: str
    components: list[str]
    component_index: int


def _deep_agent(system_prompt: str, tools: list):
    return create_deep_agent(
        model=get_model_id(),
        tools=tools,
        system_prompt=system_prompt,
    )


def expand_idea_node(state: LandingState) -> dict[str, Any]:
    idea = state["idea"]
    expand_graph = build_sequential_graph(
        [
            (
                "idea_analyst",
                make_agent_node(
                    f"{AGENT_PROMPTS['senior_idea_analyst']}\n\n{TaskPrompts.expand().format(idea=idea)}",
                    tools=RESEARCH_TOOLS,
                ),
            ),
            (
                "strategist",
                make_agent_node(
                    f"{AGENT_PROMPTS['senior_strategist']}\n\n{TaskPrompts.refine_idea()}",
                    tools=RESEARCH_TOOLS,
                ),
            ),
        ]
    )
    result = expand_graph.invoke(
        {"messages": [{"role": "user", "content": idea}], "output": ""}
    )
    expanded = result.get("output", "")
    return {"expanded_idea": expanded, "output": expanded}


def choose_template_node(state: LandingState) -> dict[str, Any]:
    idea = state["idea"]
    prompt = (
        f"{AGENT_PROMPTS['senior_react_engineer']}\n\n"
        f"{TaskPrompts.choose_template().format(idea=idea)}\n\n"
        f"{TaskPrompts.update_page().format(idea=idea)}"
    )
    agent = _deep_agent(prompt, DEVELOPER_TOOLS)
    result = agent.invoke({"messages": [{"role": "user", "content": idea}]})
    raw = final_text(result)
    cleaned = raw.replace("\n", "").replace(" ", "").replace("```", "")
    components = json.loads(cleaned)
    return {"components": components, "component_index": 0, "output": raw}


def process_component_node(state: LandingState) -> dict[str, Any]:
    idx = state.get("component_index", 0)
    components = state["components"]
    if idx >= len(components):
        return {}

    component = components[idx]
    expanded_idea = state["expanded_idea"]
    component_path = f"./workdir/{component.split('./')[-1]}"
    file_content = Path(component_path).read_text()

    editor = make_agent_node(
        (
            f"{AGENT_PROMPTS['senior_content_editor']}\n\n"
            f"{TaskPrompts.component_content().format(expanded_idea=expanded_idea, file_content=file_content, component=component)}"
        ),
        tools=RESEARCH_TOOLS,
    )
    editor_result = editor({"messages": [], "output": expanded_idea})
    copy_suggestions = editor_result.get("output", "")

    update_prompt = (
        f"{AGENT_PROMPTS['senior_react_engineer']}\n\n"
        f"{TaskPrompts.update_component().format(component=component, file_content=file_content)}\n\n"
        f"text suggestions:\n{copy_suggestions}"
    )
    update_agent = _deep_agent(update_prompt, DEVELOPER_TOOLS)
    update_agent.invoke(
        {"messages": [{"role": "user", "content": f"update {component}"}]}
    )

    qa_prompt = (
        f"{AGENT_PROMPTS['senior_react_engineer']}\n\n"
        f"{TaskPrompts.qa_component().format(component=component)}"
    )
    qa_agent = _deep_agent(qa_prompt, DEVELOPER_TOOLS)
    qa_agent.invoke({"messages": [{"role": "user", "content": f"qa {component}"}]})

    return {"component_index": idx + 1}


def has_more_components(state: LandingState) -> str:
    if state.get("component_index", 0) < len(state.get("components", [])):
        return "continue"
    return "done"


def build_landing_graph():
    graph = StateGraph(LandingState)
    graph.add_node("expand_idea", expand_idea_node)
    graph.add_node("choose_template", choose_template_node)
    graph.add_node("process_component", process_component_node)

    graph.add_edge(START, "expand_idea")
    graph.add_edge("expand_idea", "choose_template")
    graph.add_edge("choose_template", "process_component")
    graph.add_conditional_edges(
        "process_component",
        has_more_components,
        {"continue": "process_component", "done": END},
    )
    return graph.compile()
