"""screenplay writer: sequential langgraph pipeline from discussion text."""

from __future__ import annotations

import re
from textwrap import dedent

from config import final_text
from pipeline import build_sequential_graph, make_agent_node


DISCUSSION = dedent(
    """
    From: keith@cco.caltech.edu (Keith Allan Schneider)
    Subject: Re: <Political Atheists?
    Organization: California Institute of Technology, Pasadena
    Lines: 50
    NNTP-Posting-Host: punisher.caltech.edu

    bobbe@vice.ICO.TEK.COM (Robert Beauchaine) writes:

    >>I think that about 70% (or so) people approve of the
    >>death penalty, even realizing all of its shortcomings.  Doesn't this make
    >>it reasonable?  Or are *you* the sole judge of reasonability?
    >Aside from revenge, what merits do you find in capital punishment?

    Are we talking about me, or the majority of the people that support it?
    Anyway, I think that "revenge" or "fairness" is why most people are in
    favor of the punishment.  If a murderer is going to be punished, people
    that think that he should "get what he deserves."  Most people wouldn't
    think it would be fair for the murderer to live, while his victim died.

    >Revenge?  Petty and pathetic.

    Perhaps you think that it is petty and pathetic, but your views are in the
    minority.

    >We have a local televised hot topic talk show that very recently
    >did a segment on capital punishment.  Each and every advocate of
    >the use of this portion of our system of "jurisprudence" cited the
    >main reason for supporting it:  "That bastard deserved it".  True
    >human compassion, forgiveness, and sympathy.

    Where are we required to have compassion, forgiveness, and sympathy?  If
    someone wrongs me, I will take great lengths to make sure that his advantage
    is removed, or a similar situation is forced upon him.  If someone kills
    another, then we can apply the golden rule and kill this person in turn.
    Is not our entire moral system based on such a concept?

    Or, are you stating that human life is sacred, somehow, and that it should
    never be violated?  This would sound like some sort of religious view.

    >>I mean, how reasonable is imprisonment, really, when you think about it?
    >>Sure, the person could be released if found innocent, but you still
    >>can't undo the imiprisonment that was served.  Perhaps we shouldn't
    >>imprision people if we could watch them closely instead.  The cost would
    >>probably be similar, especially if we just implanted some sort of
    >>electronic device.
    >Would you rather be alive in prison or dead in the chair?

    Once a criminal has committed a murder, his desires are irrelevant.

    And, you still have not answered my question.  If you are concerned about
    the death penalty due to the possibility of the execution of an innocent,
    then why isn't this same concern shared with imprisonment.  Shouldn't we,
    by your logic, administer as minimum as punishment as possible, to avoid
    violating the liberty or happiness of an innocent person?

    keith
    """
).strip()

SPAM_FILTER_PROMPT = dedent(
    """
    you are an expert spam filter with years of experience.
    you detest advertisements, newsletters and vulgar language.
    read the newsgroup post. if it contains vulgar language reply with STOP.
    if it is spam reply with STOP. otherwise reply with OK.
    """
).strip()

ANALYST_PROMPT = dedent(
    """
    you are an expert discussion analyst.
    distill all arguments from all discussion members. identify who said what.
    you may reword what they said as long as the main discussion points remain.
    """
).strip()

SCRIPTWRITER_PROMPT = dedent(
    """
    you are an expert on writing natural sounding movie script dialogues.
    turn the analyzed conversation into a movie script dialogue between two persons.
    only write dialogue. do not start sentences with actions.
    do not specify situational descriptions. do not write parentheticals or wrylies.
    skip directional notes.
    """
).strip()

FORMATTER_PROMPT = dedent(
    """
    you are an expert text formatter.
    format the script exactly like this:
      ## (person 1):
      (first text line from person 1)

      ## (person 2):
      (first text line from person 2)

      ## (person 1):
      (second text line from person 1)

      ## (person 2):
      (second text line from person 2)

    leave out actions between brackets, eg (smiling).
    """
).strip()

SCORER_PROMPT = dedent(
    """
    you are an expert at scoring conversations on a scale of 1 to 10.
    score the dialogue on clarity, relevance, conciseness, politeness, engagement,
    flow, coherence, responsiveness, language use, and emotional intelligence.
    only give the score as a number, nothing else. do not give an explanation.
    """
).strip()


def build_screenplay_graph(discussion: str):
    analyst_prompt = f"{ANALYST_PROMPT}\n\nanalyze in detail:\n### discussion:\n{discussion}"
    return build_sequential_graph(
        [
            ("analyst", make_agent_node(analyst_prompt)),
            ("scriptwriter", make_agent_node(SCRIPTWRITER_PROMPT)),
            ("formatter", make_agent_node(FORMATTER_PROMPT)),
        ]
    )


def run_spam_filter(discussion: str) -> str:
    from langchain.agents import create_agent

    from config import get_model_id

    agent = create_agent(
        model=get_model_id(),
        tools=[],
        system_prompt=SPAM_FILTER_PROMPT,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"### newsgroup post:\n{discussion}"}]}
    )
    return final_text(result)


def run_scorer(dialogue: str) -> str:
    from langchain.agents import create_agent

    from config import get_model_id

    agent = create_agent(
        model=get_model_id(),
        tools=[],
        system_prompt=SCORER_PROMPT,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"read the following dialogue:\n{dialogue}"}]}
    )
    return final_text(result).split("\n")[0]


def main() -> None:
    discussion = DISCUSSION
    filter_result = run_spam_filter(discussion)
    if "STOP" in filter_result.upper():
        print("this spam message will be filtered out")
        return

    graph = build_screenplay_graph(discussion)
    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": discussion}],
            "output": "",
        }
    )
    screenplay = re.sub(r"\(.*?\)", "", result.get("output", ""))

    print("===================== end result from pipeline ===================================")
    print(screenplay)
    print("===================== score ==================================================")
    score = run_scorer(screenplay)
    print(f"scoring the dialogue as: {score}/10")


if __name__ == "__main__":
    main()
