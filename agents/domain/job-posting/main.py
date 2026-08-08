"""job posting: sequential multi-role langgraph pipeline."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from pipeline import build_sequential_graph, make_agent_node
from tools import RESEARCH_TOOLS, WRITER_REVIEW_TOOLS


RESEARCH_BASE = dedent(
    """
    role: research analyst
    goal: analyze the company website and provided description to extract insights on
    culture, values, and specific needs.
    backstory: expert in analyzing company cultures and identifying key values and needs
    from various sources, including websites and brief descriptions.
    use your search tools when you need current market or company information.
    """
).strip()

WRITER_BASE = dedent(
    """
    role: job description writer
    goal: use insights from prior research to create a detailed, engaging, and enticing job posting.
    backstory: skilled in crafting compelling job descriptions that resonate with the company's
    values and attract the right candidates.
    use search tools and the job description example file when helpful.
    """
).strip()

REVIEW_BASE = dedent(
    """
    role: review and editing specialist
    goal: review the job posting for clarity, engagement, grammatical accuracy, and alignment
    with company values and refine it to ensure perfection.
    backstory: a meticulous editor with an eye for detail, ensuring every piece of content is
    clear, engaging, and grammatically perfect.
    """
).strip()


def build_graph(
    company_description: str,
    company_domain: str,
    hiring_needs: str,
    specific_benefits: str,
):
    culture_prompt = dedent(
        f"""
        {RESEARCH_BASE}

        analyze the provided company domain "{company_domain}" and description:
        "{company_description}". focus on understanding the company's culture, values, and mission.
        identify unique selling points and specific projects or achievements.
        compile a report summarizing these insights, specifically how they can be leveraged
        in a job posting to attract the right candidates.

        expected output: a comprehensive report detailing the company's culture, values, and mission,
        along with specific selling points relevant to the job role.
        """
    ).strip()

    industry_prompt = dedent(
        f"""
        {RESEARCH_BASE}

        conduct an in-depth analysis of the industry related to the company's domain:
        "{company_domain}" and description "{company_description}".
        investigate current trends, challenges, and opportunities within the industry.
        assess how these factors could impact the role being hired for.

        expected output: a detailed analysis report on industry trends, challenges, and opportunities
        relevant to the company's domain and the specific job role.
        """
    ).strip()

    role_requirements_prompt = dedent(
        f"""
        {RESEARCH_BASE}

        based on the hiring manager's needs: "{hiring_needs}", identify the key skills, experiences,
        and qualities the ideal candidate should possess for the role.
        consider the company's current projects, competitive landscape, and industry trends.
        prepare a list of recommended job requirements and qualifications.

        expected output: a list of recommended skills, experiences, and qualities for the ideal candidate.
        """
    ).strip()

    draft_prompt = dedent(
        f"""
        {WRITER_BASE}

        draft a job posting for the role described by the hiring manager: "{hiring_needs}".
        use insights on "{company_description}" to start with a compelling introduction, followed by
        a detailed role description, responsibilities, and required skills and qualifications.
        ensure the tone aligns with the company's culture.
        specific benefits: "{specific_benefits}"

        expected output: a detailed, engaging job posting with introduction, role description,
        responsibilities, requirements, and unique company benefits.
        """
    ).strip()

    review_prompt = dedent(
        f"""
        {REVIEW_BASE}

        review the draft job posting for the role: "{hiring_needs}".
        check for clarity, engagement, grammatical accuracy, and alignment with the company's
        culture and values. edit and refine the content so it speaks directly to desired candidates.
        format the final output in markdown.

        expected output: a polished, error-free job posting formatted in markdown.
        """
    ).strip()

    return build_sequential_graph(
        [
            ("research_culture", make_agent_node(culture_prompt, tools=RESEARCH_TOOLS)),
            ("industry_analysis", make_agent_node(industry_prompt, tools=RESEARCH_TOOLS)),
            ("role_requirements", make_agent_node(role_requirements_prompt, tools=RESEARCH_TOOLS)),
            ("draft_posting", make_agent_node(draft_prompt, tools=WRITER_REVIEW_TOOLS)),
            ("review_posting", make_agent_node(review_prompt, tools=WRITER_REVIEW_TOOLS)),
        ]
    )


def main() -> None:
    company_description = "We are a shoe making company"
    company_domain = "We make designer shoes"
    hiring_needs = "Shoe making experience"
    specific_benefits = "Remote work"

    graph = build_graph(company_description, company_domain, hiring_needs, specific_benefits)
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"create a job posting. company: {company_description}. "
                        f"domain: {company_domain}. role needs: {hiring_needs}. "
                        f"benefits: {specific_benefits}."
                    ),
                }
            ],
            "output": "",
        }
    )

    final_posting = result.get("output", "")
    output_path = Path(__file__).resolve().parent / "job_posting.md"
    output_path.write_text(final_posting, encoding="utf-8")

    print("Job Posting Creation Process Completed.")
    print("Final Job Posting:")
    print(final_posting)
    print(f"\nwrote output to {output_path}")


if __name__ == "__main__":
    main()
