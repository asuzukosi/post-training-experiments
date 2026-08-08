# LangGraph / Deep Agents examples

Sequential and multi-phase pipelines using LangGraph `create_agent` and Deep Agents `create_deep_agent`. Each project has its own local `config.py` (and `pipeline.py` when needed).

## Basic
- [azure-model](./azure-model/) — single researcher agent
- [markdown-validator](./markdown-validator/) — lint tool + create_agent
- [starter-template](./starter-template/) — two-node sequential graph template
- [game-builder](./game-builder/) — engineer → qa → chief qa
- [job-posting](./job-posting/) — research → draft → review
- [instagram-post](./instagram-post/) — copy + image brief pipeline
- [trip-planner](./trip-planner/) — city → guide → itinerary
- [prep-for-a-meeting](./prep-for-a-meeting/) — research → briefing

## Advanced
- [stock-analysis](./stock-analysis/) — SEC/Yahoo/search tools
- [landing-page-generator](./landing-page-generator/) — multi-phase; deep agent for file edits
- [screenplay-writer](./screenplay-writer/) — discussion to screenplay
- [email workflow](../workflows/crewai-langgraph/) — LangGraph loop + agent subgraph
