import json

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore

from prompts import TOOL_CALLER_PROMPT


def make_retrieve_tool(retriever):
    @tool
    async def retrieve(query: str):
        """Retrieve relevant information from the knowledge base given a query."""
        result = await retriever.ainvoke(query)
        docs = [{"id": doc.id, "metadata": doc.metadata, "page_content": doc.page_content} for doc in result]
        return json.dumps(docs, indent=4, ensure_ascii=False)

    return retrieve


async def query_or_respond(state: MessagesState, config: RunnableConfig, store: BaseStore, llm, retrieve_tool):
    """Generate tool call for retrieval or respond."""

    print("Messages to model:", state["messages"])

    llm_with_tools = llm.bind_tools([retrieve_tool])

    sys_message = [SystemMessage(content=TOOL_CALLER_PROMPT)]

    print("System message:", sys_message)

    response = await llm_with_tools.ainvoke(sys_message + state["messages"])
    return {"messages": [response]}
