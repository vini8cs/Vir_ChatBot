from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore
from toon import encode

from templates.prompts import TOOL_CALLER_PROMPT


def make_retrieve_tool(retriever):
    @tool
    async def retrieve(query: str):
        """Retrieve relevant information
        from the knowledge base given a query."""
        result = await retriever.ainvoke(query)
        docs = [
            {
                "id": doc.id,
                "metadata": doc.metadata,
                "page_content": doc.page_content,
            }
            for doc in result
        ]
        return encode(docs)

    return retrieve


async def query_or_respond(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore,
    llm,
    retrieve_tool,
    system_prompt=None,
):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve_tool])
    prompt = (
        system_prompt if system_prompt is not None else TOOL_CALLER_PROMPT
    )
    sys_message = [SystemMessage(content=prompt)]
    response = await llm_with_tools.ainvoke(
        sys_message + state["messages"], config
    )
    return {"messages": [response]}
