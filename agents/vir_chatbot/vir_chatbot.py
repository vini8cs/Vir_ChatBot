import asyncio
import logging
import os
from functools import partial

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import config as _
from llms.langgraph_functions import make_retrieve_tool, query_or_respond


class Vir_ChatBot:
    def __init__(
        self,
        retriever,
        llm_model,
        temperature,
        max_retries,
        checkpointer,
    ):
        self.retriever = retriever
        self.checkpointer = checkpointer
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=temperature, max_retries=max_retries)
        self.graph = None

    async def build_graph(self):
        builder = StateGraph(MessagesState)

        retrieve_tool = make_retrieve_tool(self.retriever)
        node = partial(query_or_respond, llm=self.llm, retrieve_tool=retrieve_tool)

        tools = ToolNode([retrieve_tool])

        builder.add_node("query_or_respond", node)
        builder.add_node("tools", tools)

        builder.set_entry_point("query_or_respond")

        builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {"tools": "tools", END: END},
        )
        builder.add_edge("tools", "query_or_respond")
        builder.add_edge("query_or_respond", END)

        return builder.compile(checkpointer=self.checkpointer)


async def load_global_vectorstore(retriever_limit: int | None = None):
    vectorstore_index = os.path.join(_.VECTORSTORE_PATH, "index.faiss")
    if not os.path.exists(vectorstore_index):
        logging.info(f"VectorStore not found at {_.VECTORSTORE_PATH}. It will be created when PDFs are added.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model=_.EMBEDDING_MODEL)
    logging.info("Loading VectorStore...")
    vectorstore = await asyncio.to_thread(
        FAISS.load_local,
        _.VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    k = retriever_limit if retriever_limit is not None else _.RETRIEVER_LIMIT
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


async def create_graph(
    global_retriever,
    llm_model: str | None = None,
    temperature: float | None = None,
    max_retries: int | None = None,
):
    # Use WAL mode connection string for better concurrency
    db_path = _.SQLITE_MEMORY_DATABASE
    conn_string = f"file:{db_path}?mode=rwc"

    checkpointer_cm = AsyncSqliteSaver.from_conn_string(conn_string)
    checkpointer = await checkpointer_cm.__aenter__()

    # Enable WAL mode for better concurrent access
    if hasattr(checkpointer, "conn") and checkpointer.conn:
        await checkpointer.conn.execute("PRAGMA journal_mode=WAL;")
        await checkpointer.conn.execute("PRAGMA busy_timeout=30000;")  # 30 second timeout

    bot = Vir_ChatBot(
        retriever=global_retriever,
        llm_model=llm_model if llm_model is not None else _.GEMINI_MODEL,
        temperature=temperature if temperature is not None else _.TEMPERATURE,
        max_retries=max_retries if max_retries is not None else _.MAX_RETRIES,
        checkpointer=checkpointer,
    )

    graph = await bot.build_graph()
    return graph, checkpointer_cm
