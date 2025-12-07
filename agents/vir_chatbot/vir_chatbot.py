import asyncio
from functools import partial

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import config as _
from langgraph_functions import make_retrieve_tool, query_or_respond


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


async def load_global_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model=_.EMBEDDING_MODEL)
    vectorstore = await asyncio.to_thread(
        FAISS.load_local,
        _.VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": _.RETRIEVER_LIMIT})


async def create_graph(global_retriever):
    checkpointer_cm = AsyncSqliteSaver.from_conn_string(_.SQLITE_MEMORY_DATABASE)
    checkpointer = await checkpointer_cm.__aenter__()

    bot = Vir_ChatBot(
        retriever=global_retriever,
        llm_model=_.GEMINI_MODEL,
        temperature=_.TEMPERATURE,
        max_retries=_.MAX_RETRIES,
        checkpointer=checkpointer,
    )

    graph = await bot.build_graph()
    return graph, checkpointer_cm
