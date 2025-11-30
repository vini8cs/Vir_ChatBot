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
        vectorstore_path,
        llm_model,
        embedding_model,
        limit,
        temperature,
        max_retries,
        thread_id,
        user_id,
        checkpointer,
    ):
        self.vectorstore_path = vectorstore_path
        self.checkpointer = checkpointer
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=temperature, max_retries=max_retries)
        self.limit = limit
        self.thread_id = thread_id
        self.user_id = user_id
        self.graph = None
        self.vectorstore = None
        self.retriever = None

    async def load_vectorstore(self):
        self.vectorstore = await asyncio.to_thread(
            FAISS.load_local,
            self.vectorstore_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.limit})

    async def build_graph(self):
        await self.load_vectorstore()
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


async def create_graph(config):
    checkpointer = AsyncSqliteSaver.from_conn_string(_.SQLITE_MEMORY_DATABASE)

    bot = Vir_ChatBot(
        vectorstore_path=_.VECTORSTORE_PATH,
        llm_model=_.GEMINI_MODEL,
        embedding_model=_.EMBEDDING_MODEL,
        limit=_.RETRIEVER_LIMIT,
        temperature=_.TEMPERATURE,
        max_retries=_.MAX_RETRIES,
        thread_id=config["configurable"].get("thread_id", _.THREAD_NUMBER),
        user_id=config["configurable"].get("user_id", _.USER_ID),
        checkpointer=checkpointer,
    )

    return await bot.build_graph()
