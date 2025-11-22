# import logging
from functools import partial

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import config as _
from langgraph_functions import make_retrieve_tool, query_or_respond


class Chat:
    def __init__(
        self,
        vectorstore_path,
        sqlite_memory_database=_.SQLITE_MEMORY_DATABASE,
        llm_model=_.GEMINI_MODEL,
        embedding_model=_.EMBEDDING_MODEL,
        limit=_.RETRIEVER_LIMIT,
        temperature=_.TEMPERATURE,
        max_retries=_.MAX_RETRIES,
        thread=_.THREAD_NUMBER,
        user_id=_.USER_ID,
    ):
        self.vectorstore_path = vectorstore_path
        self.sqlite_memory_database = sqlite_memory_database
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=temperature, max_retries=max_retries)
        self.limit = limit
        self.thread = thread
        self.user_id = user_id

        self.vectorstore = FAISS.load_local(
            self.vectorstore_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.limit})

    def create_graph(self, checkpointer):
        graph_builder = StateGraph(MessagesState)

        retrieve_tool = make_retrieve_tool(self.retriever)

        node_query_or_respond = partial(query_or_respond, llm=self.llm, retrieve_tool=retrieve_tool)

        tools = ToolNode([retrieve_tool])

        graph_builder.add_node("node_query_or_respond", node_query_or_respond)
        graph_builder.add_node("tools", tools)

        graph_builder.set_entry_point("node_query_or_respond")
        graph_builder.add_conditional_edges(
            "node_query_or_respond",
            tools_condition,
            {"tools": "tools", END: END},
        )
        graph_builder.add_edge("tools", "node_query_or_respond")
        graph_builder.add_edge("node_query_or_respond", END)

        return graph_builder.compile(checkpointer=checkpointer)

    async def run_chat(self):
        print("Starting chat test...")
        async with AsyncSqliteSaver.from_conn_string(self.sqlite_memory_database) as checkpointer:

            graph = self.create_graph(checkpointer)

            self.config = {"configurable": {"thread_id": self.thread, "user_id": self.user_id}}

            while True:
                user_input = input("You: ")
                if not user_input and not user_input.strip():
                    print("Empty input, please enter a message.")
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    break

                async for step in graph.astream(
                    {
                        "messages": [{"role": "user", "content": user_input}],
                        "preferences": {"theme": "dark"},
                    },
                    self.config,
                    stream_mode="values",
                ):
                    step["messages"][-1].pretty_print()


# import asyncio

# if __name__ == "__main__":
#     chat = Chat(
#         vectorstore_path="/mnt/vini/VirBotAI/vectorstore",
#         sqlite_memory_database="/mnt/vini/VirBotAI/memory.sqlite",
#     )
#     asyncio.run(chat.test())
