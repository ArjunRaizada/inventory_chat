import streamlit as st
from pathlib import Path
from sqlalchemy import create_engine
import sqlite3

from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_groq import ChatGroq

st.set_page_config(page_title="Chat with your inventory", page_icon="")
st.title("Chat with your inventory")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

options = [
    "Use SQLite3 Database - inventory.db",
    "Connect to MySQL Database"
]
selected = st.sidebar.radio("Choose the DB to chat", options)

if selected == options[1]:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = LOCALDB

api_key = st.secrets("API_KEY")

@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        db_path = (Path(__file__).parent / "inventory.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        engine = create_engine("sqlite://", creator=creator)
        return SQLDatabase(engine, include_tables=["Inventory"])
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        engine = create_engine(
            f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
        )
        return SQLDatabase(engine, include_tables=["Inventory"])

if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)

llm = ChatGroq(groq_api_key=api_key, model_name="llama3-70b-8192", streaming=True)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input("Ask anything about the Inventory table")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    streamlit_callback = StreamlitCallbackHandler(st.container())
    response = agent.run(user_query, callbacks=[streamlit_callback])
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
