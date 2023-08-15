import os
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import DuckDuckGoSearchRun

load_dotenv()

# page configuration
st.set_page_config(page_title='Convo-Agency', page_icon=":zap:")
st.title('Convo-Agency')

# intialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'How can I assist you?'}]

for message in st.session_state.messages:
    st.chat_message(message['role']).write(message['content'])

llm = OpenAI(temperature=0, streaming=True, openai_api_key=os.getenv("OPENAI_API_KEY"))
tools = [DuckDuckGoSearchRun(name="Search")] # alternative [load_tools(["ddg-search"])]
agent = initialize_agent(
    tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)


if prompt := st.chat_input(placeholder='Ask a question'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    st.chat_message('user').write(prompt)
    with st.chat_message("assistant"):
        # st.write("🧠 thinking...")
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt)
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.write(response)

