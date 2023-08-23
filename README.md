# Offline AI

# Overview:

The purpose of this code is to develop a streamlit app that allows the user
to have a conversation using a local LLM regarding data that is uploaded.




# How to Set Up:

Install privategpt, here is the repo with instructions and a video tutorial on how to do it.
https://github.com/imartinez/privateGPT
https://www.youtube.com/watch?v=G7iLllmx4qc

Once installed and the local embeddings are run through ingest.py, the following streamlit app will work.
Make sure to run the embeddings(ingest.py) in the same directory as the app.py from this repo.
Make sure to check the model paths in the .env file are correct.

The model that im using right now is ggml-gpt4all-j-v1.3-groovy.bin.

# Currently ( 8-23-23 )

User can input PDFs, and have conversation about that data using local llms.

Last message is displayed with markdown.
When a message is stored into the history, it shows its sources.

# Future:

- Swap between Models, and create custom prompts using Forest of thoughts method.

- Add a page where the User will be able to download information from
websites and create files that can be passed into llm. Also the user will
be able to enter a url and then chat, based on the contents of that url.

- Add a visualization tab to the app, utilizing pygwalker.

- Export and view conversational history.


