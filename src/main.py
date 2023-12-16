import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from llama_cpp import Llama
from langchain.prompts import PromptTemplate



model = "llama-2-7b-chat.Q5_K_M.gguf"
template = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{history}

Human:{question}
Assistant:
[/INST]
"""

prompt_template = PromptTemplate.from_template(template=template)
memory = ConversationBufferMemory()


print(f"Loading the model {model} ...")
llm = Llama(model_path=f"models/{model}", chat_format="llama-2")


st.title('DocTalk 1.0')
with st.sidebar:
    st.markdown('## About Us')
    st.write('Made with ♥️ by [Ehsan Zanjani](http://www.ehsanmx.com)')



# Function: main
def main():
    """
    Docstring for main.
    """
    st.header('Upload your PDF files and start chatting with them!') 
    pdf = st.file_uploader('Upload your PDF file', type='pdf')
    if pdf is not None:
        st.info("Your file is uploaded successfully!", icon='ℹ️')
        pdf_reader = PdfReader(pdf)
        text = '';
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len)
        chunks = text_splitter.split_text(text=text)
        
        embedding = HuggingFaceEmbeddings(
            model_name = "intfloat/e5-base-v2",
            model_kwargs = {'device': 'cpu'})

        db = FAISS.from_texts(texts=chunks, embedding=embedding)
        docs = db.similarity_search(query="How Publish & Synchronization Works?", k=3)
        llm_gen(docs)
            


# Function: llm_gen
def llm_gen(docs):
    """
    Docstring for llm_gen.
    """
    llm = LlamaCpp(
        model_path=f"models/{model}",
        temperature=0.75,
        max_tokens=4096,
        n_ctx=4096,
        top_p=1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        verbose=True,  # Verbose is required to pass to the callback manager
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    prompt = st.chat_input("Say something")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role":"user","content": prompt})
        with st.chat_message("assistant"):
            with st.spinner('Thinking...'):
                full_response = []
                placeholder = st.empty()
                formatted_prompt = prompt_template.format(history=docs, question=prompt)
                for wordstream in llm.stream(formatted_prompt):
                    if wordstream:
                        full_response.append(wordstream)
                        result = "".join(full_response).strip()
                        placeholder.markdown(result)

                st.session_state.output_text = "".join(full_response).strip()
                st.toast("Processing complete!", icon='✅')
                st.spinner('Complete')
       
        st.session_state.messages.append({"role":"assistant","content": result})



# Main entry point for the Streamlit application
if __name__ == '__main__':
    # Consider refactoring this code block into a separate function
    main()