import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.chains.conversation.memory import ConversationBufferMemory
from llama_cpp import Llama
from langchain.prompts import PromptTemplate

class DocTalkApp:
    # TODO move this to config
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

    def __init__(self):
        self.main()

    def main(self):        
        self.init_ui()
        st.sidebar.markdown(f"* Loading the model {self.model} is complete.")
        db = self.init_pdf_uploader()
        self.llm_gen(db=db)

    def init_ui(self):
        with st.sidebar:
            st.title('DocTalk 1.0')
            st.write('Made with ♥️ by [Ehsan Zanjani](https://www.linkedin.com/in/ezanjani/)')
            st.markdown('## Logs')
            
    
    def init_pdf_uploader(self):
        uploaded_file = st.file_uploader('Upload your PDF file', type='pdf')
        if uploaded_file is not None:
            st.info("Your file is uploaded successfully!", icon='ℹ️')
            st.markdown("""
            ```
                    Start Analyzing the file ...
            ```
                        """)
            db = self.create_embedding_db(pdf=uploaded_file)
            st.session_state.db = db
            st.markdown("""
            ```
                    Analyzing the file is complete.
            ```
                        """)
            return db
    
    def create_embedding_db(self, pdf: PdfReader) -> FAISS:
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

        return FAISS.from_texts(texts=chunks, embedding=embedding)            


    # Function: llm_gen
    def llm_gen(self, db :FAISS):
        # llm = Llama(model_path=f"model/{model}", chat_format="llama-2")
        self.llm = LlamaCpp(
            model_path=f"model/{self.model}",
            temperature=0.75,
            max_tokens=4096,
            n_ctx=4096,
            top_p=1,
            streaming=True,            
            callbacks=[StreamingStdOutCallbackHandler()],
            verbose=True,  # Verbose is required to pass to the callback manager
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
                    docs = []
                    if 'db' in st.session_state:
                        db = st.session_state.db
                        docs = db.similarity_search(query=prompt, k=3)
                    print(docs)
                    formatted_prompt = self.prompt_template.format(history=docs, question=prompt)
                    for wordstream in self.llm.stream(formatted_prompt):
                        if wordstream:
                            full_response.append(wordstream)
                            result = "".join(full_response).strip()
                            placeholder.markdown(result)

                    st.session_state.output_text = "".join(full_response).strip()
                    st.toast("Processing complete!", icon='✅')
                    st.spinner('Complete')
        
            st.session_state.messages.append({"role":"assistant","content": result})


# Running the app
if __name__ == "__main__":
   dt = DocTalkApp()