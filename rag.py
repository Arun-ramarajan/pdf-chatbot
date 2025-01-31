MODEL = "gpt-4o-mini"
db_name = "vector_db"
api_key = "openai api key"

# Function to extract text and titles from the PDF using pdfplumber
def extract_text_and_titles_with_bold(pdf_file):
    documents = []

    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            title = None

            # Get all the words along with their font size
            words = page.extract_words()

            # Iterate over words and check if they contain 'Bold' in their font name
            for word in words:
                try:
                    if 'Bold' in word.get('fontname', ''):
                        title = word['text']
                        break
                except KeyError:
                    continue

            # Fallback to the first line if no bold title is found
            if not title:
                title = text.split('\n')[0]  # First line of text

            # Create a Document object with title and content
            document = Document(page_content=text, metadata={"title": title, "page_num": page_num + 1})
            documents.append(document)

    return documents

# Initialize Streamlit app
st.title("PDF Chatbot - Chat About Your PDF")

# File uploader widget for PDF files
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_file is not None:
    # Extract text from the uploaded PDF file
    documents = extract_text_and_titles_with_bold(pdf_file)

    # Create embeddings for the documents
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Set up the vector store (Chroma) for storing the embeddings
    db_name = "pdf_vectorstore"
    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
        print('Deleted previous vectorstore')

    # Create a new vector store from the documents
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=db_name)
    st.success(f"Vectorstore created with {vectorstore._collection.count()} documents")

    # Create the OpenAI language model
    llm = ChatOpenAI(temperature=0.7, model_name=MODEL, openai_api_key=api_key)

    # Set up the conversation memory for the chat
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Set up the retriever for the vector store
    retriever = vectorstore.as_retriever()

    # Set up the conversation chain with the retriever, memory, and LLM
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def generate_response(user_message):
        # Append user message to session state
        st.session_state.messages.append({"role": "user", "content": user_message})

        # Get the response from the model
        result = conversation_chain.invoke({"question": user_message})

        # Append model response to session state
        st.session_state.messages.append({"role": "assistant", "content": result['answer']})

    # Display chat messages after checking session state
    if st.session_state.messages:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")

    # User input for new message
    user_message = st.text_input("Your message:")

    # Process new user input only when they submit a message
    if user_message:        
        generate_response(user_message)
