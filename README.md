# DocTalk

## Description
This project enables the transformation of PDF documents into a conversational interface using LLMs. It extracts and tokenizes the content of PDF files, applies embedding techniques and vector search to organize the data efficiently, and then allows local LLMs to use this processed data for generating interactive conversational responses.

### Key Features
- **PDF Processing**: Extract text from PDF documents.
- **Data Tokenization**: Apply embedding techniques for efficient data handling.
- **Vector Search**: Facilitate efficient searching within the tokenized data.
- **LLM Integration**: Use local Large Language Models for generating conversational responses based on PDF content.

## Installation
To install and set up the project, follow these steps:

1. **Clone the Repository**:
   ```
   git clone https://github.com/ehsanmx/DocTalk.git
   ```

2. **Environment Setup**:
   - It's recommended to use a virtual environment for Python.
   ```
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   - Use `requirements.txt` to install the required packages.
   ```
   pip install -r requirements.txt
   ```

4. **Additional Configuration**:
   - Ensure all necessary models and environments are properly set up as per the project's requirements.

## Usage
To use the application, follow these steps:

1. **Start the Application**: Run the main Python script via Streamlit command.
   ```
   streamlit run src/main.py
   ```

2. **Uploading PDFs**: Use the application to upload PDF documents.

3. **Interacting with PDF Content**:
   - The application will process the uploaded PDFs and allow interaction via a conversational interface.

### Example Usage
- **Example 1**: Uploading a PDF and retrieving specific information through conversation.
- **Example 2**: Using the conversational interface to summarize the content of an uploaded PDF.

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Commit your changes and push to the branch.
4. Submit a pull request with a clear description of your changes.

## License
This project is licensed under the MIT License. See the LICENSE file in the repository for more details.

