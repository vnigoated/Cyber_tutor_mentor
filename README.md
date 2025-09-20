# Cybersecurity AI Mentor

A ChatGPT-like cybersecurity learning companion with conversation memory and RAG-powered responses using local documents.

## 🏗️ Project Structure

The application has been modularized for better maintainability and easier development:

```
Cyber_tutor_mentor/
├── main.py                    # Main entry point
├── config.py                  # Configuration settings
├── rag_engine.py             # RAG functionality (documents, indexing, queries)
├── conversation_manager.py   # Chat history and context management
├── ui_interface.py           # Gradio UI components and interface
├── utils.py                  # Utility functions (export, validation, etc.)
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (API keys)
├── README.md                 # This file
└── docs/                     # Directory containing cybersecurity documents
    ├── *.pdf                 # PDF documents
    └── *.txt                 # Text documents
```

## 📁 Module Overview

### `main.py`
- **Purpose**: Application entry point
- **Responsibilities**: 
  - Configuration validation
  - Application startup and error handling
  - Gradio app launching

### `config.py`
- **Purpose**: Centralized configuration management
- **Contains**:
  - API keys and credentials
  - Model configurations
  - UI settings
  - Application constants

### `rag_engine.py`
- **Purpose**: RAG (Retrieval-Augmented Generation) functionality
- **Responsibilities**:
  - Document loading and processing
  - Vector index creation and management
  - Query processing and context retrieval
  - Embedding model management

### `conversation_manager.py`
- **Purpose**: Conversation state management
- **Responsibilities**:
  - Chat history storage
  - Context extraction for follow-up questions
  - Conversation statistics and analytics
  - History search and filtering

### `ui_interface.py`
- **Purpose**: Gradio user interface
- **Responsibilities**:
  - UI component creation and layout
  - Event handling (chat, clear, export)
  - User input validation
  - Response formatting

### `utils.py`
- **Purpose**: Utility functions
- **Contains**:
  - File export functionality
  - Input validation
  - Response formatting
  - System status monitoring
  - Cleanup functions

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- GROQ API key

### Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Your `.env` file should contain:
   ```
   GROQ_API_KEY=your-groq-api-key-here
   ```

4. **Prepare documents**:
   Place your cybersecurity documents (PDF, TXT) in the `docs/` directory

5. **Run the application**:
   ```bash
   python main.py
   ```

## 🎯 Features

- **🤖 ChatGPT-like Interface**: Natural conversation flow with memory
- **📚 RAG-Powered Responses**: Uses your local cybersecurity documents
- **🧠 Conversation Memory**: Remembers context across messages
- **🎓 Adaptive Learning**: Adjusts responses based on user expertise level
- **� Smart Scope**: Focuses on cybersecurity topics, politely redirects off-topic questions
- **�📤 Export Conversations**: Save chat history to files
- **🔍 Input Validation**: Security checks and content filtering
- **📊 System Monitoring**: Status tracking and error logging

## 🔧 Configuration

Edit `config.py` to customize:

- **Model Settings**: Change embedding or LLM models
- **RAG Parameters**: Adjust similarity search, context length
- **UI Settings**: Modify theme, layout, sample questions
- **Export Settings**: Configure file naming, encoding

## 🛠️ Development

### Adding New Features

1. **New UI Components**: Modify `ui_interface.py`
2. **RAG Improvements**: Update `rag_engine.py`
3. **Conversation Features**: Extend `conversation_manager.py`
4. **Utility Functions**: Add to `utils.py`
5. **Configuration**: Update `config.py`

### Customization Examples

#### Adding a New User Level
```python
# In config.py
USER_LEVELS = ["beginner", "intermediate", "expert", "professional"]
```

#### Changing the Model
```python
# In config.py
LLM_MODEL_NAME = "llama-3.2-90b-text-preview"  # Different Groq model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Better embeddings
```

## 🐛 Troubleshooting

### Common Issues

1. **"GROQ_API_KEY not found"**
   - Check your `.env` file
   - Ensure the file is in the project root

2. **"Documents directory not found"**
   - Create the `docs/` directory
   - Add some PDF or TXT files

3. **Import errors**
   - Run `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

4. **Memory issues with large documents**
   - Reduce `SIMILARITY_TOP_K` in config
   - Split large documents into smaller files

## 📈 Performance Tips

- **Document Optimization**: Use text files when possible (faster than PDFs)
- **Index Caching**: The vector index is built once on startup
- **Memory Management**: Conversation history is kept in memory
- **Concurrent Users**: Current setup is single-user

## 🔒 Security Considerations

- **Input Validation**: Basic filtering implemented in `utils.py`
- **API Keys**: Stored in environment variables
- **File Access**: Limited to configured directories
- **Content Filtering**: Basic patterns in `validate_user_input()`

## 🤝 Contributing

1. Follow the modular structure
2. Add proper logging and error handling
3. Update this README for new features
4. Test with different document types and user levels