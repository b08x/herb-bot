# Herbalism Research Assistant

A Streamlit application for herbalism research that integrates Gemini models, document analysis, Google Custom Search, and data visualization.

![Herbalism Research Assistant](https://img.icons8.com/color/96/000000/herbal-medicine.png)

## Features

- **AI Chat Interface**: Interact with Google's Gemini models specialized in clinical herbalism
- **Document Analysis**: Upload and analyze PDF and DOCX files
- **Research Search**: Search for herbalism information using Google Custom Search API
- **Data Visualization**: Visualize research findings with interactive charts
- **Fact Checking**: Verify herbalism claims with reliable sources

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/streamlit_herbalism_and_research_chat.git
   cd streamlit_herbalism_and_research_chat
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   GOOGLE_SEARCH_API_KEY=your_google_search_api_key_here
   GOOGLE_SEARCH_ENGINE_ID=your_google_search_engine_id_here
   ```

## Getting API Keys

- **Gemini API Key**: Get it from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Google Search API Key**: Get it from [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
- **Google Search Engine ID**: Create a programmable search engine at [Programmable Search Engine](https://programmablesearchengine.google.com/about/)

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Application Structure

```
streamlit_herbalism_and_research_chat/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Project dependencies
├── .env                        # Environment variables (API keys)
├── .gitignore                  # Git ignore file
├── README.md                   # Project documentation
├── components/                 # UI components
│   ├── chat.py                 # Chat interface
│   ├── document_upload.py      # Document upload interface
│   ├── search.py               # Search interface
│   └── visualization.py        # Visualization components
├── services/                   # Service integrations
│   ├── gemini_service.py       # Gemini API integration
│   ├── search_service.py       # Google Custom Search integration
│   └── document_service.py     # Document processing service
├── utils/                      # Utility functions
│   ├── text_extraction.py      # Text extraction utilities
│   ├── context_management.py   # Context management utilities
│   └── visualization_utils.py  # Visualization utilities
└── data/                       # Data storage
    └── uploads/                # Uploaded documents storage
```

## Features in Detail

### Chat Interface

The chat interface allows you to interact with the Gemini model using a specialized system prompt for clinical herbalism. You can:

- Ask questions about herbs and their properties
- Get recommendations for specific health concerns
- Discuss research findings and traditional uses
- Incorporate uploaded document content into the conversation

### Document Management

The document management system allows you to:

- Upload PDF and DOCX files
- Extract and view text content
- Search within documents
- Use document content as context for chat conversations

### Search Functionality

The search functionality provides:

- General herbalism research search
- Specialized herb information search
- Herb-drug interaction search
- Fact-checking capabilities

### Data Visualization

The visualization tools include:

- Term frequency analysis
- Herb-property network visualization
- Herb property comparison charts
- Historical timeline visualization
- Document herb mention analysis

## System Prompt

The application uses a specialized system prompt for the Gemini model, focusing on clinical herbalism with structured knowledge domains including:

- Human Sciences
- Nutrition
- Phytochemistry
- Botany
- Plant Science
- Evidence-Based Botanicals

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.