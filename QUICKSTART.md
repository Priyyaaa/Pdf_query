# Quick Start Guide

## Setup Steps

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys**:
   - Create a `.env` file in the project root
   - Add at least one API key (you only need the provider you want to use):
     ```
     GOOGLE_API_KEY=your_key_here
     # OR
     GROQ_API_KEY=your_key_here
     # OR
     COHERE_API_KEY=your_key_here
     ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Use the application**:
   - Upload a PDF document
   - Select your preferred LLM provider in the sidebar
   - Start asking questions!

## Getting API Keys

- **Google Gemini**: https://makersuite.google.com/app/apikey (Free tier available)
- **Groq**: https://console.groq.com/ (Free tier available, very fast)
- **Cohere**: https://cohere.com/ (Free tier available)

## Troubleshooting

- If you get import errors, make sure all packages are installed: `pip install -r requirements.txt`
- If API errors occur, verify your API key is correct in the `.env` file
- For PDF processing errors, ensure the PDF is not password-protected


