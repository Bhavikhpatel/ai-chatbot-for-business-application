# 🤖 AI Business Assistant

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33.0-orange.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.16-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An all-in-one AI-powered business assistant built with Streamlit. This application integrates multiple tools to streamline tasks like communication, task management, financial analysis, code generation, and staying informed with curated news.

## ✨ Key Features

This multi-tool application is organized into a clean, accessible sidebar menu.

### 💬 1. Conversational AI Assistant
A powerful chat assistant that remembers the context of your conversation.
- **Powered by Groq:** Utilizes the Llama3-8B model for incredibly fast and coherent responses.
- **Conversational Memory:** Remembers the last 5 turns of your conversation for follow-up questions.
- **Built with LangChain:** Leverages LangChain for robust and scalable conversation management.

<br>

### 📋 2. Smart To-Do List Generator
Break down complex projects into actionable steps.
- **AI-Powered Task Generation:** Simply describe a high-level goal (e.g., "launch a new website"), and the AI will generate a detailed to-do list.
- **Interactive Checklist:** Track your progress with interactive checkboxes.
- **Progress Bar:** Visualize your completion rate as you check off tasks.

<br>

### 📊 3. Financial Analytics Dashboard
Gain insights from your sales data with an intuitive analytics dashboard.
- **Upload & Analyze:** Upload your sales data as a CSV or Excel file.
- **Key Metrics:** Instantly view crucial metrics like Total Sales, Average Sale Value, Total Units Sold, and more.
- **Dynamic Visualizations:** The app generates several plots to help you understand your business:
    - Monthly Sales Trends (Line Chart)
    - Product Sales Distribution (Pie Chart)
    - Payment Method Popularity (Bar Chart)
    - Customer Purchase Frequency (Bar Chart)
- **Data Schema:** Requires specific columns in the input file for accurate analysis (see [Setup](#-financial-data-format)).

<br>

### 📰 4. Curated News & Alerts
Stay updated with the latest news and security information.
- **Latest News:** Fetches top news headlines from India.
- **Scam Alerts:** Provides a dedicated feed for recent news related to scams to keep you and your business safe.
- **Powered by NewsData.io:** Aggregates news from thousands of sources.

<br>

### 📱 5. Supplier Messenger
Communicate directly with your suppliers from within the app.
- **Twilio Integration:** Send SMS messages to predefined contacts.
- **Simple Interface:** Select a supplier, type your message, and hit send.
- **Pre-configured Contacts:** Easily manage a list of important contacts within the code.

<br>

### 💻 6. Business Code Generator
Accelerate your development process by generating production-ready code.
- **Multi-Language Support:** Generate code in Python, JavaScript, HTML/CSS, React, SQL, and more.
- **Requirement-Based:** Describe your project requirements in natural language.
- **Best Practices:** The AI is prompted to produce clean, well-documented, and efficient code with proper error handling.
- **Downloadable Code:** Download the generated code snippet as a file.

## 🛠️ Tech Stack

- **Framework:** Streamlit
- **Language:** Python
- **LLM & Orchestration:** Groq (Llama3-8B-8192), LangChain
- **Data Analysis:** Pandas, Matplotlib
- **APIs & Services:**
    - [Twilio API](https://www.twilio.com/) for SMS messaging.
    - [NewsData.io API](https://newsdata.io/) for news aggregation.
    - [Groq API](https://groq.com/) for LLM inference.

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### 1. Prerequisites
- Python 3.9 or higher
- Git
- API keys for Groq, NewsData.io, and Twilio.

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ai-business-assistant.git
    cd ai-business-assistant
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    If a `requirements.txt` is not available, install the packages manually:
    ```bash
    pip install streamlit pandas matplotlib twilio groq langchain langchain-core langchain-groq requests openpyxl
    ```

### 3. Configuration

This project requires API keys to function. You must set them as environment variables.

1.  **Obtain API Keys:**
    - **GROQ_API_KEY:** Get it from the [GroqCloud Console](https://console.groq.com/keys).
    - **NEWS_API_KEY:** Get it from [NewsData.io](https://newsdata.io/register).
    - **SID & MTOKEN:** These are your **Twilio Account SID** and **Auth Token**. Get them from your [Twilio Console](https://www.twilio.com/console).

2.  **Set Environment Variables:**

    **For macOS/Linux:**
    ```bash
    export GROQ_API_KEY="your_groq_api_key"
    export NEWS_API_KEY="your_newsdata_api_key"
    export SID="your_twilio_account_sid"
    export MTOKEN="your_twilio_auth_token"
    ```
    To make them permanent, add these lines to your `.bashrc`, `.zshrc`, or shell configuration file.

    **For Windows (Command Prompt):**
    ```cmd
    set GROQ_API_KEY="your_groq_api_key"
    set NEWS_API_KEY="your_newsdata_api_key"
    set SID="your_twilio_account_sid"
    set MTOKEN="your_twilio_auth_token"
    ```
    For PowerShell, use `$env:VAR_NAME="value"`. To set them permanently, use the "Edit the system environment variables" control panel.

## ▶️ Usage

Once the installation and configuration are complete, run the Streamlit app with the following command:

```bash
streamlit run app.py
