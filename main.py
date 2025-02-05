import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from twilio.rest import Client
from groq import Groq
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import requests
import ast

class ChatBot:
    def __init__(self):
        self.apikey = os.getenv('GROQ_API_KEY')
        
        try:
            self.client = Groq(api_key=self.apikey)
        except Exception as e:
            st.error(f"Error initializing Groq client: {e}")
            self.client = None

    def get_response(self, message):
        if not self.client:
            return "API client not initialized. Please check your API key."
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error getting response: {e}"

class ChatBot2:
    def __init__(self):
        self.apikey = os.getenv('GROQ_API_KEY')
        self.model = 'llama3-8b-8192'
        
        try:
            self.groq_chat = ChatGroq(
                    groq_api_key=self.apikey, 
                    model_name=self.model
            )
        except Exception as e:
            st.error(f"Error initializing Groq chat: {e}")
            self.groq_chat = None
        
        self.promptInstruct = None
        self.promptInstructFlag = False
    
    def get_response(self, user_question, chat_history=None):
        if not self.groq_chat:
            return "Chat client not initialized. Please check your API key."
        
        if chat_history is None:
            chat_history = []

        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
        for msg in chat_history:
            memory.chat_memory.add_user_message(msg['human'])
            memory.chat_memory.add_ai_message(msg['ai'])

        conversation = LLMChain(
            llm=self.groq_chat,
            prompt=prompt,
            verbose=False,
            memory=memory,
        )

        try:
            response = conversation.predict(human_input=user_question)
            return response
        except Exception as e:
            return f"Error in conversation: {e}"

class SmartToDo:
    def __init__(self):
        self.chatBot = ChatBot()

    def createToDo(self, task):
        def strip_text_between_brackets(text):
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1:
                return text[start:end+1]
            return "Brackets not found or incomplete"

        prompt = f"""input:{task}

context: i want to create a todo list.

goal: your job is to create a detailed todo list from the task i have provided

examples:
input: "i want to make a car"
output: "[
    "Research car design and engineering principles",
    "Determine the purpose and type of the car",
    "Create a budget for the car project",
    "Sketch the design of the car",
    "Choose materials for the car's body and interior",
    "Learn or hire expertise for mechanical and electrical systems",
    "Build the chassis and frame",
    "Assemble the engine and drivetrain",
    "Install electrical systems and wiring",
    "Test and refine the car's performance",
    "Ensure compliance with safety and regulatory standards",
    "Paint and finalize the car's exterior",
    "Add finishing touches to the interior",
    "Conduct final quality and functionality tests",
    "Prepare for showcasing or using the car"
]"
"""

        response = self.chatBot.get_response(prompt)
        striptedText = strip_text_between_brackets(response)

        try:
            task_list = ast.literal_eval(striptedText)
            return task_list
        except:
            return ["Unable to generate todo list. Please try again."]

class CuratedNews:
    def __init__(self):
        self.apikey = os.getenv('NEWS_API_KEY')
        self.endpoint = f"https://newsdata.io/api/1/latest?apikey={self.apikey}&q=india"
    
    def getNews(self):
        try:
            response = requests.get(url=self.endpoint)
            if response.status_code == 200:
                news_data = response.json()
                return news_data.get('results', [])
            else:
                st.error(f"Failed to fetch news. Status code: {response.status_code}")
                return []
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            return []

class FinancialAnalytics:
    def __init__(self, df):
        self.df = df
        self.df['date'] = pd.to_datetime(self.df['date'])
    
    def generate_insights(self):
        insights = {
            'total_sales': self.df['TotalAmount'].sum(),
            'average_sale_value': self.df['TotalAmount'].mean(),
            'total_units_sold': self.df['NoUnitsPurchased'].sum(),
            'unique_products': self.df['productName'].nunique(),
            'unique_customers': self.df['costumerName'].nunique(),
            'top_products': self.df.groupby('productName')['TotalAmount'].sum().nlargest(3).to_dict(),
            'top_customers': self.df.groupby('costumerName')['TotalAmount'].sum().nlargest(3).to_dict(),
            'payment_method_distribution': self.df['paymentMethod'].value_counts(normalize=True).to_dict(),
            'monthly_sales': self.df.groupby(pd.Grouper(key='date', freq='M'))['TotalAmount'].sum().to_dict()
        }
        return insights
    
    def visualize_insights(self):
        plt.figure(figsize=(20,15))
        
        plt.subplot(2, 2, 1)
        monthly_sales = self.df.groupby(pd.Grouper(key='date', freq='M'))['TotalAmount'].sum()
        monthly_sales.plot(kind='line', title='Monthly Sales Trend')
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        product_sales = self.df.groupby('productName')['TotalAmount'].sum()
        product_sales.plot(kind='pie', autopct='%1.1f%%', title='Product Sales Distribution')
        
        plt.subplot(2, 2, 3)
        payment_dist = self.df['paymentMethod'].value_counts()
        payment_dist.plot(kind='bar', title='Payment Method Distribution')
        plt.xlabel('Payment Method')
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        customer_purchases = self.df.groupby('costumerName')['TotalAmount'].count()
        customer_purchases.plot(kind='bar', title='Customer Purchase Frequency')
        plt.xlabel('Customer Name')
        plt.ylabel('Number of Purchases')
        plt.xticks(rotation=90)
        
        plt.tight_layout()
        return plt

class Messenger:
    def __init__(self):
        self.account_sid = os.getenv("SID")
        self.auth_token = os.getenv("MTOKEN")
        self.from_number = "+17756287361"
        
        try:
            self.client = Client(self.account_sid, self.auth_token)
        except Exception as e:
            st.error(f"Error initializing Twilio client: {e}")
            self.client = None
    
    def send_message(self, to_number, message):
        if not self.client:
            return "Messaging client not initialized. Please check credentials."
        
        try:
            message = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            return "Message sent successfully!"
        except Exception as e:
            return f"Error sending message: {e}"

class CodeGenerator:
    def __init__(self):
        self.chatBot = ChatBot()

    def generate_code(self, requirement, language):
        prompt = f"""
context: you are a professional developer who writes clean, well-documented code.

requirement: {requirement}
programming language: {language}

goal: generate production-ready code based on the requirement. include:
1. necessary imports
2. clear comments explaining the code
3. proper error handling
4. best practices for the specified language
5. example usage if applicable

format the response as follows:
```{language}
[generated code here]
```

additional notes:
[any important implementation notes, setup instructions, or dependencies]
"""
        response = self.chatBot.get_response(prompt)
        return response

def main():
    st.set_page_config(page_title="AI Assistant App", page_icon="🤖", layout="wide")
    
    st.sidebar.header("🔑 API Configuration")
    
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", 
                                         value=os.getenv('GROQ_API_KEY', ''),
                                         type="password")
    if groq_api_key:
        os.environ['GROQ_API_KEY'] = groq_api_key
        st.sidebar.success("Groq API Key set successfully!")
    
    news_api_key = st.sidebar.text_input("Enter News API Key", 
                                         value=os.getenv('NEWS_API_KEY', ''),
                                         type="password")
    if news_api_key:
        os.environ['NEWS_API_KEY'] = news_api_key
        st.sidebar.success("News API Key set successfully!")
    
    st.sidebar.title("🤖 AI Assistant")
    app_mode = st.sidebar.radio("Choose a Feature", 
        [
            "Chat Assistant", 
            "Smart To-Do List", 
            "Financial Analytics", 
            "Curated News",
            "Messenger",
            "Code Generator"
        ])
    
    if app_mode == "Chat Assistant":
        st.title("💬 AI Chat Assistant")
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        user_input = st.chat_input("Enter your message")
        
        if user_input:
            chatbot = ChatBot2()
            
            st.session_state.chat_history.append({'human': user_input, 'ai': ''})
            
            response = chatbot.get_response(user_input, st.session_state.chat_history)
            
            st.session_state.chat_history[-1]['ai'] = response
            
            for msg in st.session_state.chat_history:
                st.chat_message("human").write(msg['human'])
                st.chat_message("assistant").write(msg['ai'])
    
    elif app_mode == "Smart To-Do List":
        st.title("📋 Smart To-Do List Generator")
        
        if 'todo_tasks' not in st.session_state:
            st.session_state.todo_tasks = []
        if 'task_states' not in st.session_state:
            st.session_state.task_states = {}
        
        todo_task = st.text_input("Enter the task you want to break down")
        
        if st.button("Generate Todo List"):
            if todo_task:
                todo_generator = SmartToDo()
                todo_list = todo_generator.createToDo(todo_task)
                
                st.session_state.todo_tasks = todo_list
                st.session_state.task_states = {
                    f"task_{i}": st.session_state.task_states.get(f"task_{i}", False) 
                    for i in range(len(todo_list))
                }
        
        if st.session_state.todo_tasks:
            st.subheader("Generated Todo List:")
            
            col1, col2 = st.columns([3, 1])
            
            for i, task in enumerate(st.session_state.todo_tasks):
                key = f"task_{i}"
                if key not in st.session_state.task_states:
                    st.session_state.task_states[key] = False
                
                st.session_state.task_states[key] = st.checkbox(
                    f"{task}",
                    value=st.session_state.task_states[key],
                    key=key
                )
            
            completed_tasks = sum(1 for state in st.session_state.task_states.values() if state)
            total_tasks = len(st.session_state.todo_tasks)
            progress = completed_tasks / total_tasks if total_tasks > 0 else 0
            
            with col1:
                st.progress(progress)
            with col2:
                st.write(f"Completed: {completed_tasks}/{total_tasks}")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Clear List"):
                    st.session_state.todo_tasks = []
                    st.session_state.task_states = {}
                    st.experimental_rerun()
            
            if st.checkbox("Show Debug Info", key="debug"):
                st.write("Task States:", st.session_state.task_states)
                st.write("Completed Tasks Count:", completed_tasks)
                st.write("Total Tasks:", total_tasks)
    
    elif app_mode == "Financial Analytics":
        st.title("📊 Financial Analytics")
        
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                required_columns = ['date', 'TotalAmount', 'NoUnitsPurchased', 'productName', 'costumerName', 'paymentMethod']
                if all(col in df.columns for col in required_columns):
                    fin_analytics = FinancialAnalytics(df)
                    
                    insights = fin_analytics.generate_insights()
                    
                    st.subheader("Financial Insights")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Sales", f"₹{insights['total_sales']:,.2f}")
                        st.metric("Average Sale Value", f"₹{insights['average_sale_value']:,.2f}")
                    
                    with col2:
                        st.metric("Total Units Sold", insights['total_units_sold'])
                        st.metric("Unique Products", insights['unique_products'])
                    
                    st.subheader("Data Visualizations")
                    fig = fin_analytics.visualize_insights()
                    st.pyplot(fig)
                
                else:
                    st.error("File is missing required columns. Please check the file format.")
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    elif app_mode == "Curated News":
        st.title("📰 Curated News")
        
        if st.button("Fetch Latest News"):
            news_fetcher = CuratedNews()
            news_articles = news_fetcher.getNews()
            
            if news_articles:
                for article in news_articles[:10]:
                    st.subheader(article.get('title', 'Untitled Article'))
                    st.write(f"Source: {article.get('source_id', 'Unknown')}")
                    st.write(f"Description: {article.get('description', 'No description')}")
                    st.write(f"Link: {article.get('link', 'No link available')}")
                    st.markdown("---")
            else:
                st.warning("No news articles found")
    
    elif app_mode == "Messenger":
        st.title("📱 Message Supplier")
        
        contacts = {
            "Raw Material Supplier": "+919606584017",
        }
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            message = st.text_area("Type your message", height=100)
            
            selected_contact = st.selectbox(
                "Select Supplier",
                options=list(contacts.keys())
            )
        
        with col2:
            st.info(f"Selected contact:\n{selected_contact}\n\nPhone: {contacts[selected_contact]}")
            
            if st.button("Send Message"):
                if message:
                    messenger = Messenger()
                    result = messenger.send_message(
                        contacts[selected_contact],
                        message
                    )
                    st.write(result)
                else:
                    st.warning("Please type a message first")
    
    elif app_mode == "Code Generator":
        st.title("💻 Business Code Generator")
        
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            st.markdown("""
            ### Describe Your Project
            Examples:
            - Create a responsive landing page for a restaurant
            - Build a REST API for inventory management
            - Develop a data visualization dashboard
            """)
            
            requirement = st.text_area(
                "Project Requirements",
                height=150,
                placeholder="Describe what you want to build..."
            )
            
            languages = [
                "Python", "JavaScript", "HTML/CSS", "React", "Node.js",
                "PHP", "Java", "C#", "SQL", "TypeScript"
            ]
            
            language = st.selectbox(
                "Select Programming Language/Framework",
                options=languages
            )
            
            generate_button = st.button("Generate Code", type="primary")
            
            with st.expander("Tips for Better Code Generation"):
                st.markdown("""
                1. **Be Specific**: Clearly describe the features and functionality you need
                2. **Provide Context**: Include any specific business rules or requirements
                3. **Specify Components**: List the main components or modules needed
                4. **Mention Integrations**: Include any external services or APIs needed
                5. **State Preferences**: Mention any specific coding style or patterns you prefer
                """)
        
        with right_col:
            if generate_button and requirement:
                with st.spinner("Generating code..."):
                    code_generator = CodeGenerator()
                    response = code_generator.generate_code(requirement, language)
                    
                    code_start = response.find("```")
                    code_end = response.rfind("```")
                    
                    if code_start != -1 and code_end != -1:
                        code_block = response[code_start:code_end+3]
                        st.markdown("### Generated Code")
                        st.code(
                            code_block.replace("```"+language.lower(), "").replace("```", "").strip(),
                            language=language.lower(),
                        )
                        
                        notes = response[code_end+3:].strip()
                        if notes:
                            st.markdown("### Implementation Notes")
                            st.markdown(notes)
                        
                        st.download_button(
                            "Download Code",
                            code_block.replace("```"+language.lower(), "").replace("```", "").strip(),
                            file_name=f"generated_code.{language.lower()}",
                            mime="text/plain"
                        )
            elif generate_button:
                st.warning("Please enter project requirements")

if __name__ == "__main__":
    main()