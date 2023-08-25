# 🧠 Custom-AI

### AI trained on **custom** user data.
Powered by OpenAI + Langchain + Streamlit


![App Image](https://github.com/DorsaRoh/CustomAI/blob/main/content/app.png)

Video Demo: https://youtu.be/qhgamzZxZq4

Leverages Python, Langchain and OpenAI's ChatGPT to empower you to chat with your data, and build AI trained on your PDFs, fast and accessibly. The future of AI isn't just intelligence - it's personalization.

## The AI can:
- 📁 Use your own data
- 🔠 Memorize the conversation
- 💬 Save the conversation

Data is saved in the **'data'** folder! 
<br></br>
Note: Accepted data types are .pdf and .txt files

## Installation

### 1. Clone the repository
```shell
git clone https://github.com/DorsaRoh/Custom-AI.git
```

### 2. Enter directory
```shell
cd Custom-AI
```

### 3. Install dependencies
**Ensure you have chromadb installed.**

```shell
pip install langchain openai streamlit pdfreader wikipedia-api streamlit unstructured unstructured[pdf]
```
or
```shell
pip install -r requirements.txt
```

### 4. Usage
 
```shell
streamlit run app.py
```
Navigate to local host 1 in your web browser to access the app.

🚨 **You must enter your OpenAI API key. An error will show until you enter your OpenAI API key in the application's sidebar/input field.**

Change the prompt templates according to your needs and requirements.

 ## Sample Uses
- [PatientGPT.AI](https://github.com/DorsaRoh/Custom-AI/tree/main/PatientGPT.AI) A medical sample use case of CustomAI, where doctors can upload data of their patients and have AI diagnose them for desired diseases and/or conditions.
- [RealizeAI](https://github.com/DorsaRoh/Custom-AI/tree/main/RealizeAI)
Think your unique knowledge has no real-world value? With Realize-AI, turn even the most obscure ideas into actionable tasks.
