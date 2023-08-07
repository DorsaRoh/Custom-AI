# 🧠 Realize AI




### AI trained on **custom** user data.

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

### 2. Install dependencies

```shell
pip install langchain openai streamlit pdfreader wikipedia-api streamlit
```
or
```shell
pip install -r requirements.txt
```

### 3. Usage
 
```shell
streamlit run app.py
```
Navigate to localhost:8501 in your web browser to access the app.

🚨 **Enter your OpenAI API key in the *key.py* folder.**


 ## Sample Uses
See the [PatientGPT.AI](https://github.com/DorsaRoh/Custom-AI/tree/main/Sample%20Use%20-%20PatientGPT.AI) sample case: A medical sample use case of CustomAI, where doctors can upload data of their patients and have AI diagnose them for desired diseases and/or conditions.
