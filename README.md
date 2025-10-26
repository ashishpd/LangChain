# LangChain
LangChain Example Python Code

## Python Scripts Overview

| Script Name                  | Description                  | Video Link |
|------------------------------|------------------------------|------------|
| 01_hello_python.py           | Hello world print            | https://youtu.be/9lkY-pp62II |
| 02_openai_chat_completion.py | OpenAI chat via openai lib   | TBD        |
| 03_openai_langchain_chat.py  | OpenAI chat via LangChain    | TBD        |
| 04_streamlit_langchain_chat.py | Streamlit UI for OpenAI chat | TBD        |
| 05_streamlit_gdp_llm.py        | Streamlit: LLM US GDP chart  | TBD        |

## Get the code

You can either create a local folder named `LangChain` and work there, or clone this repository directly.

## Quick start (VS Code, venv, OpenAI key)

1. Open the project folder in VS Code
2. Open Terminal
3. Create a one-time virtual environment using Python 3.11 and activate it:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

In the future, re-activate the virtual environment with:

```bash
source .venv/bin/activate
```

After activating the virtual environment, install the project dependencies:

```bash
pip install -r requirements.txt
```

If you don't have Python installed yet, install Homebrew first and then install Python (see the "Install brew on Mac" section below).

4. Make sure you have an OpenAI API key. Then add it to your environment (example):

```bash
export OPENAI_API_KEY="sk-proj-w5W..."
```

Important: issue the above command in the same terminal window where you issues the source command above.

5. Create a simple hello world script `01_hello_python.py`:

```python
print("Hello, LangChain!")
```


Run it with:

```bash
python 01_hello_python.py
```

---

## Streamlit US GDP Chart Example

To run the Streamlit app that fetches and plots US GDP (last 5 years) using an LLM:

```bash
streamlit run 05_streamlit_gdp_llm.py
```

## Install brew on Mac

If you're on macOS, install Homebrew first (Homebrew is the recommended package manager for macOS).

Install Homebrew by running the official installer in a terminal:

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

It will ask you to type sudo password. Make sure to close the Terminal and start a new one.

If you're on Linux or Windows, use your platform's package manager (apt, yum, choco, winget, etc.), or install Homebrew on Linux if you prefer.

## Install Python

If you're on macOS you can install Python 3.11 via Homebrew:

```bash
brew install python@3.11
```

## Install Ollama
Download Ollama
Open Terminal and run
ollama list -- to see models already downloaded
ollama pull gemma3:270m -- gets the gemma3:270m model
ollama run gemma3:270m -- runs the gemma3:270m model
ollama ps -- shows the running models
ollama stop gemma3:270m -- stops the gemma3:270m model








