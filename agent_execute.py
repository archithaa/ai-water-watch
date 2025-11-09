import pandas as pd
import plotly.express as px
from google import genai

client = genai.Client(vertexai=True, project="ai-water-watch-477615", location="us-central1")

def llm_analyze(question: str, df: pd.DataFrame):
    prompt = f"""
You are a data analysis assistant. The user asked a question about a dataset.

Dataset columns:
{', '.join(df.columns)}

User question:
"{question}"

Return a JSON response with:
- "answer": a short natural language answer (≤60 words)
- "code": python code that uses only the dataframe named df to produce a plotly figure (variable name must be fig)

Return ONLY valid JSON. Do not include markdown.
    """

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": prompt}]}]
    )

    import json
    data = json.loads(resp.text)

    # Execute generated code safely (globals={}, locals={'df': df})
    local_env = {'df': df, 'px': px}
    exec(data["code"], {}, local_env)
    fig = local_env.get("fig")

    return data["answer"], fig
