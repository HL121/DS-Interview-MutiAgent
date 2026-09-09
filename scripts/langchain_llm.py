import json
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def get_chat_model(temperature: float = 0.0, model: str = "gpt-4o-mini"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


def parse_json_text(text: str) -> Any:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```json", "", cleaned)
    cleaned = re.sub(r"^```", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def invoke_text(
    prompt: str,
    system_message: Optional[str] = None,
    temperature: float = 0.0,
    client=None,
) -> str:
    llm = client if client is not None and hasattr(client, "invoke") else get_chat_model(temperature=temperature)
    messages = []
    if system_message:
        messages.append(("system", system_message))
    messages.append(("human", "{prompt}"))
    chain = ChatPromptTemplate.from_messages(messages) | llm
    response = chain.invoke({"prompt": prompt})
    return response.content


def invoke_json(
    prompt: str,
    system_message: Optional[str] = None,
    temperature: float = 0.0,
    client=None,
) -> Any:
    return parse_json_text(
        invoke_text(
            prompt=prompt,
            system_message=system_message,
            temperature=temperature,
            client=client,
        )
    )
