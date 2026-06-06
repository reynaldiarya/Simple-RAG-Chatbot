from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from ollama import AsyncClient, Client
from src.core import settings, logger


class OllamaCloudChat(BaseChatModel):
    """LangChain chat model wrapper for Ollama API."""

    model_name: str
    base_url: str
    temperature: float
    top_p: float

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs.get("model", settings.ollama_model)
        self.base_url = kwargs.get("base_url", settings.ollama_api_base)
        self.temperature = kwargs.get("temperature", settings.ollama_temperature)
        self.top_p = kwargs.get("top_p", settings.ollama_top_p)

        # Configure authentication headers
        api_key = settings.ollama_api_key.get_secret_value()
        auth_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        # Initialize clients
        self._client = AsyncClient(
            host=self.base_url,
            headers=auth_headers,
        )
        self._sync_client = Client(
            host=self.base_url,
            headers=auth_headers,
        )

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Converts messages to Ollama format."""
        ollama_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                ollama_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                ollama_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, AIMessage):
                ollama_messages.append({"role": "assistant", "content": msg.content})
        return ollama_messages

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generates async response from model."""
        try:
            ollama_messages = self._convert_messages(messages)
            response = await self._client.chat(
                model=self.model_name,
                messages=ollama_messages,
                stream=False,
                options={"temperature": self.temperature, "top_p": self.top_p},
            )
            content = response["message"]["content"]
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            error_type = type(e).__name__
            logger.error(
                f"Ollama Cloud API Error: [{error_type}] — check network/model config."
            )
            raise RuntimeError(f"LLM service error: {error_type}") from None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generates sync response from model."""
        try:
            ollama_messages = self._convert_messages(messages)
            response = self._sync_client.chat(
                model=self.model_name,
                messages=ollama_messages,
                stream=False,
                options={"temperature": self.temperature, "top_p": self.top_p},
            )
            content = response["message"]["content"]
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            error_type = type(e).__name__
            logger.error(
                f"Ollama Cloud API Error: [{error_type}] — check network/model config."
            )
            raise RuntimeError(f"LLM service error: {error_type}") from None

    @property
    def _llm_type(self) -> str:
        return "ollama-cloud"
