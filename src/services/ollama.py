from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from ollama import AsyncClient, Client
from src.core import settings, logger


class OllamaCloudChat(BaseChatModel):
    """LangChain-compatible chat model using Ollama Cloud API with authentication."""
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
        
        # Create async client with auth header
        self._client = AsyncClient(
            host=self.base_url,
            headers={"Authorization": f"Bearer {settings.ollama_api_key}"}
        )
        # Sync client for non-async calls
        self._sync_client = Client(
            host=self.base_url,
            headers={"Authorization": f"Bearer {settings.ollama_api_key}"}
        )
    
    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to Ollama format."""
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
        """Async generation."""
        try:
            ollama_messages = self._convert_messages(messages)
            response = await self._client.chat(
                model=self.model_name,
                messages=ollama_messages,
                stream=False,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p
                }
            )
            content = response['message']['content']
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            logger.error(f"Ollama Cloud API Error: {str(e)}")
            raise e
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Sync generation."""
        try:
            ollama_messages = self._convert_messages(messages)
            response = self._sync_client.chat(
                model=self.model_name,
                messages=ollama_messages,
                stream=False,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p
                }
            )
            content = response['message']['content']
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            logger.error(f"Ollama Cloud API Error: {str(e)}")
            raise e
    
    @property
    def _llm_type(self) -> str:
        return "ollama-cloud"
