import logging
import sys
from .config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger("rag_bot")
