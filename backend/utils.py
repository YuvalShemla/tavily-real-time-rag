import logging, os
from pathlib import Path
from dotenv import load_dotenv

# logging 
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def _configure_logging() -> None:
    if logging.getLogger().handlers: 
        return

    fh = logging.FileHandler(LOG_DIR / "backend.log", mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s → %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), fh],
    )

def get_logger(name: str = "backend") -> logging.Logger:
    _configure_logging()
    return logging.getLogger(name)

# API keys 
def get_keys() -> tuple[str, str]:
    load_dotenv()
    openai_key  = os.getenv("OPENAI_API_KEY")
    tavily_key  = os.getenv("TAVILY_API_KEY")
    if not openai_key or not tavily_key:
        raise RuntimeError("Set OPENAI_API_KEY and TAVILY_API_KEY in .env")
    return openai_key, tavily_key
