from textwrap import shorten as _shorten
def clip(text: str, width: int = 70) -> str:
    return _shorten(text or "—", width)