# src/ai_document_assistant_with_rag/guardrails.py
import io
import threading
import time
from collections import deque

from pypdf import PdfReader

MAX_FILE_MB = 5
MAX_PAGES = 50
MAX_FILES = 3
MAX_CHUNKS = 400
MAX_QUESTION_CHARS = 500


def validate_pdf(data: bytes, name: str, n_files: int) -> tuple[bool, str]:
    if n_files >= MAX_FILES:
        return False, f"Limit of {MAX_FILES} files per session."
    if not name.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
        return False, "That doesn't look like a valid PDF."
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return False, f"File is larger than {MAX_FILE_MB} MB."
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            return False, "Encrypted PDFs aren't supported."
        if len(reader.pages) > MAX_PAGES:
            return False, f"PDF has more than {MAX_PAGES} pages."
    except Exception:
        return False, "Couldn't read that PDF."
    return True, "ok"


def check_question(q: str) -> tuple[bool, str]:
    q = q.strip()
    if not q:
        return False, "Please type a question."
    if len(q) > MAX_QUESTION_CHARS:
        return False, f"Questions are limited to {MAX_QUESTION_CHARS} characters."
    return True, "ok"


class RateLimiter:
    """Sliding-window limiter: at most max_calls per per_seconds."""

    def __init__(self, max_calls: int, per_seconds: int):
        self.max_calls, self.per_seconds = max_calls, per_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        now = time.time()
        with self._lock:
            while self._calls and now - self._calls[0] > self.per_seconds:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                return False
            self._calls.append(now)
            return True


GLOBAL_LIMITER = RateLimiter(max_calls=300, per_seconds=86400)  # whole-app daily cap