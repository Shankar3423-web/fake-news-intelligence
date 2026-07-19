import unicodedata
import re
import html
from typing import Optional

class FeedbackSanitizer:
    """
    FeedbackSanitizer provides utilities to clean, escape, and normalize comment strings.
    """
    def sanitize_comment(self, comment: Optional[str]) -> str:
        """
        Sanitizes leading/trailing/multiple spaces, strips control characters,
        normalizes unicode, and escapes unsafe content.
        """
        if comment is None:
            return ""

        # 1. Normalize Unicode (NFKC)
        cleaned = unicodedata.normalize("NFKC", comment)

        # 2. Strip Control Characters (Unicode category starts with 'C')
        cleaned = "".join(ch for ch in cleaned if not unicodedata.category(ch).startswith("C"))

        # 3. Normalize multiple spaces to a single space
        cleaned = re.sub(r"\s+", " ", cleaned)

        # 4. Strip leading/trailing spaces
        cleaned = cleaned.strip()

        # 5. Escape HTML/XML to prevent injection or malformed records
        cleaned = html.escape(cleaned)

        return cleaned
