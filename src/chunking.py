import re

class RecursiveChunker:
    """
    Split text using a sliding window approach with priority separators.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500, sentence_overlap: int = 1) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size
        self.sentence_overlap = sentence_overlap

    def chunk(self, text: str) -> list[str]:
        text = re.sub(r'\n+', '\n', text)
        if len(text) <= self.chunk_size:
            res = text.strip()
            return [res] if res else []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            
            if end >= text_len:
                chunk_str = text[start:].strip()
                if chunk_str:
                    chunks.append(chunk_str)
                break
                
            # Find the best separator cutting point between start and end
            best_end = -1
            for sep in self.separators:
                if sep == "": continue
                # Search for separator in the window
                pos = text.rfind(sep, start, end)
                # Ensure we move forward and don't get stuck
                if pos != -1 and pos > start:
                    best_end = pos + len(sep)
                    break
            
            if best_end == -1:
                # No clean separator found, force cut
                best_end = end
                
            chunk_str = text[start:best_end].strip()
            if chunk_str:
                chunks.append(chunk_str)
                
            # Calculate next start using SENTENCE overlap
            sub_text = text[start:best_end]
            matches = list(re.finditer(r'[.!?\n]+(?:\s+|$)', sub_text))
            
            if len(matches) > self.sentence_overlap:
                match_idx = len(matches) - self.sentence_overlap - 1
                boundary_match = matches[match_idx]
                start = start + boundary_match.end()
            else:
                if matches:
                    start = start + matches[0].end()
                else:
                    start = best_end
            
        # Deduplication and formatting
        res = []
        last_str = ""
        for c in chunks:
            c = c.replace("\n", " ")
            if c != last_str:
                res.append(c)
                last_str = c
        return res

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    similarity = 0.0
    dot_ab = _dot(vec_a, vec_b)
    norm_a = math.sqrt(sum(x*x for x in vec_a))
    norm_b = math.sqrt(sum(x*x for x in vec_b))
    if norm_a > 0 and norm_b > 0:
        similarity = dot_ab / (norm_a * norm_b)
    return similarity