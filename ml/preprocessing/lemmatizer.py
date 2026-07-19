import logging
from typing import List
import spacy

logger = logging.getLogger("preprocessing_pipeline")

class Lemmatizer:
    """
    Lemmatizes tokens using spaCy's English model.
    Optimized by disabling parser and ner components.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        try:
            # Disable parser and ner for optimization (major speedup)
            self.nlp = spacy.load(self.model_name, disable=["parser", "ner"])
        except Exception as e:
            logger.warning(f"Could not load spaCy model '{self.model_name}' directly: {e}. Attempting import check...")
            # Fallback to loading it by import if it was installed as a module
            try:
                import en_core_web_sm
                self.nlp = en_core_web_sm.load(disable=["parser", "ner"])
            except Exception as e_inner:
                logger.error(f"Critical error loading spaCy model: {e_inner}")
                raise e_inner

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatizes a list of tokens.
        Joins tokens, processes with spaCy, and extracts lemmas.
        """
        if not tokens:
            return []
        
        # Create doc from joined tokens to preserve some sequential context
        text = " ".join(tokens)
        doc = self.nlp(text)
        
        # Extract lemmas, ensuring we don't have trailing whitespace or empty lemmas
        lemmas = [t.lemma_ for t in doc if t.lemma_.strip()]
        return lemmas

    def lemmatize_batch(self, batch_tokens: List[List[str]]) -> List[List[str]]:
        """
        Lemmatizes a batch of token lists using spaCy's nlp.pipe for efficiency.
        """
        if not batch_tokens:
            return []
        
        texts = [" ".join(tokens) for tokens in batch_tokens]
        # Use nlp.pipe with batch size for speed
        docs = self.nlp.pipe(texts, batch_size=len(batch_tokens), n_process=1)
        
        batch_lemmas = []
        for doc in docs:
            lemmas = [t.lemma_ for t in doc if t.lemma_.strip()]
            batch_lemmas.append(lemmas)
            
        return batch_lemmas
