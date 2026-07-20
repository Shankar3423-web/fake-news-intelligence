import logging
import re
from typing import List, Dict, Any, Set
from collections import Counter

from ml.preprocessing.preprocessing_utils import get_shared_spacy_model

logger = logging.getLogger("verification_pipeline")

# Common English stopwords + news-specific words to ignore in query generation
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
    "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
    "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "said", "reporting",
    "reports", "government", "official", "officials", "statement", "announced", "says", "told", "according"
}

class KeywordExtractor:
    """
    KeywordExtractor analyzes text using spaCy to extract entities and nouns.
    If spaCy is not installed, it falls back to regex-based keyword extraction.
    """
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        try:
            self.nlp = get_shared_spacy_model(self.model_name)
        except Exception as e:
            logger.warning(f"Could not load shared spaCy model '{self.model_name}': {e}. Falling back to regex extraction.")
            self.nlp = None

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Parses text and extracts named entities and key nouns.
        Returns a dictionary with entities, nouns, and an optimized query.
        """
        if not text or not text.strip():
            return {
                "entities": [],
                "nouns": [],
                "query": ""
            }

        if self.nlp:
            return self._extract_spacy(text)
        else:
            return self._extract_fallback(text)

    def _extract_spacy(self, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)
        
        entities: List[Dict[str, str]] = []
        entity_texts: Set[str] = set()
        
        target_labels = {"ORG", "GPE", "LOC", "DATE", "EVENT", "PERSON", "NORP"}
        for ent in doc.ents:
            cleaned_ent = ent.text.strip().replace("\n", " ")
            if cleaned_ent and ent.label_ in target_labels:
                if cleaned_ent.lower() not in {e.lower() for e in entity_texts}:
                    entities.append({
                        "text": cleaned_ent,
                        "label": ent.label_
                    })
                    entity_texts.add(cleaned_ent)
                    
        nouns_list: List[str] = []
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
                word = token.text.strip()
                if len(word) > 2:
                    is_in_entity = False
                    for ent_text in entity_texts:
                        if word.lower() in ent_text.lower():
                            is_in_entity = True
                            break
                    if not is_in_entity:
                        nouns_list.append(word)

        noun_counts = Counter(nouns_list)
        top_nouns = [noun for noun, count in noun_counts.most_common(10)]

        query_components: List[str] = []
        priority_labels = {"ORG", "GPE", "PERSON", "EVENT"}
        priority_ents = [ent["text"] for ent in entities if ent["label"] in priority_labels]
        
        for ent in priority_ents[:3]:
            if ent not in query_components:
                query_components.append(ent)
                
        other_ents = [ent["text"] for ent in entities if ent["label"] not in priority_labels]
        for ent in other_ents[:2]:
            if len(query_components) >= 4:
                break
            if ent not in query_components:
                query_components.append(ent)
                
        for noun in top_nouns:
            if len(query_components) >= 6:
                break
            if noun not in query_components:
                query_components.append(noun)
                
        query_string = " ".join(query_components).strip()
        if not query_string:
            query_string = " ".join([token.text for token in doc if not token.is_stop][:5])

        logger.info(f"Extracted keywords (spaCy). Query generated: '{query_string}'")
        return {
            "entities": entities,
            "nouns": top_nouns,
            "query": query_string
        }

    def _extract_fallback(self, text: str) -> Dict[str, Any]:
        # 1. Regex Entity Guessing (look for capitalized words, except at start of sentence)
        # Tokenize by word
        words = re.findall(r'\b\w+\b', text)
        
        # Simple stopword filtering and case filtering
        entities_guessed = []
        seen_entities = set()
        
        # Guess entities: sequences of capitalized words like "Jerome Powell" or "Federal Reserve"
        # We find candidates matching capitalized sequences
        candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for cand in candidates:
            cand_clean = cand.strip()
            if cand_clean.lower() not in STOPWORDS and len(cand_clean) > 3:
                if cand_clean.lower() not in seen_entities:
                    entities_guessed.append({
                        "text": cand_clean,
                        "label": "GPE" if "Washington" in cand_clean or "D.C." in cand_clean else "ORG"
                    })
                    seen_entities.add(cand_clean.lower())

        # 2. Extract nouns/important words
        nouns_guessed = []
        for word in words:
            word_lower = word.lower()
            if word_lower not in STOPWORDS and len(word) > 2:
                # Make sure it's not already inside a guessed entity
                is_in_entity = False
                for ent in seen_entities:
                    if word_lower in ent:
                        is_in_entity = True
                        break
                if not is_in_entity:
                    nouns_guessed.append(word)

        noun_counts = Counter(nouns_guessed)
        top_nouns = [noun for noun, count in noun_counts.most_common(10)]

        # 3. Construct Query
        query_components = []
        # Add entities
        for ent in entities_guessed[:3]:
            query_components.append(ent["text"])
        # Add top nouns
        for noun in top_nouns:
            if len(query_components) >= 6:
                break
            if noun not in query_components:
                query_components.append(noun)

        query_string = " ".join(query_components).strip()
        if not query_string:
            # Fallback to first few words
            filtered_words = [w for w in words if w.lower() not in STOPWORDS]
            query_string = " ".join(filtered_words[:5])

        logger.info(f"Extracted keywords (Regex Fallback). Query generated: '{query_string}'")
        return {
            "entities": entities_guessed,
            "nouns": top_nouns,
            "query": query_string
        }
