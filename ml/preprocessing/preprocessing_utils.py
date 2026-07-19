import sys
import logging
import nltk
import spacy

logger = logging.getLogger("preprocessing_pipeline")

def ensure_nltk_resources() -> None:
    """Ensures necessary NLTK corpora are downloaded."""
    resources = ["stopwords", "punkt"]
    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}" if resource == "stopwords" else f"tokenizers/{resource}")
            logger.debug(f"NLTK resource '{resource}' is already available.")
        except LookupError:
            logger.info(f"Downloading NLTK resource '{resource}'...")
            try:
                nltk.download(resource, quiet=True)
                logger.info(f"Successfully downloaded NLTK resource '{resource}'.")
            except Exception as e:
                logger.error(f"Failed to download NLTK resource '{resource}': {e}")
                # Re-try without quiet mode if it fails
                try:
                    nltk.download(resource)
                except Exception as e_inner:
                    logger.error(f"Critical failure downloading NLTK resource '{resource}': {e_inner}")

def ensure_spacy_model(model_name: str = "en_core_web_sm") -> bool:
    """
    Ensures that the requested spaCy model is installed and loadable.
    Downloads it programmatically if missing.
    """
    try:
        spacy.load(model_name)
        logger.debug(f"spaCy model '{model_name}' is already available.")
        return True
    except OSError:
        logger.info(f"spaCy model '{model_name}' not found. Attempting download...")
        try:
            # spaCy provides a cli download helper
            from spacy.cli import download as spacy_download
            spacy_download(model_name)
            logger.info(f"Successfully downloaded spaCy model '{model_name}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to download spaCy model '{model_name}' via spacy.cli: {e}")
            
            # Subprocess fallback
            import subprocess
            try:
                logger.info(f"Attempting to download spaCy model '{model_name}' via pip subprocess...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"https://github.com/explosion/spacy-models/releases/download/{model_name}/{model_name}-py3-none-any.whl"])
                logger.info(f"Successfully installed spaCy model '{model_name}' via wheel download.")
                return True
            except Exception as e_sub:
                logger.critical(f"Failed all attempts to download spaCy model '{model_name}': {e_sub}")
                return False
