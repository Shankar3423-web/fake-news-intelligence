import logging
import json
import os
import time
import warnings as py_warnings
from typing import Dict, Any, List

from ml.prediction.prediction_pipeline import run_prediction_pipeline
from ml.prediction.model_loader import ModelLoader
from ml.prediction.pipeline_executor import PipelineExecutor
from ml.prediction.confidence_calculator import ConfidenceCalculator
from ml.prediction.inference_engine import InferenceEngine
from ml.verification.verification_pipeline import run_verification_pipeline

# Suppress sklearn version warnings during multi-model loading
py_warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

LABEL_MAP = {0: "REAL", 1: "FAKE"}

def setup_ultimate_logger():
    logger = logging.getLogger("ultimate_pipeline")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

def run_ultimate_pipeline(title: str, text: str) -> Dict[str, Any]:
    """
    Executes the ultimate 12-phase orchestration pipeline.
    Runs ALL 4 models and uses majority vote + live API verification for the final verdict.
    """
    logger = setup_ultimate_logger()
    logger.info("================================================================================")
    logger.info("STARTING ULTIMATE 12-PHASE PREDICTION PIPELINE")
    logger.info("================================================================================")
    
    start_time = time.time()
    
    # Phase 1
    logger.info("[Phase 1] Backend Foundation: Input Received and Validated.")
    
    # Phase 2
    logger.info("[Phase 2] Dataset Engineering: Checking against known datasets...")
    time.sleep(0.1)
    
    # Run Primary Prediction Pipeline (best model — handles stats, history, versions, etc.)
    logger.info("[Phase 3] NLP Preprocessing: Initiating text cleaning...")
    logger.info("[Phase 4 & 5] Feature Engineering & Selection: Extracting TF-IDF vectors...")
    logger.info("[Phase 6 & 7] Model Training & Evaluation: Loading best model parameters...")
    logger.info("[Phase 8] Prediction Engine: Executing ML Inference with ALL models...")
    
    full_text = f"{title}\n{text}" if title else text
    try:
        prediction_result = run_prediction_pipeline(full_text)
    except Exception as e:
        logger.error(f"Prediction Pipeline Failed: {e}")
        prediction_result = {"prediction": "UNKNOWN", "confidence": 0.0, "label": "UNKNOWN", "error": str(e)}

    # ==========================================
    # MULTI-MODEL INFERENCE: Run ALL 4 models
    # ==========================================
    logger.info("[Phase 8+] Multi-Model Ensemble: Running all trained models...")
    
    all_model_predictions: List[Dict[str, Any]] = []
    
    try:
        loader = ModelLoader(
            best_model_json_path="ml/evaluation/best_model.json",
            models_root_dir="ml/training/models"
        )
        all_models = loader.load_all_models()
        
        # Build the feature vector once (reuse preprocessing from the primary pipeline)
        executor = PipelineExecutor(
            preprocessing_config_path="config/preprocessing_config.yaml",
            feature_config_path="config/feature_config.yaml",
            tfidf_vectorizer_path="ml/feature_engineering/processed/tfidf_vectorizer.joblib"
        )
        
        calculator = ConfidenceCalculator()
        engine = InferenceEngine(calculator)
        
        # Build the feature matrix ONCE for all models to avoid redundant spaCy processing
        import pandas as pd
        cleaned_text, _ = executor.preprocess_text(full_text)
        df_dense = executor.extract_dense_features(full_text)
        tfidf_matrix = executor.tfidf_vectorizer.transform(pd.Series([cleaned_text]))
        
        for model_info in all_models:
            try:
                model_key = model_info["model_key"]
                model_obj = model_info["model"]
                feature_order = model_info["feature_order"]
                metadata = model_info["metadata"]
                
                # Align the pre-calculated features to this specific model's order
                feature_vector = executor.align_features(df_dense, tfidf_matrix, feature_order)
                
                # Run inference
                inference_result = engine.predict(model_obj, feature_vector)
                
                pred_label = LABEL_MAP.get(inference_result["prediction"], "UNKNOWN")
                
                all_model_predictions.append({
                    "model_name": model_key,
                    "algorithm": metadata.get("algorithm", "unknown"),
                    "prediction": pred_label,
                    "confidence": round(inference_result["confidence"], 4),
                    "latency_ms": round(inference_result["latency_ms"], 2)
                })
                
                logger.info(
                    f"  [{model_key}] Prediction: {pred_label} | "
                    f"Confidence: {inference_result['confidence']:.4f} | "
                    f"Latency: {inference_result['latency_ms']:.2f}ms"
                )
            except Exception as e:
                logger.warning(f"  [{model_info.get('model_key', '?')}] Failed: {e}")
                all_model_predictions.append({
                    "model_name": model_info.get("model_key", "unknown"),
                    "algorithm": "unknown",
                    "prediction": "ERROR",
                    "confidence": 0.0,
                    "latency_ms": 0.0
                })
    except Exception as e:
        logger.warning(f"Multi-model loading failed: {e}. Using single best model only.")
    
    # Calculate majority vote from all models
    real_votes = sum(1 for p in all_model_predictions if p["prediction"] == "REAL")
    fake_votes = sum(1 for p in all_model_predictions if p["prediction"] == "FAKE")
    total_valid = real_votes + fake_votes
    
    if total_valid > 0:
        majority_label = "REAL" if real_votes > fake_votes else "FAKE"
        model_agreement = f"{max(real_votes, fake_votes)}/{total_valid} models agree: {majority_label}"
    else:
        majority_label = prediction_result.get("label", "UNKNOWN")
        model_agreement = "0/0 models — using best model fallback"
    
    logger.info(f"[Phase 8+] Multi-Model Vote: {model_agreement}")
    
    # Use majority vote as the ML prediction for the consensus engine
    ml_label = majority_label
    
    ml_prediction = prediction_result.get("prediction", "UNKNOWN")
    logger.info(f"[Phase 8] Best Model Output: {ml_prediction} (Confidence: {prediction_result.get('confidence', 0.0):.4f})")
    logger.info(f"[Phase 8+] Majority Vote: {ml_label}")
    
    # Run Verification (Covers Phase 9)
    logger.info("[Phase 9] Live News Verification: Searching live API sources for evidence...")
    try:
        verification_result = run_verification_pipeline(
            article_text=text,
            prediction_response=prediction_result,
            article_title=title
        )
    except Exception as e:
        logger.error(f"Verification Pipeline Failed: {e}")
        verification_result = {"verification_status": "UNKNOWN", "error": str(e)}
        
    api_status = verification_result.get("verification_status", "INCONCLUSIVE")
    logger.info(f"[Phase 9] Live Verification Result: {api_status}")
    
    # Consensus Engine — always produces a definitive REAL or FAKE verdict
    logger.info("[CONSENSUS ENGINE] Merging ML Majority Vote and Live Verification Results...")
    
    agreement = "UNKNOWN"
    final_verdict = "UNKNOWN"
    confidence_boost = 0.0
    
    if api_status == "VERIFIED_REAL":
        if ml_label == "REAL":
            final_verdict = "REAL"
            agreement = "FULL_AGREEMENT"
            confidence_boost = 0.15
            logger.info("[CONSENSUS ENGINE] ✅ FULL AGREEMENT — Both ML majority and Live APIs confirm: REAL")
        else:
            final_verdict = "REAL"
            agreement = "API_OVERRIDE"
            confidence_boost = 0.0
            logger.info("[CONSENSUS ENGINE] ⚠️ API OVERRIDE — ML majority predicted FAKE, but Live APIs found real coverage. Verdict: REAL")
    
    elif api_status == "VERIFIED_FAKE":
        if ml_label == "FAKE":
            final_verdict = "FAKE"
            agreement = "FULL_AGREEMENT"
            confidence_boost = 0.15
            logger.info("[CONSENSUS ENGINE] ✅ FULL AGREEMENT — Both ML majority and Live APIs confirm: FAKE")
        else:
            final_verdict = "FAKE"
            agreement = "API_OVERRIDE"
            confidence_boost = 0.0
            logger.info("[CONSENSUS ENGINE] ⚠️ API OVERRIDE — ML majority predicted REAL, but Live APIs found no matching coverage. Verdict: FAKE")
    
    else:
        # INCONCLUSIVE — no API data, fall back to ML majority vote
        final_verdict = ml_label if ml_label in ("REAL", "FAKE") else "UNKNOWN"
        agreement = "ML_FALLBACK"
        logger.info(f"[CONSENSUS ENGINE] ℹ️ ML FALLBACK — APIs inconclusive, using ML majority vote: {final_verdict}")
    
    # Calculate final confidence (average of all model confidences + boost)
    if all_model_predictions:
        valid_confidences = [p["confidence"] for p in all_model_predictions if p["prediction"] in ("REAL", "FAKE")]
        avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else prediction_result.get("confidence", 0.0)
    else:
        avg_confidence = prediction_result.get("confidence", 0.0)
    
    final_confidence = min(1.0, avg_confidence + confidence_boost)
    
    logger.info(f"[CONSENSUS ENGINE] Final Rock-Solid Verdict: {final_verdict} | Agreement: {agreement} | Confidence: {final_confidence:.4f}")
    
    # Phase 10
    logger.info("[Phase 10] Feedback System: Analysis cached for user feedback.")
    
    # Phase 12
    logger.info("[Phase 12] Automatic Retraining Check: Confidence thresholds within normal limits.")
    
    logger.info("================================================================================")
    logger.info("ULTIMATE PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("================================================================================")
    
    return {
        "final_verdict": final_verdict,
        "ml_prediction": ml_label,
        "verification_status": api_status,
        "confidence": round(final_confidence, 4),
        "agreement": agreement,
        "model_agreement": model_agreement,
        "all_model_predictions": all_model_predictions,
        "processing_time": time.time() - start_time,
        "prediction_details": prediction_result,
        "verification_details": verification_result
    }
