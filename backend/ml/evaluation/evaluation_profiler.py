import time
import logging
from typing import Dict, Any, Generator
from contextlib import contextmanager
from ml.evaluation.evaluation_utils import get_memory_usage

logger = logging.getLogger("model_evaluation_pipeline")

class EvaluationProfiler:
    """
    Profiler helper for measuring time and memory utilization of evaluation pipeline phases.
    Can be used as a context manager to track CPU/Memory benchmarks.
    """
    @staticmethod
    @contextmanager
    def profile(phase_name: str) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager to benchmark a block of code.
        
        Usage:
            with EvaluationProfiler.profile("XGBoost Evaluation") as stats:
                # evaluation/prediction code here
        """
        logger.info(f"Profiling phase '{phase_name}' started...")
        
        start_time = time.perf_counter()
        start_rss, start_peak = get_memory_usage()
        
        stats: Dict[str, Any] = {}
        
        try:
            yield stats
        finally:
            end_time = time.perf_counter()
            end_rss, end_peak = get_memory_usage()
            
            duration = end_time - start_time
            mem_used_rss = max(0.0, end_rss - start_rss)
            
            stats["phase_name"] = phase_name
            stats["duration_sec"] = round(duration, 4)
            stats["memory_before_rss_mb"] = start_rss
            stats["memory_after_rss_mb"] = end_rss
            stats["memory_used_rss_mb"] = round(mem_used_rss, 2)
            stats["peak_memory_mb"] = max(start_peak, end_peak)
            
            logger.info(
                f"Profiling phase '{phase_name}' completed. "
                f"Duration: {stats['duration_sec']}s, "
                f"Memory RSS Change: {stats['memory_used_rss_mb']} MB, "
                f"Peak RSS: {stats['peak_memory_mb']} MB"
                f" (Throughput is evaluated downstream)"
            )
