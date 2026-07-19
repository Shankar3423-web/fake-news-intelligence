import os
import time
from typing import Tuple

def get_memory_usage() -> Tuple[float, float]:
    """
    Returns (current_memory_mb, peak_memory_mb) of the current process.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss = round(mem_info.rss / (1024 * 1024), 2)
        
        # Track peak memory using peak_wset on Windows
        if hasattr(mem_info, 'peak_wset'):
            peak = round(mem_info.peak_wset / (1024 * 1024), 2)
        else:
            peak = rss
        return rss, peak
    except Exception:
        return 0.0, 0.0
