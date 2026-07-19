from typing import Dict, Any, List, Optional

class ResponseBuilder:
    """
    ResponseBuilder constructs the standardized response object for the feedback pipeline.
    """
    def build_response(
        self,
        status: str,
        record: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        error: Optional[str] = None,
        version: Optional[str] = None,
        file_paths: Optional[Dict[str, str]] = None,
        hashes: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Compiles the response payload.
        """
        return {
            "feedback_id": record.get("feedback_id") if record else None,
            "status": status,
            "record": record,
            "warnings": warnings or [],
            "error": error,
            "version": version,
            "file_paths": file_paths or {},
            "hashes": hashes or {}
        }
