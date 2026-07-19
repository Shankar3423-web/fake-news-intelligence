import os
import logging
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Handles loading of datasets from various file formats (CSV, Excel) into Pandas DataFrames.
    Includes robust error handling and logging.
    """
    
    def __init__(self) -> None:
        pass

    def load_csv(self, file_path: str, **kwargs: Any) -> pd.DataFrame:
        """
        Loads a CSV file into a Pandas DataFrame.
        
        Args:
            file_path: The absolute or relative path to the CSV file.
            **kwargs: Extra parameters to pass to pandas.read_csv.
            
        Returns:
            A pandas DataFrame.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or parsing fails.
        """
        logger.info(f"Loading CSV dataset from: {file_path}")
        if not os.path.exists(file_path):
            error_msg = f"CSV file not found at: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Successfully loaded CSV. Shape: {df.shape}")
            return df
        except pd.errors.EmptyDataError as e:
            error_msg = f"CSV file is empty: {file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except pd.errors.ParserError as e:
            error_msg = f"Failed to parse CSV file: {file_path}. Error: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error loading CSV: {file_path}. Error: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def load_excel(self, file_path: str, sheet_name: Optional[Any] = 0, **kwargs: Any) -> pd.DataFrame:
        """
        Loads an Excel (.xlsx, .xls) file into a Pandas DataFrame.
        
        Args:
            file_path: The absolute or relative path to the Excel file.
            sheet_name: The sheet name or index to load (default is first sheet).
            **kwargs: Extra parameters to pass to pandas.read_excel.
            
        Returns:
            A pandas DataFrame.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If loading/parsing the Excel file fails.
        """
        logger.info(f"Loading Excel dataset from: {file_path}")
        if not os.path.exists(file_path):
            error_msg = f"Excel file not found at: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            # Use openpyxl engine as default for .xlsx
            engine = kwargs.pop("engine", "openpyxl")
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine, **kwargs)
            logger.info(f"Successfully loaded Excel. Shape: {df.shape}")
            return df
        except Exception as e:
            error_msg = f"Error loading Excel file: {file_path}. Detail: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def load_dataset(self, file_path: str, **kwargs: Any) -> pd.DataFrame:
        """
        Loads a dataset based on its file extension.
        
        Args:
            file_path: The path to the file.
            **kwargs: Extra arguments to pass to the loaders.
            
        Returns:
            A pandas DataFrame.
            
        Raises:
            ValueError: If the file extension is unsupported or loading fails.
        """
        _, ext = os.path.splitext(file_path.lower())
        if ext == ".csv":
            return self.load_csv(file_path, **kwargs)
        elif ext in [".xlsx", ".xls"]:
            return self.load_excel(file_path, **kwargs)
        else:
            error_msg = f"Unsupported file extension '{ext}' for file: {file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
