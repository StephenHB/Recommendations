from typing import List, Union
from pathlib import Path


class Base:
    """
    Configs for the BLP model
    """
    
    def __init__(self):
        self.input_table_path: Union[str, Path] = ""
        self.reg_param: float = 0.0
        self.target_name: str = ""
        self.exog_ind_names: List[str] = []
        self.exog_dep_names: List[str] = []
        self.instrument_variable_names: List[str] = []