# infrastructure/adapters/finance_adapter.py
from enum import Enum


class FilingType(str, Enum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    # FORM_10K = "10-K"             # domain/models/finance.py
    # FORM_10Q = "10-Q"
    # FORM_8K = "8-K"
    # FORM_S1 = "S-1"
    # FORM_4 = "4"

