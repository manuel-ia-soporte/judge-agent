# infrastructure/adapters/finance_adapter.py
from enum import Enum


class FilingType(str, Enum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"
