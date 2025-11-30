# connections_solver/__init__.py

from .puzzle import Puzzle
from .similarity import SimilarityBackend, get_default_backend

__all__ = ["Puzzle", "SimilarityBackend", "get_default_backend"]
