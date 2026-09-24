import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
# Make `botboi` and the test helpers importable without installing anything.
sys.path[:0] = [str(TESTS.parent), str(TESTS)]
