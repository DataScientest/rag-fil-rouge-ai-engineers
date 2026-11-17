"""Test setup: the gateway variables must exist at import time, even for tests that never call the network."""
import os

from dotenv import load_dotenv

load_dotenv()  # real values from .env when present

os.environ.setdefault("LLM_BASE_URL", "http://localhost:0/unused")
os.environ.setdefault("LLM_API_KEY", "unused-in-fast-tests")
