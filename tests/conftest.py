import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

from acme_rag import AcmeRag  # noqa: E402

from rag_quality_monitor import Chunk, Store  # noqa: E402
from rag_quality_monitor.judge import FakeEmbedder, FakeJudge  # noqa: E402


@pytest.fixture
def store():
    return Store()


@pytest.fixture
def judge():
    return FakeJudge()


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def rag():
    return AcmeRag()


def chunk(cid: str, text: str = "text", score: float = 0.5) -> Chunk:
    return Chunk(id=cid, doc_id="d", text=text, score=score)
