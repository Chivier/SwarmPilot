"""PyLet submit survives a dropped keep-alive connection."""

from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("pylet")

from pylet.errors import NotFoundError

from swarmpilot.planner.pylet import client

KW = {"name": "m-vllm-abc", "gpu": 1, "cpu": 1}


@pytest.fixture
def fake_pylet(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(client, "pylet", fake)
    return fake


def test_resubmits_when_head_never_saw_it(fake_pylet):
    inst = MagicMock()
    fake_pylet.submit.side_effect = [httpx.ReadError("reset"), inst]
    fake_pylet.get.side_effect = NotFoundError("no")
    assert client._submit("cmd", KW) is inst
    assert fake_pylet.submit.call_count == 2


def test_returns_existing_when_head_saw_it(fake_pylet):
    inst = MagicMock()
    fake_pylet.submit.side_effect = httpx.ReadError("reset")
    fake_pylet.get.return_value = inst
    assert client._submit("cmd", KW) is inst
    fake_pylet.get.assert_called_once_with("m-vllm-abc")
    assert fake_pylet.submit.call_count == 1


def test_other_errors_are_not_retried(fake_pylet):
    fake_pylet.submit.side_effect = ValueError("bad")
    with pytest.raises(ValueError):
        client._submit("cmd", KW)
    assert fake_pylet.submit.call_count == 1
