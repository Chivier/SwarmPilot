"""Run the adapter contract suite against every installed adapter."""

import pytest

from swarmpilot.engines import adapter_names, get_adapter
from swarmpilot.engines.contract import check_adapter


def test_builtin_adapters_are_registered():
    assert {"vllm", "sglang"} <= set(adapter_names())


@pytest.mark.parametrize("name", adapter_names())
def test_adapter_contract(name: str):
    adapter = get_adapter(name)
    assert adapter.name == name
    assert check_adapter(adapter) == []
