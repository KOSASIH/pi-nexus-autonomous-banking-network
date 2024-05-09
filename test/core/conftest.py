import pytest
import core.module_a
import core.module_b

@pytest.fixture
def module_a():
    return core.module_a.ModuleA()

@pytest.fixture
def module_b():
    return core.module_b.ModuleB()
