import pytest

from support import crm


@pytest.fixture(autouse=True)
def fresh_crm():
    crm.reset()
