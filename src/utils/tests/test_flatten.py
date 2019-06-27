from src.utils.flatten import flatten


def test_flatten_flat():
    assert flatten({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}


def test_flatten_deep():
    assert flatten({'a': 1, 'b': {'c': 2, 'd': 3}}, sep='__') == \
           {'a': 1, 'b__c': 2, 'b__d': 3}
