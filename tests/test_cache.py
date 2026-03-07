import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.semantic_cache import SemanticCache

def test_cache():
    c = SemanticCache(threshold=0.9)
    v1 = np.array([1, 0, 0])
    v2 = np.array([0.95, 0.05, 0]) # Semantic similar
    
    # Miss
    res, _ = c.lookup(v1, 1)
    assert res is None and c.misses == 1
    
    # Hit
    c.add("hi", v1, "hello", 1)
    res, score = c.lookup(v2, 1)
    assert res["result"] == "hello" and c.hits == 1
    print("Cache Unit Tests Passed!")

if __name__ == "__main__":
    test_cache()