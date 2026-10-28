
from model import train

def test_training_pipeline():
    accuracy = train()
    assert 0 <= accuracy <= 1
