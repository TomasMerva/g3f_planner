import pytest
import warnings
"""
This script contains tests of the pumafabrics examples, but NOT of the evaluation scripts. 
"""

def blueprint_test(test_main):
    """
    Blueprint for environment tests.
    An environment main always has the one argument:
        - render: bool

    The function verifies if the main returns a list of observations.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        history = test_main(render=False, timesteps=100)
    assert isinstance(history, dict)

def test_deadlock_resolution():
    from example_prf_dinovas import main
    blueprint_test(main)

def test_g3f():
    from example_g3f_dinovas import main
    blueprint_test(main)

def test_vanilla_fabrics():
    from example_gf_dinovas import main
    blueprint_test(main)