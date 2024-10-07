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
        history = test_main(render=False, timesteps=200)
    assert isinstance(history, dict)

def test_kuka_fabrics():
    from examples.example_deadlock_resolution import main
    blueprint_test(main)

def test_kuka_ModulationIK():
    from examples.example_rgf_dinovas import main
    blueprint_test(main)

def test_kuka_TamedPUMA():
    from examples.example_vanilla_fabrics import main
    blueprint_test(main)
