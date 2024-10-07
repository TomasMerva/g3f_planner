import pytest
import warnings
"""
This script contains tests of the pumafabrics examples, but NOT of the evaluation scripts. 
"""


def blueprint_test_evaluation(test_main):
    """
    The function verifies if the main returns a list of observations from the evaluation scripts
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        history = test_main(render=False, n_runs=1, timesteps=200)
    assert isinstance(history, dict)

def test_evaluation_two_tables():
    from evaluation.dinovas_tworobots_without_obst_twotables import main
    blueprint_test_evaluation(main)

def test_evaluation_no_obst():
    from evaluation.dinovas_tworobots_without_obst_compare import main
    blueprint_test_evaluation(main)

def test_evaluation_with_obst():
    from evaluation.dinovas_tworobots_static_obst_compare import main
    blueprint_test_evaluation(main)
