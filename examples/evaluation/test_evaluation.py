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
        history = test_main(render=False, n_runs=1, timesteps=200, save_data=False, cases=["G3F","GF", "PRF"],)
    assert isinstance(history, dict)

def test_evaluation_two_tables():
    from examples.evaluation.dinovas_two_tables.dinovas_twotables_compare import main
    blueprint_test_evaluation(main)

def test_evaluation_one_table():
    from examples.evaluation.dinovas_one_table.dinovas_onetable_compare import main
    blueprint_test_evaluation(main)

def test_evaluation_noncooperative():
    from examples.evaluation.dinovas_single_agent.dinovas_singleagent_compare import main
    blueprint_test_evaluation(main)

def test_evaluation_with_obst():
    from evaluation.dinovas_three_robots.dinovas_threerobots_compare import main
    blueprint_test_evaluation(main)