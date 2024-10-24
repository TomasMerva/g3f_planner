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
        history = test_main(render=False, n_runs=1, timesteps=200, save_data=False)
    assert isinstance(history, dict)

def test_evaluation_two_tables():
    from evaluation.dinovas_crossover.dinovas_crossover_twotables_compare import main
    blueprint_test_evaluation(main)

def test_evaluation_no_obst():
    from examples.evaluation.single_agent.dinova_single_agent_compare import main
    blueprint_test_evaluation(main)

def test_evaluation_with_obst():
    from examples.evaluation.dinovas_static.dinovas_static_obst_compare import main
    blueprint_test_evaluation(main)

def test_evaluation_with_obst():
    from evaluation.dinovas_3robots.dinovas_3robots_compare import main
    blueprint_test_evaluation(main)