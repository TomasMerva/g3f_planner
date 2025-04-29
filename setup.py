# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['g3f_planner',
 'g3f_planner.scripts.constraints',
 'g3f_planner.scripts.cpp_bridge',
 'g3f_planner.scripts.rollout',
 'g3f_planner.scripts.solver',
 'g3f_planner.scripts.utils']

package_data = \
{'': ['*'], 'g3f_planner': ['src/*']}

install_requires = \
['casadi==3.6.6',
 'fabrics<0.9.6',
 'latextable>=1.0.1,<2.0.0',
 'osqp>=0.6.5',
 'robotmodels @ git+https://github.com/maxspahn/robotmodels.git',
 'spatial-casadi==1.1.0',
 'texttable>=1.7.0,<2.0.0',
 'tqdm>=4.66.5,<5.0.0',
 'urdfenvs>=0.9.13,<0.10.0']

setup_kwargs = {
    'name': 'g3f_planner',
    'version': '0.1.0',
    'description': '',
    'long_description': '# Globally-Guided Geometric Fabrics\nImplementation of Globally-Guided Geometric Fabrics presented in our paper **"Globally-Guided Geometric Fabrics for Reactive Mobile Manipulation in Dynamic Environments"**\n\n[![](assets/paper_teaser.png)](https://ieeexplore.ieee.org/abstract/document/10967245/)\n\nMobile manipulators operating in dynamic environments shared with humans and robots must adapt in\nreal time to environmental changes to complete their tasks effectively. While global planning methods are effective at\nconsidering the full task scope, they lack the computational efficiency required for reactive adaptation. In contrast, local\nplanning approaches can be executed online but are limited by their inability to account for the full task’s duration. To\ntackle this, we propose **Globally-Guided Geometric Fabrics (G3F)**, a framework for real-time motion generation along\nthe full task horizon, by interleaving an optimization-based planner with a fast reactive geometric motion planner, called\nGeometric Fabrics (GF). The approach adapts the path and explores a multitude of acceptable target poses, while\naccounting for collision avoidance and the robot’s physical constraints. This results in a real-time adaptive framework\nconsidering whole-body motions, where a robot operates in close proximity to other robots and humans. We validate\nour approach through various simulations and real-world experiments on mobile manipulators in multi-agent settings,\nachieving improved success rates compared to vanilla GF, Prioritized Rollout Fabrics and Model Predictive Control.\n\nA **project page** showcasing the presented approach can be found [here](https://autonomousrobots.nl/paper_websites/g3f).\n\n### How to cite this work\nIf you found this repository useful, please consider citing the associated paper below:\n\n```bash\n@article{merva2025globally,\n  title={Globally-Guided Geometric Fabrics for Reactive Mobile Manipulation in Dynamic Environments},\n  author={Merva, Tomas and Bakker, Saray and Spahn, Max and Zhao, Danning and Virgala, Ivan and Alonso-Mora, Javier},\n  journal={IEEE Robotics and Automation Letters},\n  year={2025},\n  publisher={IEEE}\n}\n```\n\n## Teaser\n<img src="assets/video_crossover.gif" alt="2 Robots applying G3F" height="300">\n\n\n## Build\n### Fabrics controller (C++)\n\nYou can automatically generate C++ code for a custom fabrics controller using the [create_dinovas_planner.py](https://github.com/TomasMerva/grasp_fabrics/blob/main/examples/generate_controllers/create_dinovas_planner.py) example. The custom fabrics controller is built based on the robot\'s URDF file and [configuration files](https://github.com/TomasMerva/grasp_fabrics/tree/main/config), and is stored in the [g3f_planner/src/](https://github.com/TomasMerva/grasp_fabrics/tree/main/fabrics_rollouts/src) folder. To use it, you need to rebuild it in the `build` folder or run the following script:\n\n```bash\n./build.sh\n```\nThe precompiled library \n\n### Virtual environment (advised)\nYou can install the necessary dependencies using [poetry](https://python-poetry.org/docs/) virtual environment. After installing poetry, move into `g3f_planner` and run:\n```bash\npoetry install\n```\nAccess the virtual environment using:\n```bash\npoetry shell\n```\n\n\n## Example\nThe example code for the G3F planner, specifically tailored to the [Dinova](https://github.com/INTERACT-tud-amr/dinova) mobile manipulator, is provided in `examples/example_g3f_dinovas.py`, and the corresponding optimization problem is formulated in `examples/g3f_planner_dinova.py`. To run the example:\n```bash\npython examples/example_g3f_dinovas.py\n```\n\n\n',
    'author': 'Tomas Merva',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://github.com/TomasMerva/grasp_fabrics',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<3.10',
}


setup(**setup_kwargs)

