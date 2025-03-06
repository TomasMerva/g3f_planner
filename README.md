# Globally-Guided Geometric Fabrics
Implementation of Globally-Guided Geometric Fabrics presented in our paper **"Globally-Guided Geometric Fabrics for Reactive Mobile Manipulation in Dynamic Environments"**

TODO: Add paper teaser image

Mobile manipulators operating in dynamic environments shared with humans and robots must adapt in
real time to environmental changes to complete their tasks effectively. While global planning methods are effective at
considering the full task scope, they lack the computational efficiency required for reactive adaptation. In contrast, local
planning approaches can be executed online but are limited by their inability to account for the full task’s duration. To
tackle this, we propose **Globally-Guided Geometric Fabrics (G3F)**, a framework for real-time motion generation along
the full task horizon, by interleaving an optimization-based planner with a fast reactive geometric motion planner, called
Geometric Fabrics (GF). The approach adapts the path and explores a multitude of acceptable target poses, while
accounting for collision avoidance and the robot’s physical constraints. This results in a real-time adaptive framework
considering whole-body motions, where a robot operates in close proximity to other robots and humans. We validate
our approach through various simulations and real-world experiments on mobile manipulators in multi-agent settings,
achieving improved success rates compared to vanilla GF, Prioritized Rollout Fabrics and Model Predictive Control.

A **project page** showcasing the presented approach can be found [here](https://autonomousrobots.nl/paper_websites/g3f).

### How to cite this work
If you found this repository useful, please consider citing the associated paper below:

```bash
TODO:add_bibtex
```



## Build
1. Fabrics controller (C++)

You can automatically generate C++ code for a custom fabrics controller using the [create_dinovas_planner.py](https://github.com/TomasMerva/grasp_fabrics/blob/main/examples/generate_controllers/create_dinovas_planner.py) example. The custom fabrics controller is built based on the robot's URDF file and [configuration files](https://github.com/TomasMerva/grasp_fabrics/tree/main/config), and is stored in the [g3f_planner/src/](https://github.com/TomasMerva/grasp_fabrics/tree/main/fabrics_rollouts/src) folder. To use it, you need to rebuild it in the `build` folder or run the following script:

```bash
./build.sh
```
The precompiled library 

2. Virtual environment (advised)
You can install the necessary dependencies using [poetry](https://python-poetry.org/docs/) virtual environment. After installing poetry, move into `g3f_planner` and run:
```bash
poetry install
```
Access the virtual environment using:
```bash
poetry shell
```


## Example
The example code for the G3F planner, specifically tailored to the [Dinova](https://github.com/INTERACT-tud-amr/dinova) mobile manipulator, is provided in `examples/example_g3f_dinovas.py`, and the corresponding optimization problem is formulated in `examples/g3f_planner_dinova.py`. To run the example:
```bash
python examples/example_g3f_dinovas.py
```


