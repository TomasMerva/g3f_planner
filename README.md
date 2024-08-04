# grasp_fabrics


## Requirements
1. [yaml-cpp](https://github.com/jbeder/yaml-cpp)
```bash
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp && mkdir build
cd build
cmake ..
make
sudo make install
```

## Build
1. Fabrics Server (c++)
```bash
git clone git@github.com:TomasMerva/grasp_fabrics.git
cd grasp_fabrics && mkdir build
cd build
cmake ..
make
```
2. Fabrics Client (python)
```bash
cd grasp_fabrics
poetry install
```

## Use
1. Fabrics Server (Terminal 1): 
```bash
./build/rollout_fabrics
```

2. Fabrics Client (Terminal 2)
```bash
python examples/rollouts/dinova_planner_client.py
```

## Warnings
1. generated fabrics C-code has to be in the `/fabrics_rollouts/headers/` folder
2. if you change fabrics C-Code, you need to rebuild fabrics server
