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
cd grasp_fabrics/rollouts && mkdir build
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
./rollouts/build/rollout_fabrics
```

2. Fabrics Client (Terminal 2)
```bash
python rollouts/create_dinova_planner.py
```

## TODO
1. Python script for creating communication protocol
- right now, it is manually predefined and the msg order or a fabrics controller cannot be changed
2. closing server socket on the server side
