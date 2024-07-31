# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['fabrics_rollouts', 'fabrics_rollouts.scripts']

package_data = \
{'': ['*'],
 'fabrics_rollouts': ['build/*',
                      'build/CMakeFiles/*',
                      'build/CMakeFiles/3.16.3/*',
                      'build/CMakeFiles/3.16.3/CompilerIdC/*',
                      'build/CMakeFiles/3.16.3/CompilerIdCXX/*',
                      'build/CMakeFiles/rollout_fabrics.dir/*',
                      'build/CMakeFiles/rollout_fabrics.dir/src/*',
                      'config/*',
                      'headers/*',
                      'src/*']}

install_requires = \
['fabrics @ git+https://github.com/tud-amr/fabrics.git',
 'grasp_planning>=0.6,<0.7',
 'pytorch_kinematics>=0.6.1,<0.7.0',
 'robotmodels @ git+https://github.com/maxspahn/robotmodels.git',
 'urdfenvs>=0.9.13,<0.10.0']

setup_kwargs = {
    'name': 'fabrics_rollouts',
    'version': '0.1.0',
    'description': '',
    'long_description': '# grasp_fabrics\n\n\n## Requirements\n1. [yaml-cpp](https://github.com/jbeder/yaml-cpp)\n```bash\ngit clone https://github.com/jbeder/yaml-cpp.git\ncd yaml-cpp && mkdir build\ncd build\ncmake ..\nmake\nsudo make install\n```\n\n## Build\n1. Fabrics Server (c++)\n```bash\ngit clone git@github.com:TomasMerva/grasp_fabrics.git\ncd grasp_fabrics/rollouts && mkdir build\ncd build\ncmake ..\nmake\n```\n2. Fabrics Client (python)\n```bash\ncd grasp_fabrics\npoetry install\n```\n\n## Use\n1. Fabrics Server (Terminal 1): \n```bash\n./rollouts/build/rollout_fabrics\n```\n\n2. Fabrics Client (Terminal 2)\n```bash\npython rollouts/create_dinova_planner.py\n```\n\n## TODO\n1. Python script for creating communication protocol\n- right now, it is manually predefined and the msg order or a fabrics controller cannot be changed\n2. closing server socket on the server side\n',
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

