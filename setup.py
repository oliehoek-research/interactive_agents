#!/usr/bin/env python
from setuptools import setup

setup(name='interactive_agents',
      version='0.0.1',
      description='Reinforcement learning for Human-AI interaction',
      author='Robert Loftin',
      author_email='r.t.loftin@tudelft.nl',
      packages=['interactive_agents'],
      install_requires=[
            "GitPython==3.1.26",
            "gym==0.22.0",
            "matplotlib==3.5.1",
            "numpy>=1.22",
            "pandas==1.3.4",
            "PettingZoo==1.17.0",
            "protobuf==3.19.4",  # What did we need this for (we never import it ourselves)? - is this specific version important?
            "pyglet==1.5.15",  # We only need this for certain visualizations
            "PyYAML==6.0",
            "scipy==1.7.2",
            "SuperSuit==3.3.5",
            "tensorboardX==2.5",
            "torch==1.11.0"
      ]
)