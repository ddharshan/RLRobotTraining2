
"""Module setuptools script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from setuptools import setup, find_packages
from importlib import import_module

here = os.path.abspath(os.path.dirname(__file__))
meta_module = import_module('RobotTraining2')  #change--------------------------------------------------------------------------------------
meta = meta_module.__dict__
 

setup(
     
    url='https://github.com/ddharshan/RLRobotTraining2', #change---------------------------------------------------------------------------------
    keywords='Robot Training2', #change ---------------------------------------------------------------------------------------------------
    packages=[
         
        # application change ------------------------------------------------------------------------------
        *find_packages(include=('RobotTraining2'
                                'RobotTraining2.*')),  
    ],
 
)
