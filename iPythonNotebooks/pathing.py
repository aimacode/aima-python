"""This small utility is used specifically for 
   iPython Notebooks to create a import module
   path to the directory containing all of the
   algorithms, namely the parent directory.
"""

import os
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd.rstrip('iPythonNotebooks')+"aimaPy/")
