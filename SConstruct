#!/usr/bin/env scons
#

import os

from os.path import join
from nestly.scons import SConsWrap
from nestly import Nest
import SCons.Script as sc

# Command line options

sc.AddOption('--seeds', type='int', help="SEEDS", default=1)
sc.AddOption('--output', type='string', help="output folder", default='_output')

env = sc.Environment(
        ENV=os.environ,
        num_seeds=sc.GetOption('seeds'),
        output=sc.GetOption('output'))

sc.Export('env')

env.SConsignFile()

# Simulation scenario 1
flag = 'simulations_recalib'
sc.SConscript(flag + '/sconscript', exports=['flag'])

# Simulation scenario 2
flag = 'simulations_revise'
sc.SConscript(flag + '/sconscript', exports=['flag'])

# Simulation scenario 3
flag = 'simulations_refit'
sc.SConscript(flag + '/sconscript', exports=['flag'])

# COPD data analysis
#flag = 'analysis_copd'
#sc.SConscript(flag + '/sconscript', exports=['flag'])
