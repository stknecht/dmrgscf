#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Stefan Knecht <stefan@algorithmiq.fi>
#

'''
QCMaquis DMRG solver for CASCI and CASSCF.
'''
import ctypes
import os
import sys
import struct

import tempfile
from subprocess import check_call, check_output, STDOUT, CalledProcessError
import numpy
from pyscf import lib
from pyscf import tools
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf import mcscf
from pyscf.dmrgscf import dmrg_sym
from pyscf import __config__


# Settings
try:
    from pyscf.dmrgscf import settings
except ImportError:
    settings = lambda: None
    settings.QCMVERSION = getattr(__config__, 'dmrgscf_QCMVERSION', None)
    settings.QCMLIB = getattr(__config__, 'dmrgscf_QCMLIB', None)
    if (settings.QCMVERSION is None or settings.QCMLIB is None):
        sys.stderr.write('settings.py not found for module qcmaquis.  Please create %s\n'
                         % os.path.join(os.path.dirname(__file__), 'settings.py'))

# Libraries
libqcm = lib.load_library(settings.QCMLIB)

# fcidumpFromIntegral = libqcm.fcidumpFromIntegral
# fcidumpFromIntegral.restype = None
# fcidumpFromIntegral.argtypes = [
#     ctypes.c_char_p,
#     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
#     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
#     ctypes.c_size_t,
#     ctypes.c_size_t,
#     ctypes.c_double,
#     ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
#     ctypes.c_size_t,
# ]
sayhello = libqcm.qcmaquis_interface_say_hello
sayhello.restype = None
sayhello.argtypes = [
    ctypes.c_char_p,
]

class qcmDMRGCI(lib.StreamObject):
    '''QCMaquis program interface and the object to hold QCMaquis program input parameters.

    Attributes:

    Examples:

    >>> mol = gto.M(atom='C 0 0 0; C 0 0 1')
    >>> mf = scf.RHF(mol).run()
    >>> mc = mcscf.CASCI(mf, 4, 4)
    >>> mc.fcisolver = qcmDMRGCI(mol)
    >>> mc.kernel()
    -74.379770619390698
    '''
    def __init__(self, mol=None, maxM=None, tol=None, num_thrds=1, memory=None):
        self.mol = mol
        print("hello my friend")
        string1 = "YEAH!"
        # create byte objects from the strings
        b_string1 = string1.encode('utf-8')
        sayhello(b_string1)
        if mol is None:
            self.stdout = sys.stdout
            self.verbose = logger.NOTE
        else:
            self.stdout = mol.stdout
            self.verbose = mol.verbose
        self.outputlevel = 2
        self.hf_occ = 'integral'


        # self.executable = settings.BLOCKEXE
        # self.scratchDirectory = os.path.abspath(settings.BLOCKSCRATCHDIR)
        # self.mpiprefix = settings.MPIPREFIX
        self.memory = memory

        self.integralFile = "FCIDUMP"
        self.configFile = "dmrg.conf"
        self.outputFile = "dmrg.out"
        if getattr(settings, 'BLOCKRUNTIMEDIR', None):
            self.runtimeDir = settings.BLOCKRUNTIMEDIR
        else:
            self.runtimeDir = '.'
        self.maxIter = 20
        self.approx_maxIter = 6
        self.twodot_to_onedot = 15
        self.dmrg_switch_tol = 1e-3
        self.nroots = 1
        self.weights = []
        self.wfnsym = 1
        self.extraline = []

        if tol is None:
            self.tol = 1e-7
        else:
            self.tol = tol
        if maxM is None:
            self.maxM = 1000
        else:
            self.maxM = maxM
        self.num_thrds= num_thrds
        self.startM =  None
        self.restart = False
        self.nonspinAdapted = False
        self.scheduleSweeps = []
        self.scheduleMaxMs  = []
        self.scheduleTols   = []
        self.scheduleNoises = []
        self.onlywriteIntegral = False
        self.spin = 0
        self.orbsym = []
        if mol is None:
            self.groupname = None
        else:
            if mol.symmetry:
                self.groupname = mol.groupname
            else:
                self.groupname = None
        ##################################################
        # don't modify the following attributes, if you do not finish part of calculation, which can be reused.
        #DO NOT CHANGE these parameters, unless you know the code in details
        self.twopdm = True #By default, 2rdm is calculated after the calculations of wave function.
        self.block_extra_keyword = [] #For Block advanced user only.
        self.has_fourpdm = False
        self.has_threepdm = False
        self.has_nevpt = False
        # This flag _restart is set by the program internally, to control when to make
        # Block restart calculation.
        self._restart = False
        # self.generate_schedule()
        self.returnInt = False
        self._keys = set(self.__dict__.keys())


    @property
    def max_memory(self):
        if self.memory is None:
            return self.memory
        elif isinstance(self.memory, int):
            return self.memory * 1e3 # GB -> MB
        else:  # str
            val, unit = self.memory.split(',')
            if unit.trim().upper() == 'G':
                return float(val) * 1e3
            else: # MB
                return float(val)
    @max_memory.setter
    def max_memory(self, x):
        self.memory = x * 1e-3

    @property
    def threads(self):
        return self.num_thrds
    @threads.setter
    def threads(self, x):
        self.num_thrds = x

