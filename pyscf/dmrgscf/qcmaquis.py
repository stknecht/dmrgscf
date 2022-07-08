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

ndpointer = numpy.ctypeslib.ndpointer

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

# qcmINIT = libqcm.qcmaquis_interface_preinit(int nel, int L, int spin, int irrep,
#                                 const int* site_types, V conv_thresh, int m, int nsweeps,
#                                 const int* sweep_m, int nsweepm, const char* project_name,
#                                 bool meas_2rdm)

qcmINIT = libqcm.qcmaquis_interface_preinit
qcmINIT.restype = None
qcmINIT.argtypes = [
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
    ctypes.c_double,
    ctypes.c_int32,
    ctypes.c_int32,
    ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_char_p,
    ctypes.c_bool,
]

# qcmRDM12 = libqcm.qcmaquis_interface_get_2rdm(int* indices, V* values, int size)
qcmRDM12 = libqcm.qcmaquis_interface_get_2rdm
qcmRDM12.restype = None
qcmRDM12.argtypes = [
    ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
]

# qcmRDM12 = libqcm.qcmaquis_interface_update_integrals(const int* integral_indices, const V* integral_values, int integral_size) 
qcmUPINT = libqcm.qcmaquis_interface_update_integrals
qcmUPINT.restype = None
qcmUPINT.argtypes = [
    ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
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
    def __init__(self, mol=None, maxM=None, tol=None, num_thrds=1,
                 maxIter=10):
        self.mol = mol
        print("hello my friend")
        string1 = "YEAH!"
        # create byte objects from the strings
        sayhello(string1.encode('utf-8'))
        if mol is None:
            self.stdout = sys.stdout
            self.verbose = logger.NOTE
        else:
            self.stdout = mol.stdout
            self.verbose = mol.verbose
        self.outputlevel = 2

        self.maxIter = 10
        self.nroots  = 1
        self.weights = []
        self.wfnsym  = 1

        if tol is None:
            self.tol = 1e-7
        else:
            self.tol = tol
        if maxM is None:
            self.maxM = 1000
        else:
            self.maxM = maxM
        self.num_thrds= num_thrds
        self.restart = False
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
        # This flag _restart is set by the program internally, to control when to make
        # a QCMaquis restart calculation.
        self._restart = False
        self.returnInt = False
        self._keys = set(self.__dict__.keys())


    @property
    def threads(self):
        return self.num_thrds
    @threads.setter
    def threads(self, x):
        self.num_thrds = x

    def get_size12(norb):
        sz = 0
        p = 0
        while p < norb:
            q = 0
            while q < norb:
                r = 0
                while r < min(p,q):
                    s = r
                    while s < norb:
                        sz+=1
                        s+=1
                    r+=1
                q+=1
            p+=1
        return sz

    def make_rdm1(self, state, norb, nelec, link_index=None, **kwargs):
        # Avoid calling self.make_rdm12 because it may be overloaded
        return qcmDMRGCI.make_rdm12(self, state, norb, nelec, link_index, **kwargs)[0]

    def make_rdm12(self, state, norb, nelec, link_index=None, **kwargs):
        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
          nelectrons = nelec
        else:
          nelectrons = nelec[0]+nelec[1]

        # The 2RDMs written by "save_spatial_twopdm_text" in BLOCK and STACKBLOCK
        # are written as E2[i1,j2,k2,l1]
        # and stored here as E2[i1,l1,j2,k2] (for PySCF purposes)
        # This is NOT done with SQA in mind.
        sz = qcmDMRGCI.get_size12(norb)
        rdm2 = numpy.zeros( (  sz) )
        ints = numpy.zeros( (4*sz), dtype=numpy.int32 )
        # qcmRDM12(ints, rdm2, sz)

        twopdm = numpy.zeros( (norb, norb, norb, norb) )

        # (This is coherent with previous statement about indexes)
        onepdm = numpy.einsum('ikjj->ki', twopdm)
        onepdm /= (nelectrons-1)
        return onepdm, twopdm

    def kernel(self, h1e, eri, norb, nelec, fciRestart=None, ecore=0, **kwargs):
        if self.nroots == 1:
            roots = 0
        else:
            roots = range(self.nroots)
        if fciRestart is None:
            fciRestart = self.restart or self._restart

        if 'orbsym' in kwargs:
            self.orbsym = kwargs['orbsym']

        ''' transfer Hamiltonian (1e- and 2e-integrals) to QCMaquis '''
        print('hello from kernel - sz is {}'.format(qcmDMRGCI.get_size12(norb = norb)))
        print('hello from kernel - 1e ints are')
        print(h1e)
        print('hello from kernel - 2e ints are')
        print(eri)
        print('hello from kernel - ecore is {}'.format(ecore))
        setHAM(self, h1e, eri, norb, nelec, ecore)
        #writeDMRGConfFile(self, nelec, fciRestart)
        #if self.verbose >= logger.DEBUG1:
        #    inFile = os.path.join(self.runtimeDir, self.configFile)
        #    logger.debug1(self, 'Block Input conf')
        #    logger.debug1(self, open(inFile, 'r').read())
        if self.onlywriteIntegral:
            logger.info(self, 'Only write integral')
            try:
                calc_e = 0.0 # readEnergy(self)
            except IOError:
                if self.nroots == 1:
                    calc_e = 0.0
                else :
                    calc_e = [0.0] * self.nroots
            return calc_e, roots
        if self.returnInt:
            return h1e, eri

        # executeBLOCK(self)
        # if self.verbose >= logger.DEBUG1:
        #     outFile = os.path.join(self.runtimeDir, self.outputFile)
        #     logger.debug1(self, open(outFile).read())
        # calc_e = readEnergy(self)
        # if self.restart:
        #     # Restart only the first iteration
        #     self.restart = False
        calc_e = 0.0 # readEnergy(self)

        return calc_e, roots

def setHAM(qcmDMRGCI, h1e, eri, norb, nelec, ecore):

    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelec//2 + nelec%2
        nelecb = nelec - neleca
    else :
        neleca, nelecb = nelec

    ueri, ieri = get_unique_eri(eri,norb)
    # qcmUPINT()

def get_unique_eri(eri, nmo, tol=1e-99):
    npair = nmo*(nmo+1)//2

    dim_eri = npair*(npair+1)//2

    ueri = numpy.zeros( (  dim_eri) )
    ieri = numpy.zeros( (4*dim_eri), dtype=numpy.int32 )

    # assumes 4-fold symmetry (default in pySCF)
    if eri.ndim == 2: # 4-fold symmetry
        assert(eri.size == npair**2)
        ij = 0
        uindex = 0
        for i in range(nmo):
            for j in range(0, i+1):
                kl = 0
                for k in range(0, nmo):
                    for l in range(0, k+1):
                        if ij >= kl:
                            if abs(eri[ij,kl]) > tol:
                                print('({},{},{},{}) = {} '.format(i+1, j+1, k+1, l+1,eri[ij,kl]))
                                ueri[uindex] = eri[ij,kl]
                                ieri[uindex] = i
                                ieri[uindex+1] = j
                                ieri[uindex+2] = k
                                ieri[uindex+3] = l
                                uindex += 1
                        kl += 1
                ij += 1
    return ueri, ieri

