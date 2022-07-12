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
    # ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
    ctypes.c_void_p,
    ctypes.c_double,
    ctypes.c_int32,
    ctypes.c_int32,
    # ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
    ctypes.c_void_p,
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

qcmSETSTATE = libqcm.qcmaquis_interface_set_state
qcmSETSTATE.restype = None
qcmSETSTATE.argtypes = [
    ctypes.c_int32,
]

qcmSETKW    = libqcm.qcmaquis_interface_set_param
qcmSETKW.restype = None
qcmSETKW.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
]

qcmOPT = libqcm.qcmaquis_interface_optimize
qcmOPT.restype = None
qcmOPT.argtypes = [
]

qcmENERGY = libqcm.qcmaquis_interface_get_energy
qcmENERGY.restype = ctypes.c_double
qcmENERGY.argtypes = [
]

qcmOPT = libqcm.qcmaquis_interface_optimize
qcmOPT.restype = None
qcmOPT.argtypes = [
]

qcmIOD = libqcm.qcmaquis_interface_stdout
qcmIOD.restype = None
qcmIOD.argtypes = [
    ctypes.c_char_p,
]

qcmIOR = libqcm.qcmaquis_interface_restore_stdout
qcmIOR.restype = None
qcmIOR.argtypes = [
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
                 maxIter=10, project="myQCM"):

        print("init again ... {}".format(project))
        self.mol = mol
        if mol is None:
            self.stdout = sys.stdout
            self.verbose = logger.NOTE
        else:
            self.stdout = mol.stdout
            self.verbose = mol.verbose
        self.outputlevel = 2

        self.maxIter = maxIter
        self.dmrg_switch_tol = 1e-3
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
        self.project = project
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
            if mol.spin:
                self.spin = mol.spin
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
        for p in range(0, norb):
            for q in range(0, norb):
                for r in range(min(p,q), norb):
                    for s in range(r, norb):
                        sz+=1
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

        sz = qcmDMRGCI.get_size12(norb)
        rdm2 = numpy.ascontiguousarray(numpy.zeros( (  sz) ))
        ints = numpy.ascontiguousarray(numpy.zeros( (4*sz),
                                       dtype=numpy.int32 ),
                                       dtype=numpy.int32)
        # read packed 2-RDM from QCMaquis
        qcmRDM12(ints, rdm2, sz)

        # expand to full 2-RDM and reorder indices
        twopdm = get_full_2rdm(rdm2, ints, sz, norb)

        # (This is coherent with the previous statement about indices)
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
        setHAM(self, h1e, eri, norb, nelec, ecore)
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

        ''' setup QCMaquis configuration '''
        setQCM(self, norb, nelec)
        ''' run QCMaquis '''
        for i in range(0,self.nroots):
            runQCM(self, i)
            calc_e = eneQCM(self)
            # if self.restart:
            #     # Restart only the first iteration
            #     self.restart = False
            print("Optimized energy: {}".format(calc_e))
        return calc_e, roots

    def approx_kernel(self, h1e, eri, norb, nelec, fciRestart=None, ecore=0, **kwargs):
        fciRestart = True

        if self.nroots == 1:
            roots = 0
        else:
            roots = range(self.nroots)

        if 'orbsym' in kwargs:
            self.orbsym = kwargs['orbsym']

        ''' transfer Hamiltonian (1e- and 2e-integrals) to QCMaquis '''
        setHAM(self, h1e, eri, norb, nelec, ecore)

        ''' setup QCMaquis configuration '''
        setQCM(self, norb, nelec)

        ''' run QCMaquis '''
        for i in range(0,self.nroots):
            runQCM(self, i)
            calc_e = eneQCM(self)
            # if self.restart:
            #     # Restart only the first iteration
            #     self.restart = False
            print("Optimized energy: {}".format(calc_e))
        return calc_e, roots

    def restart_scheduler_(self):
        def callback(envs):
            info_str = ""
            self._restart = False
            if (envs['norm_gorb'] < self.dmrg_switch_tol):
                self._restart = True
                info_str += "Orb grad < dmrg_switch_tol "
            if 'norm_ddm' in envs and envs['norm_ddm'] < self.dmrg_switch_tol*10:
                self._restart = True
                info_str += "Norm_ddm < dmrg_switch_tol*10 "
            if self._restart:
                logger.debug(self, "%s, set DMRG restart", info_str)
        return callback

def qcmDMRGSCF(mf, norb, nelec, maxM=1000, tol=1.e-8,
               maxIter=10, project="myQCM", *args, **kwargs):
    '''Shortcut function to setup CASSCF using the QCMaquis-DMRG solver.
    The DMRG solver is properly initialized in this function so that the 1-step
    algorithm can be applied with DMRG-CASSCF.
    Examples:
    >>> mol = gto.M(atom='C 0 0 0; C 0 0 1')
    >>> mf = scf.RHF(mol).run()
    >>> mc = qcmDMRGSCF(mf, 4, 4)
    >>> mc.kernel()
    -74.414908818611522
    '''
    if getattr(mf, 'with_df', None):
        mc = mcscf.DFCASSCF(mf, norb, nelec, *args, **kwargs)
    else:
        mc = mcscf.CASSCF(mf, norb, nelec, *args, **kwargs)
    print("incoming project name is {}".format(project))
    mc.fcisolver = qcmDMRGCI(mf.mol, maxM=maxM, tol=tol, maxIter=maxIter,
                             project=project)
    mc.callback = mc.fcisolver.restart_scheduler_()

    if mc.chkfile == mc._scf._chkfile.name:
        # Do not delete chkfile after mcscf
        mc.chkfile = tempfile.mktemp(dir=settings.QCMSCRATCHDIR)
        if not os.path.exists(settings.QCMSCRATCHDIR):
            os.makedirs(settings.QCMSCRATCHDIR)
    return mc

def setHAM(qcmDMRGCI, h1e, eri, norb, nelec, ecore):

    ueri, ieri, nint = get_unique_eri(h1e, eri, ecore, norb)
    qcmUPINT(ieri, ueri, nint)

def setQCM(qcmDMRGCI, norb, nelec):

    nele = 0
    if isinstance(nelec, (int, numpy.integer)):
       nele = nelec
    else:
       nele = nelec[0]+nelec[1]

    print("set QCMaquis config ... N e-: {} norb: {} and Project={}".format(nele,norb,qcmDMRGCI.project))

    # DOES NOT work for symmetry yet, this requires the first none to provide the orbital symmetries
    qcmINIT(nele, norb, qcmDMRGCI.spin, (qcmDMRGCI.wfnsym - 1),
            None, qcmDMRGCI.tol, qcmDMRGCI.maxM, qcmDMRGCI.maxIter,
            None, -1, qcmDMRGCI.project.encode('utf-8'), qcmDMRGCI.twopdm)

    qcmSETKW("MEASURE[1rdm]".encode('utf-8'), "0".encode('utf-8'))

def runQCM(qcmDMRGCI, state):

    # I/O redirect
    qcmIO = qcmDMRGCI.project + ".optimization.out"
    qcmIOD(qcmIO.encode('utf-8'))

    # update # of sweeps
    # qcmaquis_interface_set_nsweeps(nsweeps_,ngrowsweeps_,nmainsweeps_);

    # set I/O files for state i
    qcmSETSTATE(state)

    # given state i: optimize MPS + calculate <MPS|O|MPS> for a list of O's
    qcmOPT()

    # restore I/O
    qcmIOR()

def eneQCM(qcmDMRGCI):

    # extract state-specific electronic energy after MPS optimization
    return qcmENERGY();

def get_unique_eri(h1e, eri, ecore, nmo, tol=1e-99):
    npair = nmo*(nmo+1)//2

    if eri.size == nmo**4:
        print('restore')
        eri = ao2mo.restore(8, eri, nmo)

    #                2e-             1e-   core-energy
    dim_eri = npair*(npair+1)//2 + npair + 1

    ueri = numpy.zeros( (  dim_eri) )
    ieri = numpy.zeros( (4*dim_eri), dtype=numpy.int32 )

    uindex = 0
    # 2e-part: assume either 4-fold symmetry for eri (default in pySCF)
    if eri.ndim == 2: # 4-fold symmetry
        assert(eri.size == npair**2)
        ij = 0
        for i in range(nmo):
            for j in range(0, i+1):
                kl = 0
                for k in range(0, nmo):
                    for l in range(0, k+1):
                        if ij >= kl:
                            if abs(eri[ij,kl]) > tol:
                                # print('({},{},{},{}) = {} '.format(i+1, j+1, k+1, l+1,eri[ij,kl]))
                                ueri[uindex] = eri[ij,kl]
                                ieri[4*uindex] = i+1
                                ieri[4*uindex+1] = j+1
                                ieri[4*uindex+2] = k+1
                                ieri[4*uindex+3] = l+1
                                uindex += 1
                        kl += 1
                ij += 1
    else: # 8-fold symmetry
        # print('8-fold ')
        assert(eri.size == npair*(npair+1)//2)
        ij = 0
        ijkl = 0
        for i in range(nmo):
            for j in range(0, i+1):
                kl = 0
                for k in range(0, i+1):
                    for l in range(0, k+1):
                        if ij >= kl:
                            if abs(eri[ijkl]) > tol:
                                # print('({},{},{},{}) = {} '.format(i+1, j+1, k+1, l+1,eri[ijkl]))
                                ueri[uindex] = eri[ijkl]
                                ieri[4*uindex] = i+1
                                ieri[4*uindex+1] = j+1
                                ieri[4*uindex+2] = k+1
                                ieri[4*uindex+3] = l+1
                                uindex += 1
                            ijkl += 1
                        kl += 1
                ij += 1

    # 1e-part
    ij = 0
    for i in range(nmo):
        for j in range(0, i+1):
            if abs(h1e[i,j]) > tol:
                # print('({},{}) = {} '.format(i+1, j+1,h1e[i,j]))
                ueri[uindex] = h1e[i,j]
                ieri[4*uindex] = i+1
                ieri[4*uindex+1] = j+1
                uindex += 1
    # ecore
    ueri[uindex] = ecore
    return numpy.ascontiguousarray(ueri), numpy.ascontiguousarray(ieri,dtype=numpy.int32), dim_eri

def get_full_2rdm(c2rdm, cindx, nelements, norb):
    twopdm = numpy.zeros( (norb, norb, norb, norb) )
    for i in range(0,nelements):
        ii = 4*i
        p = cindx[ii+0]
        q = cindx[ii+1]
        r = cindx[ii+2]
        s = cindx[ii+3]
        # reorder from E2[p1,q2,r2,s1] -> E2[p1,s1,q2,r2] (for PySCF purposes)
        twopdm[p,s,q,r] = c2rdm[i]
        twopdm[q,r,p,s] = c2rdm[i]
        twopdm[r,q,s,p] = c2rdm[i]
        twopdm[s,p,r,q] = c2rdm[i]
    return twopdm

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    b = 1.4
    mol = gto.Mole()
    mol.build(
        verbose = 7,
        output = 'out-dmrgci',
        atom = [['H', (0.,0.,i-3.5)] for i in range(8)],
        basis = {'H': 'sto-3g'},
        symmetry = True
    )
    m = scf.RHF(mol)
    m.scf()

    mc = DMRGSCF(m, 4, 4)
    mc.fcisolver.tol = 1e-9
    emc_1 = mc.mc2step()[0]

    mc = mcscf.CASCI(m, 4, 4)
    mc.fcisolver = DMRGCI(mol)
    emc_0 = mc.casci()[0]

    b = 1.4
    mol = gto.Mole()
    mol.build(
        verbose = 7,
        output = 'out-casscf',
        atom = [['H', (0.,0.,i-3.5)] for i in range(8)],
        basis = {'H': 'sto-3g'},
        symmetry = True
    )
    m = scf.RHF(mol)
    m.scf()

    mc = mcscf.CASSCF(m, 4, 4)
    emc_1ref = mc.mc2step()[0]

    mc = mcscf.CASCI(m, 4, 4)
    emc_0ref = mc.casci()[0]

    print('DMRGCI  = %.15g CASCI  = %.15g' % (emc_0, emc_0ref))
    print('DMRGSCF = %.15g CASSCF = %.15g' % (emc_1, emc_1ref))
