from sklearn.externals import joblib
import numpy as np
import pandas as pd
import os
import pickle
import sys
from PyBioMed import Pymolecule
from PyBioMed.PyInteraction import PyInteraction
from pychem import topology,moreaubroto,estate
from rdkit.Chem import AllChem
from rdkit import Chem
import math
import warnings

_log2 = math.log(2)
_log2val = math.log(2)


foldername='ADMETModels'
smi_list = ['CCOC=N']

def LoadPickledModels(pklfiles):
    keywordtomodel={}
    keywordtomodeltype={}
    for pklfile in pklfiles:
        cf = joblib.load(pklfile)
        if 'AMES' in pklfile:
            keyword='AMES'
            modeltype='classification'
        elif 'caco2' in pklfile:
            keyword='caco2'
            modeltype='regression'
        elif 'logD' in pklfile:
            keyword='logD'
            modeltype='regression'
        elif 'VD' in pklfile:
            keyword='VD'
            modeltype='regression'
        elif 'T' in pklfile and 'HHT' not in pklfile:
            keyword='T'
            modeltype='regression'
        elif 'logS' in pklfile:
            keyword='logS'
            modeltype='regression'
        elif 'CL' in pklfile:
            keyword='CL'
            modeltype='regression'
        elif 'LD50' in pklfile:
            keyword='LD50'
            modeltype='regression'
        elif 'PPB' in pklfile:
            warnings.warn("Warning, PPB labels not known for desciptors computation")
            keyword='PPB'
            modeltype='regression'
            continue
        elif '2c19sub' in pklfile:
            keyword='CYP2C19-substrate'
            modeltype='classification'
        elif 'PGP_substrate_SVC_ecfp4_model' in pklfile:
            keyword='Pgp-substrate'
            modeltype='classification'
        elif 'PGP_inhibitor_SVC_ecfp4_model' in pklfile:
            keyword='Pgp-inhibitor'
            modeltype='classification'
        elif 'HHT' in pklfile:
            keyword='HHT'
            modeltype='classification'
        elif 'CYP_inhibitor_3A4_SVC_ecfp4_model' in pklfile:
            keyword='CYP3A4-inhibitor'
            modeltype='classification'
        elif 'CYP_inhibitor_2C9_SVC_ecfp4_model' in pklfile:
            keyword='CYP2C9-inhibitor'
            modeltype='classification'
        elif 'CYP_inhibitor_2C19_SVC_ecfp2_model' in pklfile:
            keyword='CYP2C19-inhibitor'
            modeltype='classification'
        elif 'CYP_inhibitor_1A2_SVC_ecfp4_model' in pklfile:
            keyword='CYP1A2-inhibitor'
            modeltype='classification'
        elif 'BBB_RF_ecfp2_model' in pklfile:
            keyword='BBB'
            modeltype='classification'
        elif 'SkinSen_9' in pklfile:
            keyword='SkinSen'
            modeltype='classification'
        elif 'HIA_9' in pklfile:
            keyword='HIA'
            modeltype='classification'
        elif 'F30_9' in pklfile:
            keyword='F30' 
            modeltype='classification'
        elif 'hERG_Model' in pklfile:
            keyword='hERG'
            modeltype='classification'
        elif 'F-20_9' in pklfile:
            keyword='F20'
            modeltype='classification'
        elif '3a4sub' in pklfile:
            keyword='CYP3A4-substrate'
            modeltype='classification'
        elif 'CYP2D6-substrate_9':
            keyword='CYP2D6-substrate'
            modeltype='classification'
        elif 'CYP2D6-inhibitor_9' in pklfile:
            keyword='CYP2D6-inhibitor'
            modeltype='classification'
        elif 'CYP2C9-substrate_9' in pklfile:
            keyword='CYP2C9-substrate'
            modeltype='classification'
        elif '1a2sub' in pklfile:
            keyword='CYP1A2-substrate'
            modeltype='classification'

        else:
            raise ValueError('input model unexpected and not supported!')
        keywordtomodel[keyword]=cf
        keywordtomodeltype[keyword]=modeltype
    return keywordtomodel,keywordtomodeltype


def BertzCT(mol, cutoff = 100, dMat = None, forceDMat = 1):
  """ A topological index meant to quantify "complexity" of molecules.
     Consists of a sum of two terms, one representing the complexity
     of the bonding, the other representing the complexity of the
     distribution of heteroatoms.
     From S. H. Bertz, J. Am. Chem. Soc., vol 103, 3599-3601 (1981)
     "cutoff" is an integer value used to limit the computational
     expense.  A cutoff value tells the program to consider vertices
     topologically identical if their distance vectors (sets of
     distances to all other vertices) are equal out to the "cutoff"th
     nearest-neighbor.
     **NOTE**  The original implementation had the following comment:
         > this implementation treats aromatic rings as the
         > corresponding Kekule structure with alternating bonds,
         > for purposes of counting "connections".
       Upon further thought, this is the WRONG thing to do.  It
        results in the possibility of a molecule giving two different
        CT values depending on the kekulization.  For example, in the
        old implementation, these two SMILES:
           CC2=CN=C1C3=C(C(C)=C(C=N3)C)C=CC1=C2C
           CC3=CN=C2C1=NC=C(C)C(C)=C1C=CC2=C3C
        which correspond to differentk kekule forms, yield different
        values.
       The new implementation uses consistent (aromatic) bond orders
        for aromatic bonds.
       THIS MEANS THAT THIS IMPLEMENTATION IS NOT BACKWARDS COMPATIBLE.
       Any molecule containing aromatic rings will yield different
       values with this implementation.  The new behavior is the correct
       one, so we're going to live with the breakage.
        
     **NOTE** this barfs if the molecule contains a second (or
       nth) fragment that is one atom.
  """
  atomTypeDict = {}
  connectionDict = {}
  numAtoms = mol.GetNumAtoms()
  if forceDMat or dMat is None:
    if forceDMat:
      # nope, gotta calculate one
      dMat = Chem.GetDistanceMatrix(mol,useBO=0,useAtomWts=0,force=1)
      mol._adjMat = dMat
    else:
      try:
        dMat = mol._adjMat
      except AttributeError:
        dMat = Chem.GetDistanceMatrix(mol,useBO=0,useAtomWts=0,force=1)
        mol._adjMat = dMat

  if numAtoms < 2:
    return 0

  bondDict, neighborList, vdList = _CreateBondDictEtc(mol, numAtoms)
  symmetryClasses = _AssignSymmetryClasses(mol, vdList, dMat, forceDMat, numAtoms, cutoff)
  #print 'Symmm Classes:',symmetryClasses
  for atomIdx in range(numAtoms):
    hingeAtomNumber = mol.GetAtomWithIdx(atomIdx).GetAtomicNum()
    atomTypeDict[hingeAtomNumber] = atomTypeDict.get(hingeAtomNumber,0)+1

    hingeAtomClass = symmetryClasses[atomIdx]
    numNeighbors = vdList[atomIdx]
    for i in range(numNeighbors):
      neighbor_iIdx = neighborList[atomIdx][i]
      NiClass = symmetryClasses[neighbor_iIdx]
      bond_i_order = _LookUpBondOrder(atomIdx, neighbor_iIdx, bondDict)
      #print '\t',atomIdx,i,hingeAtomClass,NiClass,bond_i_order
      if (bond_i_order > 1) and (neighbor_iIdx > atomIdx):
        numConnections = bond_i_order*(bond_i_order - 1)/2
        connectionKey = (min(hingeAtomClass, NiClass), max(hingeAtomClass, NiClass))
        connectionDict[connectionKey] = connectionDict.get(connectionKey,0)+numConnections

      for j in range(i+1, numNeighbors):
        neighbor_jIdx = neighborList[atomIdx][j]
        NjClass = symmetryClasses[neighbor_jIdx]
        bond_j_order = _LookUpBondOrder(atomIdx, neighbor_jIdx, bondDict)
        numConnections = bond_i_order*bond_j_order
        connectionKey = (min(NiClass, NjClass), hingeAtomClass, max(NiClass, NjClass))
        connectionDict[connectionKey] = connectionDict.get(connectionKey,0)+numConnections

  if not connectionDict:
    connectionDict = {'a':1}

  ks = connectionDict.keys()
  ks.sort()
  return _CalculateEntropies(connectionDict, atomTypeDict, numAtoms)

def _CreateBondDictEtc(mol, numAtoms):
  """ _Internal Use Only_
     Used by BertzCT
  """
  bondDict = {}
  nList = [None]*numAtoms
  vdList = [0]*numAtoms
  for aBond in mol.GetBonds():
    atom1=aBond.GetBeginAtomIdx()
    atom2=aBond.GetEndAtomIdx()
    if atom1>atom2: atom2,atom1=atom1,atom2
    if not aBond.GetIsAromatic():
      bondDict[(atom1,atom2)] = aBond.GetBondType()
    else:
      # mark Kekulized systems as aromatic
      bondDict[(atom1,atom2)] = Chem.BondType.AROMATIC
    if nList[atom1] is None:
      nList[atom1] = [atom2]
    elif atom2 not in nList[atom1]:
      nList[atom1].append(atom2)
    if nList[atom2] is None:
      nList[atom2] = [atom1]
    elif atom1 not in nList[atom2]:
      nList[atom2].append(atom1)

  for i,element in enumerate(nList):
    try:
      element.sort()
      vdList[i] = len(element)
    except:
      vdList[i] = 0
  return bondDict, nList, vdList

def _AssignSymmetryClasses(mol, vdList, bdMat, forceBDMat, numAtoms, cutoff):
  """
     Used by BertzCT
     vdList: the number of neighbors each atom has
     bdMat: "balaban" distance matrix
     
  """
  if forceBDMat:
    bdMat = Chem.GetDistanceMatrix(mol,useBO=1,useAtomWts=0,force=1,
                                   prefix="Balaban")
    mol._balabanMat = bdMat

  atomIdx = 0
  keysSeen = []
  symList = [0]*numAtoms
  for i in range(numAtoms):
    tmpList = bdMat[i].tolist()
    tmpList.sort()
    theKey = tuple(['%.4f'%x for x in tmpList[:cutoff]])
    try:
      idx = keysSeen.index(theKey)
    except ValueError:
      idx = len(keysSeen)
      keysSeen.append(theKey)
    symList[i] = idx+1
  return tuple(symList)

def PyInfoEntropy(results):
  """ Calculates the informational entropy of a set of results.

  **Arguments**

    results is a 1D Numeric array containing the number of times a
    given set hits each possible result.
    For example, if a function has 3 possible results, and the
      variable in question hits them 5, 6 and 1 times each,
      results would be [5,6,1]

  **Returns**

    the informational entropy

  """
  nInstances = float(sum(results))
  if nInstances == 0:
    # to return zero or one... that is the question
    return 0
  probs = results / nInstances

  # -------
  #  NOTE: this is a little hack to allow the use of Numeric
  #   functionality to calculate the informational entropy.
  #    The problem is that the system log function pitches a fit
  #    when you call log(0.0).  We are perfectly happy with that
  #    returning *anything* because we're gonna mutiply by 0 anyway.

  # Here's the risky (but marginally faster way to do it:
  #    add a small number to probs and hope it doesn't screw
  #    things up too much.
  # t = probs+1e-10

  # Here's a perfectly safe approach that's a little bit more obfuscated
  #  and a tiny bit slower
  t = np.choose(np.greater(probs, 0.0), (1, probs))
  return sum(-probs * np.log(t) / _log2)



def _CalculateEntropies(connectionDict, atomTypeDict, numAtoms):
  """
     Used by BertzCT
  """
  connectionList = connectionDict.values()
  totConnections = sum(connectionList)
  connectionIE = totConnections*(PyInfoEntropy(np.array(connectionList)) + math.log(totConnections)/_log2val)
  atomTypeList = atomTypeDict.values()
  atomTypeIE = numAtoms*PyInfoEntropy(np.array(atomTypeList))
  return atomTypeIE + connectionIE


def _LookUpBondOrder(atom1Id, atom2Id, bondDic):
  """
     Used by BertzCT
  """
  if atom1Id < atom2Id:
    theKey = (atom1Id,atom2Id)
  else:
    theKey = (atom2Id,atom1Id)
  tmp = bondDic[theKey]
  if tmp == Chem.BondType.AROMATIC:
    tmp = 1.5
  else:
    tmp = float(tmp)
  #tmp = int(tmp)
  return tmp

def ComputeDescriptor(pklfile,smi,keyword):
    mol=Pymolecule.PyMolecule()
    mol.ReadMolFromSmile(smi)
    rdkitmol=mol.mol
    ncarb=Pymolecule.constitution.CalculateCarbonNumber(rdkitmol)
    IC0=Pymolecule.basak.CalculateBasakIC0(rdkitmol)
    bcutpstuff=Pymolecule.bcut.CalculateBurdenPolarizability(rdkitmol)
    bcutp1=bcutpstuff['bcutp1']
    bcutvstuff=Pymolecule.bcut.CalculateBurdenVDW(rdkitmol)
    bcutv10=bcutvstuff['bcutv11']
    GMTIV=Pymolecule.topology.CalculateGutmanTopo(rdkitmol)
    nsulph=Pymolecule.constitution.CalculateSulfurNumber(rdkitmol)
    CIC6=Pymolecule.basak.CalculateBasakCIC6(rdkitmol)
    bcutmstuff=Pymolecule.bcut.CalculateBurdenMass(rdkitmol)
    bcutm12=bcutmstuff['bcutm12']
    estatestuff=Pymolecule.estate.GetEstate(rdkitmol)
    S34=estatestuff['S34']
    bcutp8=bcutpstuff['bcutp8']
    mol_des = Pymolecule.moe.GetMOE(rdkitmol)
    mol_mol_interaction1 = PyInteraction.CalculateInteraction1(mol_des,mol_des) 
    slogPVSA2=mol_mol_interaction1['slogPVSA2']
    chg_molecular_descriptor = mol.GetCharge()
    QNmin=chg_molecular_descriptor['QNmin']
    LogP2=Pymolecule.molproperty.CalculateMolLogP2(rdkitmol)
    bcutm1=bcutmstuff['bcutm1']
    Estatestuff=Pymolecule.moe.CalculateEstateVSA(rdkitmol)
    EstateVSA9=Estatestuff['EstateVSA9']
    slogPVSA1=mol_mol_interaction1['slogPVSA1']
    Hatov=Pymolecule.topology.CalculateHarmonicTopoIndex(rdkitmol)
    J=Pymolecule.topology.CalculateBalaban(rdkitmol)
    AW=Pymolecule.topology.CalculateMeanWeiner(rdkitmol)
    S7=estatestuff['S7']
    conn_molecular_descriptor = Pymolecule.connectivity.GetConnectivity(rdkitmol)
    dchi0=conn_molecular_descriptor['dchi0']
    SMRVSAstuff=Pymolecule.moe.CalculateSMRVSA(rdkitmol)
    MRVSA1=SMRVSAstuff['MRVSA1']
    LogP=Pymolecule.molproperty.CalculateMolLogP(rdkitmol)
    Tpc=chg_molecular_descriptor['Tpc']
    PEOEVSAstuff=Pymolecule.moe.CalculatePEOEVSA(rdkitmol)
    PEOEVSA0=PEOEVSAstuff['PEOEVSA0']
    Tnc=chg_molecular_descriptor['Tnc']
    S13=estatestuff['S13']
    TPSAstuff=Pymolecule.moe.CalculateTPSA(rdkitmol)
    TPSA=TPSAstuff['MTPSA']
    QHss=chg_molecular_descriptor['QHss']
    ndonr=Pymolecule.constitution.CalculateHdonorNumber(rdkitmol)
    MATSestuff=Pymolecule.moran.CalculateMoranAutoElectronegativity(rdkitmol)
    MATSe5=MATSestuff['MATSe5']
    PEOEVSA9=PEOEVSAstuff['PEOEVSA9']
    EstateVSA7=Estatestuff['EstateVSA7']
    EstateVSA0=Estatestuff['EstateVSA0']
    Chiv4=conn_molecular_descriptor['Chiv4']
    S28=estatestuff['S28']
    QOmax=chg_molecular_descriptor['QOmax']
    bcutp2=bcutpstuff['bcutp2']
    EstateVSA4=Estatestuff['EstateVSA4']
    MATSe1=MATSestuff['MATSe1']
    PC6=Pymolecule.constitution.CalculatePath6(rdkitmol)
    S24=estatestuff['S24']
    CIC0=Pymolecule.basak.CalculateBasakCIC0(rdkitmol)
    QCmax=chg_molecular_descriptor['QCmax']
    QCss=chg_molecular_descriptor['QCss']
    Geto=Pymolecule.topology.CalculateGeometricTopoIndex(rdkitmol)
    Getov=topology.CalculateGeometricTopovIndex(rdkitmol)
    bcutm11=bcutmstuff['bcutm11']
    CIC2=Pymolecule.basak.CalculateBasakCIC2(rdkitmol)
    PEOEVSA5=PEOEVSAstuff['PEOEVSA5']
    Hy=Pymolecule.molproperty.CalculateHydrophilicityFactor(rdkitmol)
    SPP=chg_molecular_descriptor['SPP']
    S36=estatestuff['S36']
    S9=estatestuff['S9']
    S16=estatestuff['S16']
    MRVSA4=SMRVSAstuff['MRVSA4']
    QOmin=chg_molecular_descriptor['QOmin']
    GMTIV=Pymolecule.topology.CalculateGutmanTopo(rdkitmol)
    UI=Pymolecule.molproperty.CalculateUnsaturationIndex(rdkitmol)
    MATSpstuff=Pymolecule.moran.CalculateMoranAutoPolarizability(rdkitmol)
    MATSp1=MATSpstuff['MATSp1']
    MATSmstuff=Pymolecule.moran.CalculateMoranAutoMass(rdkitmol)
    MATSm2=MATSmstuff['MATSm2']
    S12=estatestuff['S12']
    dchi3=conn_molecular_descriptor['dchi3']
    IDE=topology.CalculateDistanceEqualityMeanInf(rdkitmol)
    PEOEVSA7=PEOEVSAstuff['PEOEVSA7']
    bcutm9=bcutmstuff['bcutm9']
    SIC1=Pymolecule.basak.CalculateBasakSIC1(rdkitmol)
    MRVSA6=SMRVSAstuff['MRVSA6']
    IC1=Pymolecule.basak.CalculateBasakIC1(rdkitmol)
    QNmax=chg_molecular_descriptor['QNmax']
    PEOEVSA6=PEOEVSAstuff['PEOEVSA6']
    MATSe4=MATSestuff['MATSe4']
    VSAEstatestuff=Pymolecule.moe.CalculateVSAEstate(rdkitmol)
    VSAEstate8=VSAEstatestuff['VSAEstate8']
    EstateVSA3=Estatestuff['EstateVSA3']
    MRVSA5=SMRVSAstuff['MRVSA5']
    MRVSA9=SMRVSAstuff['MRVSA9']
    S19=estatestuff['S19']
    MATSvstuff=Pymolecule.moran.CalculateMoranAutoVolume(rdkitmol)
    MATSv2=MATSvstuff['MATSv2']
    S17=estatestuff['S17']
    ndb=Pymolecule.constitution.CalculateDoubleBondNumber(rdkitmol)
    AWeight=Pymolecule.constitution.CalculateAverageMolWeight(rdkitmol)
    S30=estatestuff['S30']
    MATSv5=MATSvstuff['MATSv5']
    Gravto=topology.CalculateGravitationalTopoIndex(rdkitmol)
    Chiv3c=conn_molecular_descriptor['Chiv3c']
    knotp=conn_molecular_descriptor['knotp']
    bcutp3=bcutpstuff['bcutp3']
    bcutp11=bcutpstuff['bcutp11']
    VSAEstate7=VSAEstatestuff['VSAEstate7']
    MATSp4=MATSpstuff['MATSp4']
    bcutm4=bcutmstuff['bcutm4']
    nring=Pymolecule.constitution.CalculateRingNumber(rdkitmol)
    bcutestuff=Pymolecule.bcut.CalculateBurdenElectronegativity(rdkitmol)
    bcute1=bcutestuff['bcute1']
    MATSp6=MATSpstuff['MATSp6']
    naro=Pymolecule.constitution.CalculateAromaticBondNumber(rdkitmol)
    CIC3=Pymolecule.basak.CalculateBasakCIC3(rdkitmol)
    TIAC=topology.CalculateAtomCompTotalInf(rdkitmol)
    MATSm1=MATSmstuff['MATSm1']
    slogPVSA7=mol_mol_interaction1['slogPVSA7']
    bcutm8=bcutmstuff['bcutm8']
    IDET=topology.CalculateDistanceEqualityTotalInf(rdkitmol)
    Chi10=conn_molecular_descriptor['Chi10']
    Weight=Pymolecule.constitution.CalculateMolWeight(rdkitmol)
    Rnc=chg_molecular_descriptor['Rnc']
    naccr=Pymolecule.constitution.CalculateHacceptorNumber(rdkitmol)
    bcutp5=bcutpstuff['bcutp5']
    bcutm2=bcutmstuff['bcutm2']
    Chiv1=conn_molecular_descriptor['Chiv1']
    bcutm3=bcutmstuff['bcutm3']
    Chiv9=conn_molecular_descriptor['Chiv9']
    S32=estatestuff['S32']
    nsb=Pymolecule.constitution.CalculateSingleBondNumber(rdkitmol)
    nhet=Pymolecule.constitution.CalculateHeteroNumber(rdkitmol)
    MATSe3=MATSestuff['MATSe3']
    S53=estatestuff['S53']
    PEOEVSA12=PEOEVSAstuff['PEOEVSA12']
    MATSm3=MATSmstuff['MATSm3']
    MATSm4=MATSmstuff['MATSm4']
    MATSm6=MATSmstuff['MATSm6']
    Chi4c=conn_molecular_descriptor['Chi4c']
    Chiv3=conn_molecular_descriptor['Chiv3']
    Chiv4=conn_molecular_descriptor['Chiv4']
    Chiv4c=conn_molecular_descriptor['Chiv4c']
    Chiv4pc=conn_molecular_descriptor['Chiv4pc']
    Ds=estate.CalculateDiffMaxMinEState(rdkitmol)
    QNss=chg_molecular_descriptor['QNss']
    Qmax=chg_molecular_descriptor['Qmax']
    S46=estatestuff['S46']
    ATSmstuff=moreaubroto.CalculateMoreauBrotoAutoMass(rdkitmol)
    ATSm1=ATSmstuff['ATSm1']
    ATSm2=ATSmstuff['ATSm2']
    ATSm3=ATSmstuff['ATSm3']
    ATSm4=ATSmstuff['ATSm4']
    ATSm6=ATSmstuff['ATSm6']
    Smin=estate.CalculateMinEState(rdkitmol)
    Smaxstuff=estate.CalculateMaxAtomTypeEState(rdkitmol)
    Smax45=Smaxstuff['Smax45']
    Sminstuff=estate.CalculateMinAtomTypeEState(rdkitmol)
    Smin45=Sminstuff['Smin45']
    nphos=Pymolecule.constitution.CalculatePhosphorNumber(rdkitmol)
    slogPVSA11=mol_mol_interaction1['slogPVSA11']
    nnitro=Pymolecule.constitution.CalculateNitrogenNumber(rdkitmol)
    nhev=Pymolecule.constitution.CalculateHeavyAtomNumber(rdkitmol)
    Arto=Pymolecule.topology.CalculateArithmeticTopoIndex(rdkitmol)
    temp = BertzCT(rdkitmol)
    BertzCTValue=np.log10(temp)
    MZM2=Pymolecule.topology.CalculateMZagreb2(rdkitmol)
    MZM1=Pymolecule.topology.CalculateMZagreb1(rdkitmol)
    kappastuff=mol.GetKappa()
    phi=kappastuff['phi']
    kappa3=kappastuff['kappa3']
    kappa2=kappastuff['kappa2']
    kappam3=kappastuff['kappam3']
    MATSv3=MATSvstuff['MATSv3']
    MATSv7=MATSvstuff['MATSv7']
    MATSv6=MATSvstuff['MATSv6']
    MATSm5=MATSmstuff['MATSm5']
    MATSe6=MATSestuff['MATSe6']
    MATSe2=MATSestuff['MATSe2']
    MATSp3=MATSpstuff['MATSp3']
    MATSp2=MATSpstuff['MATSp2']
    QOss=chg_molecular_descriptor['QOss']
    LDI=chg_molecular_descriptor['LDI']
    Qass=chg_molecular_descriptor['Qass']
    QHmax=chg_molecular_descriptor['QHmax']
    Rpc=chg_molecular_descriptor['Rpc']
    Qmin=chg_molecular_descriptor['Qmin']
    Mnc=chg_molecular_descriptor['Mnc']
    EstateVSA5=Estatestuff['EstateVSA5']
    EstateVSA6=Estatestuff['EstateVSA6']
    EstateVSA1=Estatestuff['EstateVSA1']
    EstateVSA2=Estatestuff['EstateVSA2']
    PEOEVSA11=PEOEVSAstuff['PEOEVSA11']
    PEOEVSA2=PEOEVSAstuff['PEOEVSA2']
    PEOEVSA1=PEOEVSAstuff['PEOEVSA1']
    PEOEVSA8=PEOEVSAstuff['PEOEVSA8']
    MRVSA3=SMRVSAstuff['MRVSA3']
    MRVSA2=SMRVSAstuff['MRVSA2']
    slogPVSA4=mol_mol_interaction1['slogPVSA4']
    slogPVSA5=mol_mol_interaction1['slogPVSA5']
    nta=Pymolecule.constitution.CalculateAllAtomNumber(rdkitmol)
    MATSv1=MATSvstuff['MATSv1']

    # HHT labels not in docs
    # PPB Labels will not load
    # labels for LD50 not in docs
    # druglikeness not in documentation
    # what are the models in old folders?
    if keyword=='AMES':
        res = mol.GetFingerprint(FPName='MACCS')
        des_list=res[-1]
    elif keyword=='CYP1A2-substrate':
        fp = tuple(AllChem.GetMorganFingerprintAsBitVect(rdkitmol, 4, nBits=1024))
        des_list=fp

    elif keyword=='CYP2C9-substrate':
        fp = tuple(AllChem.GetMorganFingerprintAsBitVect(rdkitmol, 4, nBits=1024))
        des_list=fp

    elif keyword=='CYP2D6-inhibitor':
        fp = tuple(AllChem.GetMorganFingerprintAsBitVect(rdkitmol, 4, nBits=1024))
        des_list=fp

    elif keyword=='CYP2D6-substrate': # 9 models here not clear in docs
        fp = tuple(AllChem.GetMorganFingerprintAsBitVect(rdkitmol, 4, nBits=1024))
        des_list=fp

    elif keyword=='CYP3A4-substrate':
        fp = tuple(AllChem.GetMorganFingerprintAsBitVect(rdkitmol, 4, nBits=1024))
        des_list=fp

    elif keyword=='F20':
        res = mol.GetFingerprint(FPName='MACCS')
        des_list=res[-1]

    elif keyword=='hERG': # descs not in docs
        des_list=[ndb, nsb, ncarb, nsulph, naro, ndonr, nhev, naccr, nta, nring, PC6, GMTIV, AW, Geto, BertzCTValue, J, MZM2, phi, kappa2, MATSv1, MATSv5, MATSe4, MATSe5, MATSe6, TPSA, Hy, LogP, LogP2, UI, QOss, SPP, LDI, Qass, QOmin, QNmax, Qmin, Mnc, EstateVSA7, EstateVSA0, EstateVSA3, PEOEVSA0, PEOEVSA6, MRVSA5, MRVSA4, MRVSA3, MRVSA6, slogPVSA1]
    elif keyword=='F30':
        fp = tuple(AllChem.GetMorganFingerprintAsBitVect(rdkitmol, 6, nBits=2048))
        des_list=fp
    elif keyword=='HIA': # not clear which ones is MACCS, why are there 9 and what do they correspond to in docs
        res = mol.GetFingerprint(FPName='MACCS')
        des_list=res[-1]
    elif keyword=='SkinSen': # not clear which ones is MACCS, why are there 9 and what do they correspond to in docs
        res = mol.GetFingerprint(FPName='MACCS')
        des_list=res[-1]
    elif keyword=='BBB':
        fp = tuple(AllChem.GetMorganFingerprintAsBitVect(rdkitmol, 2, nBits=2048))
        des_list=fp
    elif keyword=='CYP1A2-inhibitor':
        fp = tuple(AllChem.GetMorganFingerprintAsBitVect(rdkitmol, 4, nBits=2048))
        des_list=fp
    elif keyword=='CYP2C19-inhibitor':
        fp = tuple(AllChem.GetMorganFingerprintAsBitVect(rdkitmol, 2, nBits=2048))
        des_list=fp
    elif keyword=='CYP2C9-inhibitor':
        fp = tuple(AllChem.GetMorganFingerprintAsBitVect(rdkitmol, 4, nBits=2048))
        des_list=fp
    elif keyword=='CYP3A4-inhibitor':
        fp = tuple(AllChem.GetMorganFingerprintAsBitVect(rdkitmol, 4, nBits=2048))
        des_list=fp
    elif keyword=='CYP2C19-substrate':
        res = mol.GetFingerprint(FPName='ECFP2')
        des_list=res[0]
    elif keyword=='Pgp-substrate':
        fp = tuple(AllChem.GetMorganFingerprintAsBitVect(rdkitmol, 4, nBits=2048))
        des_list=fp
    elif keyword=='Pgp-inhibitor':
        fp = tuple(AllChem.GetMorganFingerprintAsBitVect(rdkitmol, 4, nBits=2048))
        des_list=fp
    elif keyword=='HHT': # rdkit has issues with BertzCT entropy calculation so added python code above
        des_list=[ndb, nsb, nnitro, naro, ndonr, nhet, nhev, nring, PC6, GMTIV, Geto, IDE, Arto, Hatov, BertzCTValue, Getov, J, MZM2, MZM1, phi, kappa3, kappa2, kappam3, MATSp4, MATSp6, MATSv3, MATSv2, MATSv5, MATSv7, MATSv6, MATSm4, MATSm5, MATSm6, MATSm2, MATSm3, MATSe4, MATSe5, MATSe6, MATSe1, MATSe2, MATSe3, MATSp3, MATSp2, TPSA, LogP, LogP2, UI, QNmin, QOss, QHss, SPP, LDI, Qass, QCmax, QOmax, Tpc, QOmin, QCss, QHmax, Rnc, Rpc, Qmin, Mnc, EstateVSA9, EstateVSA4, EstateVSA5, EstateVSA6, EstateVSA7, EstateVSA0, EstateVSA1, EstateVSA2, EstateVSA3, PEOEVSA11, PEOEVSA2, PEOEVSA1, PEOEVSA7, PEOEVSA6, PEOEVSA5, MRVSA5, MRVSA4, PEOEVSA9, PEOEVSA8, MRVSA3, MRVSA2, MRVSA9, MRVSA6, slogPVSA2, slogPVSA4, slogPVSA5]
    elif keyword=='caco2': 
        des_list=[ncarb,IC0,bcutp1,bcutv10,GMTIV,nsulph,CIC6,bcutm12,S34,bcutp8,slogPVSA2,QNmin,LogP2,bcutm1,EstateVSA9,slogPVSA1,Hatov,J,AW,S7,dchi0,MRVSA1,LogP,Tpc,PEOEVSA0,Tnc,S13,TPSA,QHss,ndonr]
    elif keyword=='logD': 
        des_list=[MATSe5,PEOEVSA9,EstateVSA7,S13,EstateVSA0,Chiv4,S28,AW,QOmax,bcutp2,EstateVSA4,MATSe1,PC6,Hatov,S24,CIC0,QCmax,QCss,Geto,TPSA,Getov,bcutm11,CIC2,J,S34,PEOEVSA5,Hy,SPP,S36,S9,S16,MRVSA4,LogP2,QOmin,LogP]
    elif keyword=='VD': 
        des_list=[GMTIV,UI,MATSe1,MATSp1,Chiv4,MATSm2,S12,dchi3,IDE,PEOEVSA7,bcutp1,bcutm9,SIC1,MRVSA6,IC1,QNmax,CIC0,PEOEVSA6,MATSe4,VSAEstate8,Geto,EstateVSA3,MRVSA5,LogP2,Tnc,S7,SPP,QOmin,EstateVSA7,LogP,QNmin,MRVSA9,S19,MATSv2,nsulph,S17,S9,ndb,AWeight,QCss,EstateVSA9,Hy,S16,IC0,S30]
    elif keyword=='T': # docs say 40 descriptors but there is 50
        des_list=[MATSv5,Gravto,Chiv3c,PEOEVSA7,knotp,bcutp3,bcutm9,EstateVSA3,MATSp1,bcutp11,VSAEstate7,IC0,UI,Geto,QOmin,CIC0,dchi3,MATSp4,bcutm4,Hatov,MATSe4,CIC6,Chiv4,EstateVSA9,MATSv2,nring,bcute1,VSAEstate8,MRVSA9,PEOEVSA6,SIC1,bcutp8,MATSp6,QCss,J,IDE,CIC2,Hy,MRVSA6,naro,SPP,EstateVSA7,bcutv10,S12,LogP2,bcutp2,CIC3,S17,LogP,bcutp1]

    elif keyword=='logS':
        des_list=[MATSm2,TIAC,GMTIV,IC1,naro,MATSm1,nsulph,Tpc,slogPVSA7,bcutp1,AWeight,Tnc,MRVSA9,bcutp3,IC0,AW,Hy,bcutv10,MRVSA6,PC6,bcutm1,bcutm8,slogPVSA1,IDET,Chi10,TPSA,Weight,Rnc,naccr,bcutp5,Chiv4,bcutm2,Chiv1,bcutm3,Chiv9,ncarb,bcutm4,PEOEVSA5,LogP2,LogP]
    elif keyword=='CL':
        des_list=[nsulph,VSAEstate8,QNmin,IDET,ndb,slogPVSA2,MATSv5,S32,QCss,bcutm4,S9,bcutp8,Tnc,nsb,Geto,bcutp11,S7,MATSm2,GMTIV,nhet,MATSe1,CIC0,bcutp3,Gravto,EstateVSA9,MATSe3,MATSe5,UI,S53,J,bcute1,MRVSA9,PEOEVSA0,MATSv2,IDE,AWeight,IC0,S16,bcutp1,PEOEVSA12]
    elif keyword=='LD50': # labels are not in docs on website but found in github
        des_list=[ATSm1,ATSm2,ATSm3,ATSm4,ATSm6,AWeight,Chi4c,Chiv3,Chiv4,Chiv4c,Chiv4pc,Ds,Gravto,IC0,IC1,MRVSA9,QCmax,QNss,QOmin,Qmax,S46,Smax45,Smin,Smin45,VSAEstate7,Weight,bcutm1,bcutm2,bcutp1,nhet,nphos,slogPVSA11]
    elif keyword=='PPB':
        des_list=[] # labels not on docs website and also cant access using scikit version in label folder in Github
    return des_list


def ComputeDescriptorsForAllModelsAndSmiles(keywordtomodel,smi_list):
    allsmiles_allmodels_des_list={}
    for smi in smi_list:
        allsmiles_allmodels_des_list[smi]={}
        for keyword,model in keywordtomodel.items():
            des_list=ComputeDescriptor(model,smi,keyword)
            allsmiles_allmodels_des_list[smi][keyword]=des_list

    return allsmiles_allmodels_des_list


def RunAllModelsForAllSmiles(allsmiles_allmodels_des_list,keywordtomodel,keywordtomodeltype):
    smitokeywordtomodelresults={}
    for smi,keywordtodes_list in allsmiles_allmodels_des_list.items():
        keywordtomodelresults={}
        for keyword,des_list in keywordtodes_list.items():
            model=keywordtomodel[keyword]
            modeltype=keywordtomodeltype[keyword]
            y_predict_proba,y_predict_regr=RunModel(model,des_list,smi,modeltype,printout=True)
            if modeltype=='classification':
                keywordtomodelresults[keyword]=y_predict_proba
            elif modeltype=='regression':
                keywordtomodelresults[keyword]=y_predict_regr

        smitokeywordtomodelresults[smi]=keywordtomodelresults
    return smitokeywordtomodelresults


def RunModel(model,des_list,smi,modeltype,printout=False):
    y_predict_label=None
    y_predict_proba=None
    y_predict_regr=None
    if modeltype=='classification':
        y_predict_proba = model.predict_proba(des_list)[0][0]
        if printout==True:
            print('Input molecule '+smi) 
            print('#'*10+'Results probabilities'+'#'*10)
            print(y_predict_proba)
    elif modeltype=='regression':
        y_predict_regr=model.predict(des_list)[0]
        if printout==True:
            print('Input molecule '+smi) 
            print('#'*10+'Results regression'+'#'*10)
            print(y_predict_regr)

    return y_predict_proba,y_predict_regr


def GrabModels(foldername):
    pklfiles=[]
    os.chdir(foldername)
    files=os.listdir(os.getcwd())
    for f in files:
        fsplit=f.split('.')
        end=fsplit[1]
        if end=='pkl':
            pklfiles.append(f)

    return pklfiles


pklfiles=GrabModels(foldername)
if not sys.warnoptions:
    warnings.simplefilter("ignore")
keywordtomodel,keywordtomodeltype=LoadPickledModels(pklfiles)
allsmiles_allmodels_des_list=ComputeDescriptorsForAllModelsAndSmiles(keywordtomodel,smi_list)
smitokeywordtomodelresults=RunAllModelsForAllSmiles(allsmiles_allmodels_des_list,keywordtomodel,keywordtomodeltype)
print('smitokeywordtomodelresults',smitokeywordtomodelresults)
