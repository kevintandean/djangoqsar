import os
import subprocess

__author__ = 'kevintandean'

cwd = os.getcwd()
print cwd
name = 'quinidine'
subprocess.call(['java','-jar','PaDEL-Descriptor/PaDEL-Descriptor.jar','-2d','-3d','-fingerprints','-dir','../temp_smiles/'+name+'/','-file','../temp_smiles/'+name+'/result.csv'])

