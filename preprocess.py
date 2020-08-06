import numpy as np
import re
from Bio import SeqIO
import os, sys
from os.path import join, exists, dirname, abspath, realpath


sys.path.append(dirname(abspath("__file__")))

pwd = dirname(realpath("__file__"))
arrhh = np.load(join(pwd,'Train_HHblits.npz'))
pdbs = arrhh['pdbids']
mask = np.array([50,57,58,59,60,61,62,63,64])
a = arrhh['data'][:,:,mask]
arrhh.close()


count = 0
pdbs = set([pdb.upper() for pdb in pdbs])
largepdb = []
fasta_sequences = SeqIO.parse(open("./fasta.txt"),'fasta')
with open("./residuesequences.txt", "w") as f:
for fasta in fasta_sequences:
    desc = fasta.description.split("|")
    pdbid = re.sub(r":", "-", desc[0])
    largepdb.append(pdbid)
    if pdbid in pdbs:
      count += 1
      f.write("{}\n".format(" ".join(fasta.seq)))
print(count)

largepdb = set(largepdb)
indexes = []
count = 0
for i,pdb in enumerate(pdbs):
  if pdb.upper() not in largepdb:
    count += 1
    indexes.append(i)
    print(i,pdb.upper())

a = np.delete(a,indexes,0)
np.savez_compressed('labels.npz', label = a)