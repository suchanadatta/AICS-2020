# qid vs cosine_similarity between causal and topical qrel documents
# similarity has been calculated by 'find_cosine_sim_topical_causal.py'
# Argument : per query cosine_sim value <qid '\t' similarity>


import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print('Needs 1 argument...')
    exit(0)

qid_vs_cosine_sim_file = sys.argv[1]

read = np.genfromtxt(qid_vs_cosine_sim_file, dtype=np.float64, delimiter='\t', skip_header=0)
print(read)
print(np.ndim(read))
if np.ndim(read) == 1:
    read = read[np.newaxis]
read = read[read[:, 0].argsort(kind='mergesort'), :]
print(read)

queryid = read[:, 0]
cosin_sim = read[:, 1]

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

xticks = np.linspace(0, 27, 28, endpoint=True)
yticks = np.linspace(0, 1, 11, endpoint=True)

fig, axes = plt.subplots(nrows=1, ncols=1)
n_bins = 1
# colors = ['green', 'blue', 'lime']

plt.hist(cosin_sim, n_bins, density=True,
         histtype='bar',
         color='green',
         label='green')

plt.legend(prop={'size': 10})
plt.title('matplotlib.pyplot.hist() function Example\n\n', fontweight="bold")
plt.show()