# qid vs retrieval models (BM25, JM, Dirichlet)
# plot qid vs retrieval models for each query of CAIR for all 3 retrieval models - BM25, Jelinek-Mercer and BM25
# Arguments : a. per query eval file for topical <qid '\t' map>
#             b. per query eval file for causal <qid '\t' map>

import numpy as np
import matplotlib.pyplot as plt
import os, sys

from numpy.distutils.system_info import x11_info

if len(sys.argv) < 7:
    print('Needs 6 argument...')
    exit(0)

per_query_bm25_topical = sys.argv[1]
per_query_bm25_causal = sys.argv[2]
per_query_lmjm_topical = sys.argv[3]
per_query_lmjm_causal = sys.argv[4]
per_query_lmdir_topical = sys.argv[5]
per_query_lmdir_causal = sys.argv[6]

read_bm25_topical = np.genfromtxt(per_query_bm25_topical, dtype=np.float64, delimiter='\t', skip_header=0)
read_bm25_causal = np.genfromtxt(per_query_bm25_causal, dtype=np.float64, delimiter='\t', skip_header=0)
read_lmjm_topical = np.genfromtxt(per_query_lmjm_topical, dtype=np.float64, delimiter='\t', skip_header=0)
read_lmjm_causal = np.genfromtxt(per_query_lmjm_causal, dtype=np.float64, delimiter='\t', skip_header=0)
read_lmdir_topical = np.genfromtxt(per_query_lmdir_topical, dtype=np.float64, delimiter='\t', skip_header=0)
read_lmdir_causal = np.genfromtxt(per_query_lmdir_causal, dtype=np.float64, delimiter='\t', skip_header=0)

# print(read_bm25_topical)
# print(read_bm25_causal)
# print(read_lmjm_topical)
# print(read_lmjm_causal)
# print(read_lmdir_topical)
# print(read_lmdir_causal)

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

read_files = [read_bm25_topical, read_bm25_causal, read_lmjm_topical, read_lmjm_causal, read_lmdir_topical, read_lmdir_causal]
fig, axes = plt.subplots(nrows=1, ncols=3, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [0.6]})
xticks = np.linspace(0, 25, 26, endpoint=True)
yticks = np.linspace(0, 1, 11, endpoint=True)

fig_list = ['(a) BM25', '(b) LM-JelinekMercer', '(c) LM-Dirichlet']
bar_color = ['purple', 'grey']

# Calculate optimal width (for multiple bar side by side)
width = np.min(np.diff(xticks))/2

for c, file in enumerate(read_files):
    print(c)
    if np.ndim(file) == 1:
        read_files[c] = file[np.newaxis]
    temp = file[file[:, 0].argsort(kind='mergesort'), :]
    if c < len(read_files)-1:
        temp1 = read_files[c + 1][read_files[c + 1][:, 0].argsort(kind='mergesort'), :]
    else:
        break

    print(temp)
    if c % 2 == 0:
        # for scatter+line graph
        # axes[int(c/2)].plot(temp[:, 0], temp[:, 1], label='topical')
        # axes[int(c/2)].scatter(temp[:, 0], temp[:, 1], c='red', marker='o')
        # axes[int(c/2)].plot(temp1[:, 0], temp1[:, 1], label='causal')
        # axes[int(c/2)].scatter(temp1[:, 0], temp1[:, 1], c='red', marker='o')
        # axes[int(c/2)].set_xticks(xticks)
        # axes[int(c/2)].set_yticks(yticks)

        #  for bar chart
        axes[int(c / 2)].bar(temp[:, 0], temp[:, 1], width, label='topical', edgecolor='black', align='edge')
        axes[int(c / 2)].bar(temp1[:, 0], temp1[:, 1], width, label='causal', edgecolor='black')
        axes[int(c/2)].set_xticks(xticks)
        axes[int(c/2)].set_yticks(yticks)

        for tick in axes[int(c/2)].xaxis.get_major_ticks()[0::2]:
            tick.set_pad(12)

        axes[int(c/2)].set_xlabel("QueryID", fontsize=18, fontweight='bold')
        axes[int(c/2)].set_ylabel("AP (per query)", fontsize=18, fontweight='bold')
        axes[int(c/2)].set_title(fig_list[int(c/2)], fontsize=18, fontweight='bold')
        # for scatter+line
        axes[int(c/2)].legend(loc="upper left")

fig.suptitle("QueryID vs Per_Query_AP", fontsize=24, fontweight='bold')
plt.subplots_adjust(left=0.045, right=0.965, bottom=0.55, top=0.87)

# Finding the best position for legends and putting it (for bar)
# plt.legend(loc='best')
plt.show()










