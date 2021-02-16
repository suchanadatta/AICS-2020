# topical_AP vs causal_AP (BM25, JM, Dirichlet)
# plot topical_AP vs causal_AP for each query of CAIR for all 3 retrieval models - BM25, Jelinek-Mercer and BM25

import numpy as np
import matplotlib.pyplot as plt
import sys

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
MEDIUM_SIZE = 13
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
xticks = np.linspace(0, 1, 11, endpoint=True)
yticks = np.linspace(0, 1, 11, endpoint=True)
fig_list = ['(a) BM25', '(b) LM-JelinekMercer', '(c) LM-Dirichlet']
scatter_color = ['blue', 'red', 'green']
marker = ['o', 'x', '^']

for c, file in enumerate(read_files):
    if np.ndim(file) == 1:
        read_files[c] = file[np.newaxis]
    if c >= len(read_files)-1:
        break

    if c % 2 == 0:
        merge = np.concatenate((read_files[c], read_files[c + 1]), axis=1)
        merge = merge[:, np.r_[1, 3]]
        axes[int(c/2)].scatter(merge[:, 0], merge[:, 1], c=scatter_color[int(c/2)], marker=marker[int(c/2)], s=120)
        axes[int(c/2)].set_xticks(xticks)
        axes[int(c/2)].set_yticks(yticks)

        axes[int(c/2)].set_xlabel("topical_MAP", fontsize=18, fontweight='bold')
        axes[int(c/2)].set_ylabel("causal_MAP", fontsize=18, fontweight='bold')
        axes[int(c/2)].set_title(fig_list[int(c/2)], fontsize=18, fontweight='bold')

fig.suptitle("Topical_MAP vs Causal_MAP", fontsize=24, fontweight='bold')
plt.subplots_adjust(left=0.045, right=0.965, bottom=0.55, top=0.87)
plt.show()












