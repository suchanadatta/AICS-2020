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

xticks = np.linspace(0, 25, 26, endpoint=True)
yticks = np.linspace(0, 1, 11, endpoint=True)

fig, axes = plt.subplots(nrows=1, ncols=1)

# for scatter + line
# axes.plot(queryid, cosin_sim)
# axes.scatter(queryid, cosin_sim, c='red', marker='o')  # o, ., ^

#  for bar chart
# for more colors (https://matplotlib.org/examples/color/named_colors.html)
axes.bar(queryid, cosin_sim, width=0.75, color='maroon', edgecolor='black', align='edge')

for c, i in enumerate(queryid):
    cord_text = '(' + str(queryid[c]) + ', ' + str(cosin_sim[c]) + ')'
    # axes.annotate(cord_text, (queryid[c], cosin_sim[c]), xytext=(queryid[c] + 0.1, cosin_sim[c] + 0.1), fontsize=8)

axes.set_xticks(xticks)
axes.set_yticks(yticks)

# fmt = '${}$'
axes.set_xlim(np.array([1, 26]))
axes.set_xlabel("QueryID", fontsize=20, fontweight='bold')
axes.set_ylabel("Cosine Similarity", fontsize=20, fontweight='bold')
axes.set_title("QueryID vs Cosine Similarity", fontsize=24, fontweight='bold')
# fig.suptitle("QueryID vs Cosine similarity", fontsize=16)

plt.show()