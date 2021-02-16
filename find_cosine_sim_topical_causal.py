# Program to measure the similarity between two sentences using cosine similarity.
import csv, sys, nltk, os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np

# make sure the argument is good (0 = the python file, 1+ the actual argument)
# 1. collection_dump <docid \t doc_content>
# 2. topical qrel <qid \t Q0 \t docid \t relscore>
# 3. causal qrel <qid \t Q0 \t docid \t relscore>

if len(sys.argv) < 4:
    print('Needs 3 arguments...')
    exit(0)

arg_collection_dump = sys.argv[1]
arg_topical_qrel = sys.argv[2]
arg_causal_qrel = sys.argv[3]

list_query_id = []
dict_cosine_sim = {}


def make_topical_dump(qid):
    read_topical_qrel = csv.reader(open(arg_topical_qrel, "r"), delimiter="\t")
    doc_content_topical = ""
    for row_topical in read_topical_qrel:
        if row_topical[0] == qid:
            print(row_topical)
            print("yes qid matched...")
            qid = row_topical[0]
            read_collection_dump = csv.reader(open(arg_collection_dump, "r"), delimiter="\t")
            for row_collection in read_collection_dump:
                if row_topical[2] == row_collection[0]:
                    print(row_topical[2])
                    print("found in the collection...")
                    doc_content_topical = doc_content_topical + " " + row_collection[1]
                    print("content appended...")
                    # print(doc_content_topical)
    return doc_content_topical


def make_causal_dump(qid):
    read_causal_qrel = csv.reader(open(arg_causal_qrel, "r"), delimiter="\t")
    doc_content_causal = ""
    for row_causal in read_causal_qrel:
        if row_causal[0] == qid:
            print(row_causal)
            print("yes qid matched...")
            qid = row_causal[0]
            read_collection_dump = csv.reader(open(arg_collection_dump, "r"), delimiter="\t")
            for row_collection in read_collection_dump:
                if row_causal[2] == row_collection[0]:
                    print(row_causal[2])
                    print("found in coll...")
                    doc_content_causal = doc_content_causal + " " + row_collection[1]
                    print("content appended...")
                    # print(doc_content_causal)
    return doc_content_causal


def cal_cosine_similarity(topical_dump, causal_dump):
    # tokenization
    topical_list = word_tokenize(topical_dump)
    causal_list = word_tokenize(causal_dump)

    # sw contains the list of stopwords
    sw = stopwords.words('english')
    vec_topical = []
    vec_causal = []

    stemmer = nltk.stem.porter.PorterStemmer()

    # remove stop words from the string and stem words
    topical_set = {stemmer.stem(w) for w in topical_list if not w in sw}
    # print(topical_set)
    causal_set = {stemmer.stem(w) for w in causal_list if not w in sw}
    # print(causal_set)

    # form a set containing keywords of both strings
    rvector = topical_set.union(causal_set)
    for w in rvector:
        if w in topical_set:
            vec_topical.append(1)  # create a vector
        else:
            vec_topical.append(0)
        if w in causal_set:
            vec_causal.append(1)
        else:
            vec_causal.append(0)
    c = 0

    # cosine formula
    for i in range(len(rvector)):
        c += vec_topical[i] * vec_causal[i]
    cosine = c / float((sum(vec_topical) * sum(vec_causal)) ** 0.5)
    print("similarity: ", cosine, "\n==============================\n")

    return cosine


read_topical_qrel = csv.reader(open(arg_topical_qrel, "r"), delimiter="\t")
for row_topical in read_topical_qrel:
    list_query_id.append(row_topical[0])
list_query_id_uniq = set(list_query_id)
list_query_id = list(list_query_id_uniq)
list_query_id.sort()
print("QueryIDs are : ", list_query_id)

for qid in list_query_id:
    print(qid)
    topical_dump = make_topical_dump(qid)
    topical_dump = nltk.re.sub("\s\s+", " ", topical_dump)
    topical_dump = nltk.re.sub(r'\W+', ' ', topical_dump).lower()
    # print(topical_dump)
    print("received merged topical content....\n")
    causal_dump = make_causal_dump(qid)
    causal_dump = nltk.re.sub("\s\s+", " ", causal_dump)
    causal_dump = nltk.re.sub(r'\W+', ' ', causal_dump).lower()
    # print(causal_dump)
    print("received merged causal content.....\n")
    similarity = cal_cosine_similarity(topical_dump, causal_dump)
    dict_cosine_sim[qid] = similarity

print(dict_cosine_sim)

curr_path = os.getcwd()
curr_path = curr_path + '/qid_cosin_sim.out'
print('Similarities will be saved in : ' + curr_path)
out_similarity = open(curr_path, 'w')
queryid = []
cosin_sim = []

for key, value in dict_cosine_sim.items():
    queryid.append(int(key))
    out_similarity.write(key)
    out_similarity.write('\t')
    cosin_sim.append(value)
    out_similarity.write(str(value))
    out_similarity.write('\n')
out_similarity.close()

print(queryid)
print(cosin_sim)

# plotting

# read = np.genfromtxt(curr_path, dtype=np.float64, delimiter='\t', skip_header=0)
# print(read)
# print(np.ndim(read))
# if np.ndim(read) == 1:
#     read = read[np.newaxis]
# read = read[read[:, 0].argsort(kind='mergesort'), :]
# print(read)
#
# queryid = read[:, 0]
# cosin_sim = read[:, 1]
#
# xticks = np.linspace(0, 27, 28, endpoint=True)
# yticks = np.linspace(0, 1, 11, endpoint=True)
# fig, axes = plt.subplots(nrows=1, ncols=1)
#
# axes.plot(queryid, cosin_sim)
# axes.scatter(queryid, cosin_sim, c='red', marker='o')  # o, ., ^
# for c, i in enumerate(queryid):
#     cord_text = '(' + str(queryid[c]) + ', ' + str(cosin_sim[c]) + ')'
#     # axes.annotate(cord_text, (queryid[c], cosin_sim[c]), xytext=(queryid[c] + 0.1, cosin_sim[c] + 0.1), fontsize=8)
#
# axes.set_xticks(xticks)
# axes.set_yticks(yticks)
#
# # fmt = '${}$'
# axes.set_xlabel("QueryID", fontsize=16)
# axes.set_ylabel("Cosine Similarity", fontsize=16)
# axes.set_title("QueryID vs Cosine Similarity", fontsize=20)
# # fig.suptitle("QueryID vs Cosine similarity", fontsize=16)
#
# plt.show()
