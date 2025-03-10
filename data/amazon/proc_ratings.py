from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

import os
import pandas as pd
import numpy as np

ratings_proc = pd.read_csv('rating_proc.csv', index_col=0)

reader = Reader(line_format=u'rating user item', sep=',', rating_scale=(0, 5), skip_lines=1)
data = Dataset.load_from_file('rating_proc.csv', reader=reader)

# Fit SVD Decomposition
trainset, testset = train_test_split(data, test_size=0.1)
algo = SVD(random_state=0)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

n_users = ratings_proc['user_idx'].unique().shape[0]
n_items = ratings_proc['item_idx'].unique().shape[0]
rel_matrix = np.zeros((n_users, n_items))

for uidx in range(n_users):
    uid = str(uidx)
    print(f"{uid} / {n_users}")
    for iidx in range(n_items):
        iid = str(iidx)
        est = algo.predict(uid=uid, iid=iid).est
        rel_matrix[uidx, iidx] = est

np.save(os.path.join('rel_matrix.npy'), rel_matrix)

end = True