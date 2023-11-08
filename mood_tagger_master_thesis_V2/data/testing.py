               








print(f'Oversampling! Treshold is set to {tolerance} and ratio to {os_ratio}')
train_annots = annots[self.train_indices]
gdw = GetKDE()
dense = gdw.fit(train_annots)
threshold = dense<tolerance #th
print((dense>0.5).sum())
print((dense<0.5).sum())
new_len = math.floor(((dense>0.5).sum() - threshold.sum())/os_ratio)
                import random
                
candidates = []
w = (1-dense)[threshold]
                #train_annots_thresh = train_annots[threshold]

from itertools import compress
dense_thresh_indices = list(compress(self.train_indices, threshold)) 
                #np.array(self.train_indices)[threshold]
for i in range(new_len):
    random_chosen = random.choices(dense_thresh_indices,weights = w)
    choice = random_chosen[0] #np.where(train_annots == random_chosen)[0][0]
    candidates.append(choice)
    self.train_indices = self.train_indices + candidates
    train_annots_new = annots[self.train_indices]
    gdw_new = GetKDE()
    dense_new = gdw_new.fit(train_annots_new)
    threshold = dense_new<tolerance
    w = (1-dense)[threshold]
    dense_thresh_indices = list(compress(self.train_indices, threshold)) 