import torch, os
p="/mnt/dc_cache/sem_feats/train2017"
f=sorted(os.listdir(p))[0]
d=torch.load(os.path.join(p,f))
print(d["feat"].shape, d["orig_size"], d["canonical_size"], d["semantic_size"], d["t_used"])
