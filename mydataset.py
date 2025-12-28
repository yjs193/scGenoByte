from torch.utils.data import DataLoader, Dataset


class SCDataset(Dataset):
    def __init__(self, data, labels, pathways): # <-- 接收 pathways
        self.data = data
        self.labels = labels
        self.pathways = pathways

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        pathway = self.pathways[idx]
        if hasattr(sample, "toarray"):
             sample = sample.toarray()[0]

        return sample, label, pathway


class SCDataset_all(Dataset):
    def __init__(self, data, protein_targets):
        super().__init__()
        self.data = data
        self.protein_targets = protein_targets

    def __getitem__(self, index):
        data_seq = self.data[index].toarray()[0]
        return data_seq, self.protein_targets

    def __len__(self):
        return self.data.shape[0]