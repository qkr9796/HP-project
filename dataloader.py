from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self):
        ...

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return 1

    def get_input_size(self):
        return 1

    def get_output_size(self):
        return 1

    def process_data(self, data):
        # better Logscale Amount data
        # other processes needs to be done to dataset(encoding, preprocessing)

        return data


def create_dataloader(*args, **kwargs):
    return DataLoader(*args, **kwargs)
