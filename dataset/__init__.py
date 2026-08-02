import torch
from torch.utils.data import Dataset

class RopeDataset(Dataset):
    def __init__(self, data_segments, labels, window_size=64, stride=32):
        """
        data_segments: list of numpy arrays, 每个 array shape 为 (Frames, 17, 2)
        labels: list of int, 对应每个 segment 的标签 (0: 不规范, 1: 规范)
        """
        self.samples = []
        self.window_size = window_size
        
        for segment, label in zip(data_segments, labels):
            num_frames = segment.shape[0]
            # 滑窗切分
            for i in range(0, num_frames - window_size + 1, stride):
                window = segment[i:i+window_size]
                # 展平为 (window_size, 34)
                self.samples.append((window.reshape(window_size, -1), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label = self.samples[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)