import numpy as np

class TensorConverter:
    def __init__(self, data):
        self.data = data

    def convert(self):
        # Convert the data to a NumPy array
        tensor = np.array(self.data)
        return tensor
    
    def temporal_split(self, indices):
        # Split the tensor into multiple parts based on the provided indices
        splits = np.split(self.data, indices)
        return splits