from tkinter import Y
import pandas as pd 
from torch.utils.data import Dataset

class DataFrameTextClassicationDataset(Dataset):
    def __init(self, df:pd.DataFrame, x_label : str = 'text', y_label : str = 'label'):
        self.x = df[x_label]
        self.y = df[y_label]
        
        self.length = len(self.x)
        self.n_classes = len(self.y.cat.catagories)
        
    def __getitem__(self, index) -> dict:
        x = self.x.iloc[index]
        y = self.y.iloc[index]
        return {
            'x' : str(x), 
            'y' : int(y)
        }
    def __len__(self):
        return self.length 
    
    @staticmethod
    def from_file(file_path : str, 
                  x_label: str = 'text', 
                  y_label: str = 'label'): 
        df = pd.read_csv(file_path)
        
        return DataFrameTextClassicationDataset(df, x_label, y_label)
    