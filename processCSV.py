#processCSV.py
import pandas as pd

def readCSV(filepath):
    pd.read_csv(filepath_or_buffer=filepath,header=0)