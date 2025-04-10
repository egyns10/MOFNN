#processCSV.py
import csv
import numpy as np

def load_csv_to_array(filepath):
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]  # Convert the CSV file into a 2D array
    return data

def readCSVinNP(filepath):
    np.read_csv(filepath)