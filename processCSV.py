#processCSV.py
import csv

def load_csv_to_array(filepath):
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]  # Convert the CSV file into a 2D array
    return data
