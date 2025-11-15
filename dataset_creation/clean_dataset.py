import pandas as pd
from dataset_utils import clean_dataframe

def clean_multiple_datasets(filepaths):
    for path in filepaths:
        df = pd.read_csv(path)
        df = clean_dataframe(df)
        df.to_csv(path, index=False)
        print(f"Cleaned and saved: {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True, help='CSV file paths to clean')
    args = parser.parse_args()
    
    clean_multiple_datasets(args.files)

