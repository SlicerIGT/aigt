import argparse
import requests
import os
import zipfile


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data', help='Directory to download data to')
args = parser.parse_args()


# Dictionary of files and urls to download

files_and_urls = {
    'TrainingData_0_128.zip': 'https://onedrive.live.com/download?cid=7230D4DEC6058018&resid=7230D4DEC6058018%2193587&authkey=AAItx643nwpFp2Y',
    'TestingData_0_128.zip': 'https://onedrive.live.com/download?cid=7230D4DEC6058018&resid=7230D4DEC6058018%2193586&authkey=AIX-z2g1KNOvR4A',
    'TrainingDataPrep.zip': 'https://onedrive.live.com/download?cid=7230D4DEC6058018&resid=7230D4DEC6058018%2193497&authkey=ACpi1EdsTYA5nWo',
    'TestingDataPrep.zip' : 'https://onedrive.live.com/download?cid=7230D4DEC6058018&resid=7230D4DEC6058018%2193505&authkey=AK8dlFgBTHnE2aY'
    }


# Make sure data directory exists

os.makedirs(args.data_dir, exist_ok=True)
print(f'Downloading data to {args.data_dir}')


# Download files using tqdm to show progress bar

for filename, url in files_and_urls.items():
    local_file = os.path.join(args.data_dir, filename)
    
    # If local file already exists, delete it

    if os.path.exists(local_file):
        os.remove(local_file)
        print(f'Removed existing {filename}')

    response = requests.get(url)

    if response.status_code == 200:  # Check if the request was successful
        data = response.content
        
        with open(local_file, 'wb') as f:
            f.write(data)
        print(f'Downloaded {filename}')

        # Unzip file, overwriting existing files

        with zipfile.ZipFile(local_file, 'r') as zip_ref:
            zip_ref.extractall(args.data_dir)
        print(f'Unzipped {filename}')

        # Remove zip file

        os.remove(local_file)

    else:
        print(f'An error occurred: {response.status_code}')
