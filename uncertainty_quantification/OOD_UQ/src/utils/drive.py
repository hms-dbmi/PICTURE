import os
import gdown
import zipfile

# install and start server
# !apt install mongodb >log
# !service mongodb start
# !mongoimport --collection docs --type json --file /content/data/laws.json  --jsonArray

# from pymongo import MongoClient
# client = MongoClient()
# client.list_database_names()

def download(url, path):
    """Download the respective dataset from Gdrive
    Args:
        path (string): Path to the dataset.
        category (string): category from the dictonnary
    """

    # Get the dictionnary of all the file adresses from our google drive
    
    # Download the data from the Google Drive link
    # if not os.path.isfile("data/laws.json.zip"):
    #     url = 'https://drive.google.com/u/1/uc?id=17z0Kn9UNE1ZbKa82-5ML0UN7zMvs3o2y&export=download'
    #     gdown.download(url, "data/laws.json.zip", quiet=False)
    #     unzip("data/laws.json.zip")
        
    # Download the processed data from the Google Drive link
    gdown.download(url, path , quiet=False)
    
# Define a function to unzip the data file
def unzip(file_path: str) -> None:
  # Open the file as a zip file
  with zipfile.ZipFile(file_path, 'r') as zip_ref:
      # Extract the file contents to the data folder
      zip_ref.extractall("data/")
      
      
if __name__ == '__main__':
    download(url="https://drive.google.com/u/1/uc?id=1Ki6xLCVbuY__BJrzfBwvBeUupjjjOetw&export=download", path="data/yu_data.zip")
    unzip("data/yu_data.zip")
    