import os

# Download code taken from Code taken from https://stackoverflow.com questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
import requests, zipfile, os
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)    
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None
def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

file_id = '1h9SykUnuZul7J3Nbms2QGH1wa85nbN2-'
chpt_path = './checkpoints/label2city_1024p/'
if not os.path.isdir(chpt_path):
	os.makedirs(chpt_path)
destination = os.path.join(chpt_path, 'latest_net_G.pth')
download_file_from_google_drive(file_id, destination) 

