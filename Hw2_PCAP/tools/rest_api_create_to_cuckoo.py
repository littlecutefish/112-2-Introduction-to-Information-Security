import os
import requests

# Cuckoo API endpoint
API_URL = "http://localhost:8090/tasks/create/file"

# Authorization token
TOKEN = "pmrZLlYHduSXMXAm2W-GHg"

# Directory containing virus files
FILES_DIR = "/home/jolie/下載/all/adclicer"

# Loop through files in directory
for filename in os.listdir(FILES_DIR):
    file_path = os.path.join(FILES_DIR, filename)

    # Skip directories
    if os.path.isdir(file_path):
        continue

    # Open file
    with open(file_path, "rb") as file:
        files = {"file": (filename, file)}
        headers = {"Authorization": f"Bearer {TOKEN}"}

        # Send HTTP POST request
        response = requests.post(API_URL, headers=headers, files=files)

        # Check response status
        if response.status_code == 200:
            task_id = response.json()["task_id"]
            print(f"File {filename} uploaded successfully. Task ID: {task_id}")
        else:
            print(f"Failed to upload file {filename}. Status code: {response.status_code}")