import os
import requests

# Cuckoo API endpoint
API_URL_for_pcap = "http://localhost:8090/pcap/get/"
API_URL_for_filename = "http://localhost:8090/tasks/view/"

# Authorization token
TOKEN = "pmrZLlYHduSXMXAm2W-GHg"

# Directory containing virus files
# FILES_DIR = "/home/jolie/下載/all/1clickdownload"

# Directory to save the PCAP files
OUTPUT_DIR = "/home/jolie/文件/Malware/adclicer"

# List of Task IDs
TASK_IDS = [72,74,76,78,80,84,86,88]

# Headers with Authorization token
headers = {"Authorization": f"Bearer {TOKEN}"}

# Loop through each task ID
for TASK_ID in TASK_IDS:
    # Send HTTP GET request to download PCAP file
    response_for_pcap = requests.get(API_URL_for_pcap + str(TASK_ID), headers=headers)

    # Check response status
    if response_for_pcap.status_code == 200:
        # Send HTTP GET request to get task details
        response_for_filename = requests.get(API_URL_for_filename + str(TASK_ID), headers=headers)

        # Check response status
        if response_for_filename.status_code == 200:
            # Parse response JSON
            task_details = response_for_filename.json()["task"]

            # Extract filename from task details and remove path
            filename_with_extension = os.path.basename(task_details["target"])

            # Remove extension from filename
            filename_without_extension, extension = os.path.splitext(filename_with_extension)

            # Create folder with filename as folder name
            folder_name = filename_without_extension  # Use filename without extension as folder name
            folder_path = os.path.join(OUTPUT_DIR, folder_name)

            # Check if folder already exists
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Save PCAP content to a file within the folder
            file_path = os.path.join(folder_path, "dump.pcap")
            with open(file_path, "wb") as pcap_file:
                pcap_file.write(response_for_pcap.content)
            print(f"PCAP file downloaded and saved to folder '{folder_name}' successfully.")
        else:
            print(f"Failed to get task details for Task ID {TASK_ID}. Status code: {response_for_filename.status_code}")
    else:
        print(f"Failed to download PCAP file for Task ID {TASK_ID}. Status code: {response_for_pcap.status_code}")
