import requests

REST_URL = "http://localhost:8090/tasks/create/file"
SAMPLE_FILE = "/home/jolie/下載/all/1clickdownload/VirusShare_3a38ed263b0785ebca4e32aec9eea290.json"
HEADERS = {"Authorization": "Bearer S4MPL3"}

try:
    with open(SAMPLE_FILE, "rb") as sample:
        files = {"file": ("temp_file_name", sample)}
        r = requests.post(REST_URL, headers=HEADERS, files=files)
        r.raise_for_status()  # 確保請求成功，否則引發異常

        response_json = r.json()
        if "task_id" in response_json:
            task_id = response_json["task_id"]
            print(f"Task ID: {task_id}")
        else:
            print("No task ID found in the response.")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except KeyError as e:
    print(f"KeyError: {e}")
