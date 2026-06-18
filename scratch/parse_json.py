import urllib.request
import json

url = "https://pypi.org/pypi/planx-sdk/json"
try:
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode('utf-8'))
        print("Latest version:", data.get("info", {}).get("version"))
        print("Releases list:", list(data.get("releases", {}).keys()))
except Exception as e:
    print("Error:", e)
