#!/usr/bin/env python3
import requests

# Request data
data = {
    "threshold": "0.8",
}
# Request files
files = {
    "face1": open("./image/IMG_7186.JPG", "rb"),
    "face2": open("./image/abc.jpg", "rb"),
}
# Endpoint URL
url = "https://api.luxand.cloud/photo/similarity"

# Request headers
headers = {
    "token": "d1f3e363f73047ce844578e0147f37a8",
}
# Making the POST request
response = requests.request("POST", url, headers=headers, data=data, files=files)

# Printing the response
print(response.text.encode('utf8'))