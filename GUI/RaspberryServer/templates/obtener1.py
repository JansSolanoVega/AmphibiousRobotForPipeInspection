import requests

url = 'https://192.162.1.4:8080'
response = requests.get(url)

if response.status_code == 200:
    text = response.text
    print(text)
else:
    print(f'Error: {response.status_code}')