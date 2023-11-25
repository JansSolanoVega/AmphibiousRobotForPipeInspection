import requests
# URL del servidor Flask en tu Raspberry Pi
url = 'http://192.168.1.100:8080/download'

output_path = 'videoNuevo.mp4'  # Replace with the desired output path

while 1:
    response = requests.get(url)
    print(response.content)

    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print('Video downloaded successfully!')
        break
    else:
        print('Failed to download the video.')
