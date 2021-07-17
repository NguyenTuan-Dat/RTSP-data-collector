from pexels_api import API
import cv2
import os
import requests

PEXELS_API_KEY = "563492ad6f917000010000017e120edb04bc4032aff251e04b8487d2"
PATH_TO_SAVE = "/Users/ntdat/Downloads/Cropped_by_scrfd/crawl_from_pexels/Glasses+Mask"

api = API(PEXELS_API_KEY)

for idx in range(0, 100):
    api.search('eyeglasses mask', page=idx, results_per_page=100)

    photos = api.get_entries()
    # Loop the five photos
    for photo in photos:
        # Print photographer
        print('Photographer: ', photo.photographer)
        # Print url
        print('Photo url: ', photo.url)
        # Print original size url
        print('Photo original size: ', photo.large2x)
        file_name = photo.large2x.split("/")[-1].split("?")[0]
        session = requests.Session()
        headers = {
            'User-Agent': "Edg/91.0.864.59",
            "Authorization": PEXELS_API_KEY
        }
        req = requests.Request('GET', photo.large2x, headers=headers)

        prepared = session.prepare_request(req)
        img_data = session.send(prepared).content
        with open(os.path.join(PATH_TO_SAVE, file_name), 'wb') as handler:
            handler.write(img_data)
