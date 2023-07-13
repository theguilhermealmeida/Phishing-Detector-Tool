import requests
import time
import torch
import cv2
import os
import json
from PIL import Image
from urllib.parse import urlparse
from html2image import Html2Image

OUTPUT_FOLDER = 'user/html2image_screenshot/'
COMPANIES_URLS = 'companies_urls.json'
THRESHOLD = 0.55

# check if url is valid
def is_valid_url(url):
    try:
        response = requests.get(url)
        print("Responde code: " + str(response.status_code) + "\n")

        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# check if url structure is valid
def check_url_structure(url):
    parsed_url = urlparse(url)
    return all([parsed_url.scheme, parsed_url.netloc])

# check if image path is valid
def validate_image_path(path):
    if not os.path.isfile(path):
        print("Invalid path: File does not exist.")
        return False

    try:
        Image.open(path)
        return True
    except IOError:
        print("Invalid image format or corrupted file.")
        return False

# compare user url with companies urls
def compare_urls(user_url):
    user_url_normalized = urlparse(user_url).netloc.lower()
    with open(COMPANIES_URLS) as f:
        companies_urls = json.load(f)

    for company in companies_urls['companies']:
        for url in company['websites']:
            url_normalized = urlparse(url).netloc.lower()

            if (url_normalized == user_url_normalized):
                return company['name']

    return None


def main():
    # menu
    choice = 0
    while choice != "1" and choice != "2":
        choice = input("1) Provide url (we'll take the screenshot)\n2) Provide url and screenshot path\nSelect an option: ")


    if choice == "1":
        valid_url = 0
        while valid_url == 0:
            url = input("\nEnter url: ")
            if is_valid_url(url):
                print("\nValid URL.\n")
                valid_url = 1
            else:
                print("Invalid URL or unable to connect.\n")
        print("Taking screenshot...\n")
        hti = Html2Image(output_path=OUTPUT_FOLDER)
        filename = time.strftime("%Y%m%d-%H%M%S") + ".png"
        hti.screenshot(url=url, save_as=filename)
        print(f"\nScreenshot saved as {filename} in {OUTPUT_FOLDER} folder.")

    elif choice == "2":
        valid_url = 0
        while valid_url == 0:
            url = input("\nEnter url: ")
            if check_url_structure(url):
                print("\nValid URL structure.\n")
                valid_url = 1
            else:
                print("Invalid URL structure.\n")
        valid_path = 0
        while valid_path == 0:
            file_path= input("Enter screenshot path: ")
            if (validate_image_path(file_path)):
                print("Valid screenshot path.")
                valid_path = 1


    # get trained model
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5/runs/train/yolo_logo_detection23/weights/best.pt')

    # run inference
    print("\nFinding logo on screenshot...\n")
    results = model(OUTPUT_FOLDER + filename if choice == "1" else file_path)

    # display results
    results.print()
    results.show()

    # check if logo detected
    if len(results.pandas().xyxy[0]) == 0:
        print("\nImpossible to determine if website is phishing.")
        print("No logo detected.")
        print("Possible reasons:")
        print("1) Company logo is not present in the dataset.")
        print("2) AI model couldn't detect logo.")
        print("3) Screenshot does not show logo.")
        exit()

    else:
        print("Logo detected!\n")
        df = results.pandas().xyxy[0]
        for i,c in zip(df['name'], df['confidence']):
            print(i + " logo detected" + "\n")


    companies_found, conf = df['name'], df['confidence']
    company_url_matched = compare_urls(url)

    companies = []
    for company, conf in zip(companies_found, conf):
        if company_url_matched == company and company not in companies:
            if (conf < THRESHOLD):
                print("Unable to determine if website is phishing - level of confidence is too low\n")
                exit()
            companies.append(company)
            print("URL provided matches with " + company + " website.\n")
            print("Website is not phishing - confidence: " + str(round(conf, 2)) + "\n")
            exit()


    if (round(df['confidence'].max(), 2) < THRESHOLD):
        print("Unable to determine if website is phishing - level of confidence is too low\n")
        exit()

    print("URL provided does not match with " + company + " website.\n")
    print("Website is phishing - confidence: " + str(round(df['confidence'].max(), 2)) + "\n")

if __name__ == "__main__":
    main()
