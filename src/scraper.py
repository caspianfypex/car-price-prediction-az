import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd

dataPath = Path(__file__).resolve().parent.parent / 'data'
MAIN_URL = 'https://turbo.az/'
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/116.0.0.0 Safari/537.36"
}

session = requests.Session()
session.headers.update(headers)

def get_cars(brand_id,page=1):
    url = f'{MAIN_URL}autos?q[make][]={brand_id}&page={page}'
    links = []
    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f'Error occurred: {e}')
        return None
    soup = BeautifulSoup(response.text, 'lxml')
    products = soup.find_all('a', class_='products-i__link')
    for p in products:
        links.append(p['href'])
    return links


def get_product_info(car_url):
    target_url = f'{MAIN_URL}{car_url}'
    car_features = []
    try:
        response = session.get(target_url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f'Error occurred: {e}')
        return None
    soup = BeautifulSoup(response.text, 'lxml')
    features = soup.find_all('div', class_='product-properties__i')
    for f in features:
        if f.find('label').text in ('Marka', 'Yürüş', 'Model', 'Buraxılış ili', 'Vəziyyəti'):
            car_features.append(f.find('span').text)
        elif f.find('label').text == 'Mühərrik':
            add_features = [None, None, None]
            arr = f.find('span').text
            arr = arr.split(' / ')
            for s in arr:
                if 'L' in s or 'sm3' in s:
                    add_features[0] = s
                elif 'a.g' in s:
                    add_features[1] = s
                else:
                    add_features[2] = s
            car_features.extend(add_features)
    car_features.append(soup.find('div', class_='product-price__i product-price__i--bold').text)
    return car_features

def get_final_page(brand_id):
    url = f'{MAIN_URL}autos?q[make][]={brand_id}'
    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f'Error occurred: {e}')
        return None
    soup = BeautifulSoup(response.text, 'lxml')
    if soup.find('span', class_='last'):
        last_page = soup.find('span', class_='last')
        last_page_url = 'https://turbo.az/' + last_page.find('a')['href']
    else:
        last_page = soup.find_all('span', class_='page')
        if len(last_page) != 0:
            last_page_url = 'https://turbo.az/' + last_page[-1].find('a')['href']
        else:
            return 1
    try:
        response = session.get(last_page_url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f'Error occurred: {e}')
        return None
    soup = BeautifulSoup(response.text, 'lxml')
    last_page_number = int(soup.find('span', class_='page current').text)
    print(last_page_number)
    return last_page_number

def get_brand_ids():
    try:
        response = session.get(MAIN_URL, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f'Error occurred: {e}')
        return None
    soup = BeautifulSoup(response.text, 'lxml')
    dropdown = soup.find('select', {'id': 'q_make'})
    if not dropdown:
        print("Not found dropdown, error")
        return None
    brands = []
    for option in dropdown.find_all('option'):
        make_id = option.get('value')
        if make_id:
            brands.append(make_id)
    return brands

def extract_data(brand_id=-1, delay=5, filename='data'):
    data = []
    output = None
    if brand_id != -1:
        max_page=get_final_page(brand_id)
        for i in range(1,max_page+1):
            print(f'Scraping Page: {i}/{max_page}')
            cars = get_cars(brand_id,i)
            for c in cars:
                data.append(get_product_info(c))
                time.sleep(delay)
            output = pd.DataFrame(data, columns=['Brand', 'Model', 'Year', 'Engine Size', 'Horse Power', 'Fuel Type', 'Kilometrage', 'Status', 'Price'])
            output.to_csv(dataPath / f'{filename}.csv', index=False)
        return output
    else:
        brand_ids = get_brand_ids()
        for x in brand_ids:
            max_page = get_final_page(x)
            for i in range(1, max_page + 1):
                print(f'Scraping Page: {i}/{max_page}')
                cars = get_cars(x, i)
                for c in cars:
                    data.append(get_product_info(c))
                    time.sleep(delay)
                output = pd.DataFrame(data, columns=['Brand', 'Model', 'Year', 'Engine Size', 'Horse Power', 'Fuel Type', 'Kilometrage', 'Status', 'Price'])
                output.to_csv(dataPath / f'{filename}.csv', index=False)
        return output
