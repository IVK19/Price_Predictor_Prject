import requests
from bs4 import BeautifulSoup
from time import sleep
import sqlite3

headers = {'User-Agent': 
           'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'}

def get_url():    
    for count in range(19, 22):

        url = f'https://msk.pulscen.ru/price/190215-noutbuki?page={count}'
    
        response = requests.get(url, headers=headers)
    
        html = BeautifulSoup(response.text, 'lxml')
    
        data = html.find_all('div', class_='product-listing__product-wrapper')
    
        for i in data:
            info = i.find('div', class_='product-listing__product-description js-find-mark').text
            price = int(i.find('i', class_='bp-price fsn').text.replace(' ', ''))
            yield info, price
        
base = sqlite3.connect('notebooks.db')
cur = base.cursor()
base.execute('CREATE TABLE IF NOT EXISTS {}(information text, price int)'.format('notebooks_data'))
base.commit()

for notebook in get_url():
    sleep(3)
    cur.execute('INSERT INTO notebooks_data VALUES(?, ?)', notebook)
    base.commit()

