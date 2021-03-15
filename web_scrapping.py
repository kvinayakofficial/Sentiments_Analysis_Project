import requests
from bs4 import BeautifulSoup as bsp
from selenium import webdriver as wd
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
import re
import io
import pandas as pd
import sqlite3 as sql
from csv import writer

browser = wd.Chrome(ChromeDriverManager().install())

def review_scrapper():
    for i in range(1,11):
        url = "https://www.etsy.com/in-en/c/jewelry/earrings/ear-jackets-and-climbers?ref=pagination&page={}".format(i)
        browser.get(url)
        sleep(3)
        for product in range(1,65):
            prod_xpath = browser.find_element_by_xpath('//*[@id="content"]/div/div[1]/div/div[3]/div[2]/div[2]/div[2]/div/div/ul/li[{0}]/div/a/div[1]/div[1]/div/div/div/div/img'.format(product))
            prod_xpath.click()
            windows = browser.window_handles
            for handle in windows[1:]:
                browser.switch_to.window(handle)
                html_page = browser.page_source
                soup = bsp(html_page,'html.parser')
                soup.encode("utf-8")
                prod_review = soup.find_all('p',id=re.compile('^review-preview-toggle-\d+'))
                for r in prod_review:
                    p_text=[]
                    p_text.append(r.getText())
                    with open('etsy_scrap_review.csv', 'a') as csv_file:
                        data_write = writer(csv_file)
                        try:
                            data_write.writerow(p_text)
                        except UnicodeEncodeError:
                            pass
                        csv_file.close()
                sleep(3)
                browser.close()
                browser.switch_to.window(windows[0])
            sleep(1)

review_scrapper()

df = pd.read_csv(r'etsy_scrap_review.csv')
df.columns=['reviews']
conn = sql.connect('scrappedReviewsAll.db')
df.to_sql('scrappedReviewsAllTable', conn)