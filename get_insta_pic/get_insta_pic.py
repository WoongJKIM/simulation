from selenium import webdriver as wd 
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from selenium.webdriver.common.keys import Keys

from bs4 import BeautifulSoup
import urllib.request

import time 
import re 
import json 
import pandas as pd

-- 로그인 하는 부분
url = "https://www.instagram.com/accounts/login/"

driver = wd.Chrome(executable_path = driver_path) 
driver.get(url)
time.sleep(10)

facebook_login_page_css = ".sqdOP.L3NKy.y3zKF" 
user_id="id" 
user_passwd="password"
facebook_id_form_name = "username" 
facebook_pw_form_name = "password"

id_input_form = driver.find_element_by_name(facebook_id_form_name) 
pw_input_form = driver.find_element_by_name(facebook_pw_form_name) 
id_input_form.send_keys(user_id) 
pw_input_form.send_keys(user_passwd) 
time.sleep(10) 

-- 정보 입력 창 미루기 버튼 누르기
facebook_login_btn = driver.find_element_by_css_selector(facebook_login_page_css) 
facebook_login_btn.click()
time.sleep(10)

facebook_login_btn = driver.find_element_by_css_selector(facebook_login_page_css) 
facebook_login_btn.click()
time.sleep(10)

after_facebook_alarm_btn_css = ".aOOlW.HoLwm"
after_alarm_btn = driver.find_element_by_css_selector(after_facebook_alarm_btn_css)
after_alarm_btn.click()

-- 특정 인스타그래머의 사진 다운로드
url = "https://www.instagram.com/jennie__kjn/"
driver.get(url)

req = driver.page_source
soup=BeautifulSoup(req, 'html.parser')

articles = soup.find_all('div', "v1Nh3 kIKUG _bz0w")

for article in articles:
    img_tag = article.find('img')
    img_url = img_tag['src']
    urllib.request.urlretrieve(img_url, "00000001.jpg")
