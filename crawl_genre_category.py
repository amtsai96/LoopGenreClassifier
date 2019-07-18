from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from time import sleep
import xlrd
import json
import random
import os
import sys
import re

####
download_path = 'C:/Users/amandatsai/Downloads'

def get_list(browser, f, num, f2=None):
    all_list = []
    num=40
    with open(f,'w') as o:
        for i in range(1, num+1):
            s = Select(browser.find_element_by_name('cid'))
            s.select_by_index(i)
            name = s.first_selected_option.text
            print(name)
            all_list.append(name)
            o.write(name+'\n')

    print(all_list)
    if f2:
        with open(f2, 'w') as o: o.write(str(all_list))

def main():
    mode = int(input('Enter mode(genre=1, cat=0):'))
    #num = int(input('enter num(genre=64, cat=40):'))
    if mode == 0:
        num = 40
        f = 'categories.txt'
        f2 = 'category_list.txt'
    else:
        num = 64
        f = 'genres.txt'
        f2 = 'genre_list.txt'


    options = Options()
    options.add_experimental_option("prefs", {
        #"download.prompt_for_download": False,
        #"download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })

    # open chrome
    browser = webdriver.Chrome('./chromedriver')
    browser.get('https://www.looperman.com/loops')
    #os.chdir(download_path)

    get_list(browser, f, num, f2)


if __name__ == "__main__":
    main()
