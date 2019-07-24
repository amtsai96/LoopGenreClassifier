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
total_index = 0  # for dataset
# last_Page = max(int(input('Enter start "last_page":')), 0)
####
download_path = 'C:/Users/amandatsai/Downloads'
start_genre = int(input('Enter genre #:'))
start_page = 1


def getfiles(browser):
    global change_index
    global total_index
    # get links per page
    enter_list = browser.find_elements_by_xpath(
        "//div[@class='player-stats-wrapper']/a[@class='player-stats-icons stats-downloads']")

    temp_list = []  # preserve link
    for enter in enter_list:
        temp_list.append(enter.get_attribute("href"))

    for tmp in temp_list: print(tmp)


def main():
    
    # set download path (ex: D:\\)
    # options = Options()
    # options.add_experimental_option("prefs", {
    #     # "download.default_directory": r'/Users/summer/Desktop/test',
    #     "download.prompt_for_download": False,
    #     "download.directory_upgrade": True,
    #     "safebrowsing.enabled": True
    # })

    # open chrome
    browser = webdriver.Chrome('./chromedriver')
    #browser.get('https://www.looperman.com/account/login')
    browser.get('https://www.looperman.com/loops')
    # read_account()
    # login(browser)
    #browser.find_element_by_link_text('Loops & Samples').click()
    #os.chdir(download_path)

    for i in range(start_genre, start_genre+1):#65):
        # by genre selector
        s = Select(browser.find_element_by_name('gid'))
        s.select_by_index(i)
        genre_name = s.first_selected_option.text
        print(genre_name)
        #browser.find_element_by_class_name('form-button').click()
        genre_name = genre_name.replace(' ', '-').lower()
        browser.get('https://www.looperman.com/loops/genres/royalty-free-'+genre_name+'-loops-samples-sounds-wavs-download')
        # get genre last page
        # url = browser.find_element_by_link_text('Last').get_attribute('href')
        # temp = url.split('=', 1)
        # last_Page = temp[1].split('&', 1)[0]

        # if(last_Page == 0):
        #     url = browser.find_element_by_link_text('Last').get_attribute('href')
        #     last_Page = url.split('=', 1)[-1]
        # print("last_Page:"+str(last_Page))

        # page href form
        first_page = browser.current_url
        # s1 = first_page.split('=', 1)
        # s2 = s1[1].split('&', 1)[1]
        # s1 = s1[0]+'='
        # s2 = '&'+s2
        s1 = first_page.split('=', 1)
        s1 = s1[0]+'?page='

        # by page selector
        # only need 1 page
        now_page = s1+str(start_page)
        print(now_page)
        browser.get(now_page)
        getfiles(browser)
        # for i in range(start_page, int(last_Page)+1):
        #     now_page = s1+str(i)  # +s2
        #     print(now_page)
        #     browser.get(now_page)
        #     getfiles(browser)


if __name__ == "__main__":
    main()
