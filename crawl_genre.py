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
f = 'genre_list.txt'
#total_index = 0  # for dataset
# last_Page = max(int(input('Enter start "last_page":')), 0)
####
download_path = 'C:/Users/amandatsai/Downloads'
#start_genre = 20
#start_page = 1

# account_index=0
# account_list=[]
# change_index=0 #set for limited download


# def read_account():
#     global account_list
#     myWorkbook = xlrd.open_workbook("account.xlsx")
#     mySheets = myWorkbook.sheets()
#     mySheet = mySheets[0]
#     print("Read Total Account:")
#     for i in range(mySheet.nrows):
#         myRowValues = mySheet.row_values(i)
#         print(myRowValues)
#         account_list.append(myRowValues)


# def login(browser):
#     global account_index
#     sleep(3)
#     user = browser.find_element_by_id('user_email')
#     password = browser.find_element_by_id('upass')
#     btn = browser.find_element_by_id('submit')
#     user_agree = browser.find_element_by_id('user_disclaimer')

#     print("Now Account: "+account_list[account_index][0])
#     user.clear()
#     password.clear()
#     user.send_keys(account_list[account_index][0])
#     password.send_keys(account_list[account_index][1])
#     account_index = account_index+1

#     while(user_agree.is_selected() == False):
#         sleep(random.uniform(0.2, 1.0))

#     sleep(random.uniform(0.95, 3.01))  # avoid bang bang bang!!!
#     btn.click()


#def getfiles(browser):
    #global change_index
    #global total_index
    # get links per page
    #enter_list = browser.find_elements_by_xpath(
    #    "//div[@class='player-stats-wrapper']/a[@class='player-stats-icons stats-downloads']")

    #temp_list = []  # preserve link
    #for enter in enter_list:
    #    temp_list.append(enter.get_attribute("href"))

    #for tmp in temp_list:
    #    print(tmp)
    # print(temp_list)

    # ######   download file
    # data = {}
    # data['metadata'] = []
    # for index in range(len(enter_list)):
    #     print(index)
    #     browser.get(temp_list[index])

    #     # for change filename
    #     num=temp_list[index].split('detail/',1)[1]
    #     num=num.split('/')[0]
    #     num='0'+num
    #     print(num)
    #     sleep(random.uniform(0.2, 2.0))
    #     browser.find_element_by_link_text('Download').click() #if Logged in, text is 'Download'
    #     #browser.find_element_by_link_text('Login To Download').click() #if Logged in, text is 'Download'

    #     ### crawl metadata
    #     title = browser.find_elements_by_xpath("//div[@class='player-top']/h1/a[@class='player-title']")[0].text
    #     author = browser.find_elements_by_xpath("//div[@class='player-sub-title']/a[@class='icon-small icon-user']")[0].text
    #     print(">",title,"---",author)
    #     tags = []
    #     tags_list = browser.find_elements_by_xpath("//*[@id='body-left']/div[3]/a")
    #     for i in range(len(tags_list)):
    #         tmp_tag = tags_list[i].text
    #         # print('tag:',tmp_tag)
    #         tags.append(tmp_tag)
    #     print(tags)

    #     # save to json format
    #     data['metadata'].append({ 'index': total_index,'title': title, 'author': author, 'tags': tags, })

    #     sleep(random.uniform(3.5, 5.0))
    #     ### rename file
    #     dirs=os.listdir(os.getcwd())
    #     for files in dirs:
    #         if (files.find(num)!=-1):
    #             os.rename(files,str(total_index)+".wav")
    #             break

    #     total_index=total_index+1
    #     change_index=change_index+1

    #     # change to other account
    #     if(change_index==50):
    #         change_index=0
    #         browser.find_element_by_link_text('Log Out').click()
    #         sleep(random.uniform(0.95, 3.01)) # avoid bang bang bang!!!
    #         browser.find_element_by_link_text('Log In').click()
    #         login(browser)

    # with open('data.txt', 'w') as out:
    #     json.dump(data, out)


def main():
    
    # set download path (ex: D:\\)
    options = Options()
    options.add_experimental_option("prefs", {
        # "download.default_directory": r'/Users/summer/Desktop/test',
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })

    # open chrome
    browser = webdriver.Chrome('./chromedriver')
    #browser.get('https://www.looperman.com/account/login')
    browser.get('https://www.looperman.com/loops')
    # read_account()
    # login(browser)
    #browser.find_element_by_link_text('Loops & Samples').click()
    os.chdir(download_path)

    glist = []
    with open(f,'w') as o:
        for i in range(1, 65):
        # by genre selector
            s = Select(browser.find_element_by_name('gid'))
            s.select_by_index(i)
            genre_name = s.first_selected_option.text
            print(genre_name)
            glist.append(genre_name)
            o.write(genre_name+'\n')
            #browser.find_element_by_class_name('form-button').click()
            #genre_name = genre_name.replace(' ', '-').lower()
    print(glist)
        #browser.get('https://www.looperman.com/loops/genres/royalty-free-'+genre_name+'-loops-samples-sounds-wavs-download')
        # get genre last page
        # url = browser.find_element_by_link_text('Last').get_attribute('href')
        # temp = url.split('=', 1)
        # last_Page = temp[1].split('&', 1)[0]

        # if(last_Page == 0):
        #     url = browser.find_element_by_link_text('Last').get_attribute('href')
        #     last_Page = url.split('=', 1)[-1]
        # print("last_Page:"+str(last_Page))

        # page href form
        #first_page = browser.current_url
        # s1 = first_page.split('=', 1)
        # s2 = s1[1].split('&', 1)[1]
        # s1 = s1[0]+'='
        # s2 = '&'+s2
        #s1 = first_page.split('=', 1)
        #s1 = s1[0]+'?page='

        # by page selector
        # only need 1 page
        #now_page = s1+str(start_page)
        #print(now_page)
        #browser.get(now_page)
        #getfiles(browser)
        # for i in range(start_page, int(last_Page)+1):
        #     now_page = s1+str(i)  # +s2
        #     print(now_page)
        #     browser.get(now_page)
        #     getfiles(browser)

    # data = {'key1': value1, 'key2': value2}
    # ret = json.dumps(data)

    # open('out.json', 'w') with fp:
    #     fp.write(ret)


if __name__ == "__main__":
    main()
