from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from time import sleep
import xlrd
import json
import random
import os
import re
import datetime
####
total_index = max(int(input('Enter start "total_index":')), 0)  # for dataset
last_Page = max(int(input('Enter start "last_page":')), 0)
account_index = max(min(int(input('Enter start "account_index":')), 43), 16)
metadata_file = './data/data.txt'

current_page_urls = './data/current_urls.txt'
# record [the last index] which was already done
current_index = './data/current_index.txt'
# record [category / pages] which was already done
current_page = './data/current_page.txt'

not_edited_files = './data/not_edited_files.txt'
log_file = './data/log' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'
download_path = './Downloads'
download_size = 150
####

account_list = []
change_index = 0  # set for limited download
start_genre = 1  # start index of genre i
start_page = 1  # start index of page #
start_index = 0  # start index of urls in currrent page


def read_account():
    global account_list
    myWorkbook = xlrd.open_workbook("account.xlsx")
    mySheets = myWorkbook.sheets()
    mySheet = mySheets[0]
    print("Read Total Account...")
    for i in range(mySheet.nrows):
        myRowValues = mySheet.row_values(i)
        account_list.append(myRowValues)


def login(browser):
    global account_index
    sleep(3)
    user = browser.find_element_by_id('user_email')
    password = browser.find_element_by_id('upass')
    btn = browser.find_element_by_id('submit')
    user_agree = browser.find_element_by_id('user_disclaimer')

    print(">>> Current Account: "+account_list[account_index][0])
    print(">>> Current Account: " +str(account_index)+"_"+
          account_list[account_index][0], file=open(log_file, 'a'))
    user.clear()
    password.clear()
    user.send_keys(account_list[account_index][0])
    password.send_keys(account_list[account_index][1])
    account_index = max(16, (account_index+1) % len(account_list))

    while(user_agree.is_selected() == False):
        sleep(random.uniform(0.2, 1.0))

    sleep(random.uniform(0.95, 3.01))  # avoid bang bang bang!!!
    btn.click()

def getfiles(browser, isFirst):
    global change_index
    global total_index
    global start_index

    get_new_links = True
    if isFirst and total_index != 0:  # start from specific index
        isFirst = False
        if os.path.exists(current_index) and os.path.exists(current_page):
            get_new_links = False
            with open(current_index, 'r') as i:
                start_index = int(i.readline()) + 1

            temp_list = []  # save the urls last time
            with open(current_page_urls, 'r') as i:
                for ind, line in enumerate(i.readlines()):
                    if ind < start_index:
                        continue
                    temp_list.append(line)
        else:
            total_index = 0
        print('-> start_genre_index = ', start_genre)
        print('-> start_page_index = ', start_page)
        print('-> start_url_index = ', start_index)
        print('-> start_genre_index = ', start_genre, file=open(log_file, 'a'))
        print('-> start_page_index = ', start_page, file=open(log_file, 'a'))
        print('-> start_url_index = ', start_index, file=open(log_file, 'a'))

    if(get_new_links):
        # get links per page
        enter_list = browser.find_elements_by_xpath(
            "//div[@class='player-stats-wrapper']/a[@class='player-stats-icons stats-downloads']")

        temp_list = []  # preserve link
        for enter in enter_list:
            temp_list.append(enter.get_attribute("href"))

        # print(temp_list,len(temp_list))
        # save current progress
        with open(current_page_urls, 'w') as o:
            for line in temp_list:
                o.write(line+'\n')

    # download file
    data = []
    for index in range(len(temp_list)):  # enter_list)):
        print(str(index+1)+' of ' + str(len(temp_list)))
        print(str(index+1)+' of ' + str(len(temp_list)), file=open(log_file, 'a'))
        browser.get(temp_list[index])

        # for change filename
        num = temp_list[index].split('detail/', 1)[1]
        num = num.split('/')[0]
        num = "%07d" % int(num)
        print("total_index:", total_index)
        print("num:", num)
        print("total_index:", total_index, file=open(log_file, 'a'))
        print("num:", num, file=open(log_file, 'a'))
        sleep(random.uniform(0.2, 2.0))
        # if Logged in, text is 'Download'
        browser.find_element_by_link_text('Download').click()
        # browser.find_element_by_link_text('Login To Download').click() #if Logged in, text is 'Download'

        # crawl metadata
        title = browser.find_elements_by_xpath(
            "//div[@class='player-top']/h1/a[@class='player-title']")[0].text
        author = browser.find_elements_by_xpath(
            "//div[@class='player-sub-title']/a[@class='icon-small icon-user']")[0].text
        print(">", title, "---", author)
        print(">", title, "---", author, file=open(log_file, 'a', encoding='utf-8'))
        tags = []
        tags_list = browser.find_elements_by_xpath(
            "//*[@id='body-left']/div[3]/a")
        for i in range(len(tags_list)):
            tmp_tag = tags_list[i].text
            # print('tag:',tmp_tag)
            tags.append(tmp_tag)
        print(tags)
        print(tags, file=open(log_file, 'a'))

        # save to json format
        metadata_now = {'index': "%06d" % int(
            total_index), 'file_index': num, 'title': title, 'author': author, 'tags': tags, }
        data.append(metadata_now)

        sleep(random.uniform(3.5, 5.0))
        # rename file
        renameDone = False
        maxIter = 80
        while not renameDone:
            if maxIter <= 0:
                if os.path.exists(not_edited_files):
                    append_write = 'a'
                else:  # create new file
                    append_write = 'w'
                with open(not_edited_files, append_write) as out:
                    json.dump({'loop index': "%06d" % int(total_index)}, out)
                    out.write('\n')
                break
            maxIter -= 1
            sleep(random.uniform(0.5, 1.3))
            dirs = os.listdir(os.getcwd())  # return all folders & files
            for files in dirs:
                if os.path.isfile(files) and files.startswith('looperman') and files.endswith('.wav') and str('-'+num+'-') in str(files):
                    print('*'+files)
                    print('*'+files, file=open(log_file, 'a'))
                    os.rename(files, "%06d" % int(total_index)+".wav")
                    print('**'+"%06d" % int(total_index)+".wav")
                    print('**'+"%06d" % int(total_index) +
                          ".wav", file=open(log_file, 'a'))
                    renameDone = True
                    break

                # if (files.find(num)!=-1):
                #     os.rename(files,"%06d" % int(total_index)+".wav")
                #     break
            if not renameDone:
                print('Sleeping...')
                sleep(random.uniform(0.5, 1.3))
            else:
                print('RENAME DONE!!!')
                print('RENAME DONE!!!', file=open(log_file, 'a'))

        with open(current_index, 'w') as o:
            print('Writing Index file ...')
            o.write(str(index))

        renameDone = False
        total_index = total_index+1
        change_index = change_index+1

        if os.path.exists(metadata_file):
            append_write = 'a'
        else:  # create new file
            append_write = 'w'
        with open(metadata_file, append_write) as out:
            json.dump(metadata_now, out)
            out.write('\n')
        print('-----------------------')

        # change to other account
        if(change_index == download_size):
            change_index = 0
            browser.find_element_by_link_text('Log Out').click()
            sleep(random.uniform(0.95, 3.01))  # avoid bang bang bang!!!
            browser.find_element_by_link_text('Log In').click()
            login(browser)


def main():
    global start_genre
    global start_page
    global last_Page
    # set download path (ex: D:\\)
    options = Options()
    options.add_experimental_option("prefs", {
        # "download.default_directory": r'C:/Users/amandatsai/Downloads/loops',#'/Users/summer/Desktop/test',
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })

    # open chrome
    browser = webdriver.Chrome('./chromedriver')
    browser.get('https://www.looperman.com/account/login')

    read_account()
    login(browser)
    # browser.find_element_by_link_text('Loops & Samples').click()
    # os.chdir(download_path)

    # check if we need to load from specific index
    if total_index != 0:  # start from specific index
        if os.path.exists(current_index) and os.path.exists(current_page):
            with open(current_page, 'r') as i:
                start_genre = int(i.readline())  # [i value] of genre
                start_page = int(i.readline())

    for i in range(start_genre, 65):  # 64 genres
        if i != start_genre:
            start_page = 1

        browser.find_element_by_link_text('Loops & Samples').click()
        os.chdir(download_path)

        # by genre selector
        s = Select(browser.find_element_by_name('gid'))
        s.select_by_index(i)
        genre_name = s.first_selected_option.text
        print(i, s.first_selected_option.text)
        print(s.first_selected_option.text, file=open(log_file, 'a'))
        # browser.find_element_by_class_name('form-button').click()
        genre_name = genre_name.replace(' ', '-').lower()
        browser.get('https://www.looperman.com/loops/genres/royalty-free-' +
                    genre_name+'-loops-samples-sounds-wavs-download')

        # get genre last page
        if (last_Page == 0):
            url = browser.find_element_by_link_text('Last').get_attribute('href')
            # temp = url.split('=', 1)
            # last_Page = temp[1].split('&', 1)[0]
            last_Page = url.split('=', 1)[-1]
        print("last_Page:"+str(last_Page))

        # page href form
        first_page = browser.current_url
        # s1 = first_page.split('=', 1)
        # s2 = s1[1].split('&', 1)[1]
        # s1 = s1[0]+'='
        # s2 = '&'+s2
        s1 = first_page.split('=', 1)
        s1 = s1[0]+'?page='

        # by page selector
        for p in range(start_page, int(last_Page)+1):
            # now_page = s1+str(p)+s2
            now_page = s1+str(p)
            print(now_page)
            print(now_page, file=open(log_file, 'a'))
            browser.get(now_page)
            if i == start_genre and p == start_page:  # first time, needed to start in the middle point
                getfiles(browser, True)
            else:
                getfiles(browser, False)
            with open(current_page, 'w') as o:
                print('Writing Page file ...')
                o.write(str(i)+'\n')
                o.write(str(p))


if __name__ == "__main__":
    main()
