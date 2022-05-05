from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotInteractableException
import time
import pandas as pd
import re
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from tqdm.notebook import tqdm

######## TODO: input keyword via argparse.
#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument('--keyword', type=str, help='keyword for searching')
#args = parser.parse_args()

# keyword list
key_list = pd.read_csv("../../법정동코드 전체자료.txt", sep='\t', encoding='CP949')
key_list = key_list.법정동명[key_list.폐지여부=='존재'].str.split(n=2).str.get(1).unique()[1:]
key_list = key_list[pd.Series(key_list).str.endswith(('구', '동', '시', '군'))]

# init variables
review_df = pd.DataFrame(columns=['Region', 'Rating', 'Category', 'Review'])
rating_dict = {'맛있다':3, '괜찮다':2, '별로':1}

# set the scraper's option
options = webdriver.chrome.options.Options()
#options.add_argument('headless')
options.add_argument('incognito')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome('../../chromedriver', options=options)
driver.set_window_size(1000,800)
driver.implicitly_wait(0.5)
driver.get("https://www.mangoplate.com/")
time.sleep(3)

# start scrapy
for keyword in key_list:
    search_time = time.time()
    print("===============")
    print(keyword)
    driver.get("https://www.mangoplate.com/search/"+keyword)
    time.sleep(5)
    for page in driver.find_elements_by_css_selector("body > main > article > div.column-wrapper > div > div > section > div.paging-container > p.paging > a"):
        page.click()
        time.sleep(2)
        for each_restaurant in driver.find_elements_by_css_selector("a > span.title.ng-binding"):
            ActionChains(driver).key_down(Keys.CONTROL).click(each_restaurant).perform()
            driver.switch_to.window(driver.window_handles[-1])

            for info_table in driver.find_elements_by_css_selector("body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > table > tbody > tr"):
                if info_table.find_element_by_css_selector("th").text == '주소':
                    region = info_table.find_element_by_css_selector("td").text.split("\n")[0]
                    region = region.split()[1]
                elif info_table.find_element_by_css_selector("th").text == '음식 종류':
                    category = info_table.find_element_by_css_selector("td").text

            start_time = time.time()
            while (time.time() - start_time) < 601:  # 일정 사이트 자체에 더보기 오류(자체적 오류)를 해결 위해 루프 시간 10분 제한
                try: # more button click                  
                    more_button = driver.find_element_by_css_selector("body > main > article > div.column-wrapper > div.column-contents > div > section.RestaurantReviewList > div.RestaurantReviewList__MoreReviewButton")
                    ActionChains(driver).move_to_element(more_button).click(more_button).perform()
                    time.sleep(1)
                except ElementNotInteractableException:
                    break
                except NoSuchElementException:
                    break
            review_list, rating_list = [], []
            for review in driver.find_elements_by_css_selector("body > main > article > div.column-wrapper > div.column-contents > div > section.RestaurantReviewList > ul > li"):
                each_review = review.find_element_by_css_selector("a > div.RestaurantReviewItem__ReviewContent > div > p").text
                each_rating = review.find_element_by_css_selector("a > div.RestaurantReviewItem__Rating > span").text
                review_list.append(each_review)
                rating_list.append(rating_dict[each_rating])
            review_df = review_df.append(pd.DataFrame(data={'Region': [region]*len(rating_list), 'Rating':rating_list, 'Category':[category]*len(rating_list), 'Review':review_list}), ignore_index=True)
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
    print(f'{keyword} : {time.time() - search_time} 초 소요')

# finish
driver.quit()
review_df.to_parquet('mangoplate.parquet')
