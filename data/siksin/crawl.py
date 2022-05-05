from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import time
import pandas as pd
import json
import re

# variables
MAX_ITER = 10000
SAVE_CHECKPOINT_ITER = 5000
keyword_list = ['제주','대구','창원']

for keyword in keyword_list:
    print("Keyword:", keyword)

    # driver setup
    driver_path = '../../chromedriver'
    options = webdriver.chrome.options.Options()
    options.add_argument('--no-sandbox')
    options.add_argument('--headless')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(driver_path, options=options)

    # scrape all info and save df; to be called when exiting the loop
    def save_df():
        # categories
        store_div = review_div.find_elements_by_css_selector("div.store_area_menu")
        categories = [x.find_element_by_css_selector("em").text for x in store_div]

        # ratings
        ratings_list = ['1.0','1.5','2.0','2.5','3.0','3.5','4.0','4.5','5.0']
        ratings = [x.text for x in review_div.find_elements_by_css_selector("strong") if x.text in ratings_list]

        # reviews
        reviews = [x.text for x in review_div.find_elements_by_css_selector("p")]

        # save df
        data = {'Region':[keyword for _ in ratings], 'Rating':ratings, 'Category':categories, 'Review':reviews}
        file_name = f'siksin_{keyword}'
        try:
            df = pd.DataFrame(data)
            df.to_parquet(f'{file_name}.parquet')
            print('Saved', df.shape[0], f'samples ({df.shape[0]/n_review*100:.2f}%)')  
        except ValueError:
            with open(f'{file_name}.json', 'w') as f:
                json.dump(data, f)
            print('Saved json instead of dataframe. Check for corruption in the data.')


    # get url
    url = f"https://www.siksinhot.com/search?keywords={keyword}"
    driver.get(url)

    # number of reviews in the page
    review_li = driver.find_element_by_xpath("//li[@data-tag='review']")
    review_a_text = review_li.find_element_by_xpath(".//a[@href='#']").text
    p = re.compile('\d') # find 0~9
    n_review = int(''.join(p.findall(review_a_text)))
    print("Number of reviews:", n_review)

    # find element containing all of the review stuff
    review_div = driver.find_element_by_css_selector("div.review_list")

    # iterate click to see more reviews
    i = 0
    while True:
        try:
            more_button = review_div.find_element_by_css_selector("a.btn_sMore")
            more_button.click()
            time.sleep(1)
            i += 1 
            if i % 200 == 0: # count available samples
                n_samples = len([x.text for x in review_div.find_elements_by_css_selector('p')])
                print(f'{i}th iteration completed. {n_samples} samples.')
            if i % SAVE_CHECKPOINT_ITER == 0: # checkpoint for saving df
                print("Arrived at checkpoint. Saving data...")
                save_df()

        except NoSuchElementException:
            print(f"Reached end of the reviews at {i}th iteration. Saving data...")
            save_df()
            driver.quit()
            break

        except:
            save_df()
            driver.quit()
            break

        if i > MAX_ITER:
            print("Reached max iteration. Saving data...")
            save_df()
            driver.quit()
            break