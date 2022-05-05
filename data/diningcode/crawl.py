from selenium import webdriver
from selenium.common.exceptions import ElementNotInteractableException, TimeoutException, NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains 
import pandas as pd

# variables
START_QUERY = 0
SAVE_CHECKPOINT_QUERY = 20
MAX_RESTAURANT_ITER = 20
UNDESIRED_WAIT = 6

# driver setup
driver_path = '../../chromedriver'
options = webdriver.chrome.options.Options()
options.add_argument('--no-sandbox')
options.add_argument('--headless')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(driver_path, options=options)

# querys (using subway stations because maximum number of restaruants is limited to 100)
def load_querys():
    #['대전','청주','충주','세종','전주','광주','목포','여수','창원','통영','거제','울산','포항','경주','강릉','속초','양양','춘천','원주']
    return ['대구','제주','서귀포','천안','공주','나주','진주','김해','안동','구미','김천','제천','울진','논산','군산','순천','광양']
    

# function to save df
def save_df(file_name):
    data = {'Region':[query for _ in ratings], 'Rating':ratings, 'Category':categories, 'Review':reviews}
    file_name = f'diningcode_jibang2'
    df = pd.DataFrame(data)
    df.to_parquet(f'{file_name}.parquet')
    print('Saved', df.shape[0], 'samples') 

# scrape
style2num = {f'width: {20*x}%;':x for x in range(0,6)} 
reviews = []
ratings = []
categories = []
querys = load_querys()

for i, query in enumerate(querys):
    # starting from the middle (when the first queries are already used)
    if i < START_QUERY:
        continue

    # 1. href for each restaurant
    print(f"Query {i}:", query)
    url = f'https://www.diningcode.com/list.php?query={query}'
    driver.get(url)

    ## validate the url (if query is not a common region name, then url need to be modified)
    try:
        validate = driver.find_element_by_css_selector("#div_list > li:nth-child(4) > a > span.btxt").text
        if validate == '1. 명동교자 본점':
            url = url + '&rn=1'
            driver.get(url)
            print("Search by restaurant name instead of region.")
    except NoSuchElementException:
        pass


    ## load more (maximum number of restaurants is 100 for each query)
    notInteractable_count = 0
    for _ in range(MAX_RESTAURANT_ITER):
        try:
            more_restaurant = WebDriverWait(driver, 2).until(EC.element_to_be_clickable((By.ID, "div_list_more")))
            ActionChains(driver).move_to_element(more_restaurant).click(more_restaurant).perform()
        except TimeoutException: # desired exception
            print(f"End of restaurants.", end=' ')
            break
        except ElementNotInteractableException: # acceptable exception if used with loop
            continue
        
    ## get hrefs
    hrefs = []
    for li in driver.find_elements_by_css_selector('#div_list > li'):
        if li.get_attribute("onmouseenter"): # exclude ads
            hrefs.append(li.find_element_by_tag_name('a').get_attribute('href'))
    print(f"Scraped {len(hrefs)} hrefs.")


    # 2. scrape reviews in each restaurant
    for j, href in enumerate(hrefs):
        print(f'{j+1}th restaurant: ', end='')
        try:
            driver.get(href)
        except TimeoutException:
            print("Failed to get href. Moving on.")

        # load more
        while True:
            try:
                more_review = WebDriverWait(driver, 2).until(EC.element_to_be_clickable((By.ID, "div_more_review")))
                ActionChains(driver).move_to_element(more_review).click(more_review).perform()
            except TimeoutException: # desired exception; meanning the end of reviews
                break

        # reviews and ratings
        try:
            review_elem = driver.find_elements_by_css_selector("p.review_contents.btxt")
            ratings_elem = driver.find_elements_by_css_selector('i.star > i')
            # review_elem = WebDriverWait(driver, UNDESIRED_WAIT).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "p.review_contents.btxt")))
            # ratings_elem = WebDriverWait(driver, UNDESIRED_WAIT).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "i.star > i")))
            new_reviews = [x.text for x in review_elem]
            new_ratings =  [style2num[x.get_attribute('style')] for x in ratings_elem]
        # except TimeoutException: # undesired exception
        #     print("Cannot scrape reviews or ratings")
        #     continue
        except NoSuchElementException: # undesired exception
            print("Cannot scrape reviews or ratings")
            continue

        # category, which is consistent within one href(restaurant) 
        try:
            div_btxt = WebDriverWait(driver, UNDESIRED_WAIT).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#div_profile > div.s-list.pic-grade > div.btxt")))
            category_sep = [a.text for a in div_btxt.find_elements_by_tag_name('a')]
            category = ' '.join(category_sep)
        except TimeoutException: # undesired exception
            print("(Cannot scrape category)", end=' ')
            category = ''
        new_categories = [category for _ in new_ratings]

        # verfiy integrity and add data
        if len(new_reviews) == len(new_ratings):
            reviews.extend(new_reviews)
            ratings.extend(new_ratings) 
            categories.extend(new_categories)
            print(len(new_reviews), 'samples added')
        else:
            print('Discard corrupted data')

    # checkpoint to save data
    if (i+1) % SAVE_CHECKPOINT_QUERY == 0:
        save_df(str(START_QUERY) + '-' + str(i))

save_df(str(START_QUERY) + '-' + str(i))
driver.quit()