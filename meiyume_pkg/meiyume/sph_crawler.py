from pathlib import Path
import tldextract
import pandas as pd
import time
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from meiyume.utils import Logger, Browser


class Metadata(Browser):
    """ pass """
    base_url="https://www.sephora.com"
    info = tldextract.extract(base_url)
    source = info.registered_domain

    @classmethod
    def update_base_url(cls, url):
        cls.base_url = url
        cls.info = tldextract.extract(cls.base_url)
        cls.source = cls.info.registered_domain

    def __init__(self, driver_path, logs=True):
        super().__init__(driver_path)
        #self.url = base_url
        if logs:
            self.logger = Logger("sph_prod_metadata_extraction").set_log()
    
    def get_product_type_urls(self):
        """ pass """
        drv  = self.create_driver(url=self.base_url)
        cats = drv.find_elements_by_class_name("css-1t5gbpr")
        cat_urls = []
        for c in cats:
            cat_urls.append((c.get_attribute("href").split("/")[-1], c.get_attribute("href")))
            self.logger.info(str.encode(f'Category:- name:{c.get_attribute("href").split("/")[-1]} , url:{c.get_attribute("href")}', "utf-8", "ignore"))
        sub_cat_urls = []
        for cu in cat_urls: 
            cat_name = cu[0]
            cat_url = cu[1]
            drv.get(cat_url)
            time.sleep(5)
            sub_cats = drv.find_elements_by_class_name("css-16yq0cc")
            sub_cats.extend(drv.find_elements_by_class_name("css-or7ouu"))
            if len(sub_cats)>0:
                for s in sub_cats: 
                    sub_cat_urls.append((cat_name, s.get_attribute("href").split("/")[-1], s.get_attribute("href")))
                    self.logger.info(str.encode(f'SubCategory:- name:{s.get_attribute("href").split("/")[-1]} , url:{s.get_attribute("href")}', "utf-8", "ignore"))
            else:
                sub_cat_urls.append((cat_name, cat_url.split('/')[-1], cat_url))
        product_type_urls = []
        for su in sub_cat_urls:
            cat_name = su[0]
            sub_cat_name = su[1]
            sub_cat_url = su[2]
            drv.get(sub_cat_url)
            time.sleep(3)
            product_types = drv.find_elements_by_class_name('css-h6ss0r')
            if len(product_types)>0:
                for item in product_types:
                    product_type_urls.append((cat_name, sub_cat_name, item.get_attribute("href").split("/")[-1], item.get_attribute("href")))
                    self.logger.info(str.encode(f'ProductType:- name:{item.get_attribute("href").split("/")[-1]} , url:{item.get_attribute("href")}', "utf-8", "ignore"))
            else:
                product_type_urls.append((cat_name, sub_cat_name, sub_cat_url.split('/')[-1], sub_cat_url))
        df = pd.DataFrame(product_type_urls, columns = ['category_raw', 'sub_category_raw', 'product_type', 'item_url'])
        df.to_csv('sph_product_type_urls_to_extract.csv', index=None) 
        drv.close()       
        return product_type_urls

    def download_metadata(self):
        """ pass """
        product_meta_data = []
        product_type_urls = self.get_product_type_urls()

        drv  = self.create_driver(url=self.base_url)

        for pt in product_type_urls:
            cat_name = product_type_urls[0]
            sub_cat_name = product_type_urls[1]
            product_type = product_type_urls[2]
            product_type_link = product_type_urls[3]
            drv.get(product_type_link)
            time.sleep(5)
            #click and remove welcome forms 
            try:
                drv.find_element_by_xpath('/html/body/div[8]/div/div/div[1]/div/div/button').click()
            except:
                pass
            try:
                drv.find_element_by_xpath('/html/body/div[5]/div/div/div/div[1]/div/div/button').click()
            except:
                pass
            #load all the products
            self._scroll_down_page(drv)
            #check whether on the first page of product type 
            try:
                current_page = drv.find_element_by_class_name('css-x544ax').text
            except:
                self.logger.info(str.encode(f'Category: {cat_name} - SubCategory: {sub_cat_name} - ProductType {product_type} is not a top level page.(page link: {product_type_link})',
                                'utf-8', 'ignore'))
                continue
            #get a list of all avilable pages 
            pages =  []
            for page in drv.find_elements_by_class_name('css-1f9ivf5'):
                pages.append(page.text)
            #start getting product form each page 
            while True: 
                cp = 0
                self.logger.info(str.encode(f'Category: {cat_name} - SubCategory: {sub_cat_name} - ProductType: {product_type}\
                                  getting product from page {current_page}.(page link: {product_type_link})','utf-8', 'ignore')) 
                time.sleep(3)
                products = drv.find_elements_by_class_name('css-12egk0t')
                for p in products:
                    time.sleep(3)
                    try:
                       product_name = p.find_element_by_class_name('css-ix8km1').get_attribute('aria-label')
                    except NoSuchElementException or StaleElementReferenceException:
                        self.logger.info(str.encode(f'Category: {cat_name} - SubCategory: {sub_cat_name} - ProductType: {product_type} - product {products.index(p)} metadata extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                        continue
                    try:    
                        product_page = p.find_element_by_class_name('css-ix8km1').get_attribute('href')
                    except NoSuchElementException or StaleElementReferenceException:
                        product_page = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - SubCategory: {sub_cat_name} - ProductType: {product_type} - product {products.index(p)} product_page extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                    try:
                        brand = p.find_element_by_class_name('css-ktoumz').text
                    except NoSuchElementException or StaleElementReferenceException:
                        brand = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - SubCategory: {sub_cat_name} - ProductType: {product_type} - product {products.index(p)} brand extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                    try:
                        rating = p.find_element_by_class_name('css-1adflzz').get_attribute('aria-label')
                    except NoSuchElementException or StaleElementReferenceException:
                        rating = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - SubCategory: {sub_cat_name} - ProductType: {product_type} - product {products.index(p)} rating extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                    try:
                        price = p.find_element_by_class_name('css-68u28a').text
                    except NoSuchElementException or StaleElementReferenceException:
                        price = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - SubCategory: {sub_cat_name} - ProductType: {product_type} - product {products.index(p)} price extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                
                    d = {"product_name":product_name,"product_page":product_page,"brand":brand,"price":price,"rating":rating,"category":cat_name,"sub_category":sub_cat_name,                        "product_type": product_type,
                        "timestamp": time.strftime("%Y-%m-%d-%H-%M"),"complete_scrape_flag":"N"
                        }
                    cp += 1
                    self.logger.info(str.encode(f'Category: {cat_name} - Sub_Category: {sub_cat_name} - ProductType: {product_type} - Product: {product_name} - {cp} extracted sucessfully.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                    product_meta_data.append(d)
                if int(current_page) == int(pages[-1]):
                    self.logger.info(str.encode(f'Category: {cat_name} - Sub_Category: {sub_cat_name} - ProductType: {product_type} extraction complete.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                    break
                else:
                    try:
                        time.sleep(2)
                        drv.find_element_by_css_selector('body > div.css-o44is > div.css-138ub37 > div > div > div > div.css-1o80i28 > div > main > div.css-1aj5qq4 > div > div.css-1cepc9v > div.css-6su6fj > nav > ul > button').click()
                        time.sleep(3)
                        self._scroll_down_page(drv)
                        current_page = drv.find_element_by_class_name('css-x544ax').text
                    except:
                        break    
        self.logger.info('Metadata Extraction Complete')
        drv.close()
        return product_meta_data

    def extract(self):
        """ call the extraction functions here """
        product_meta_data = self.download_metadata()
        metadata_df = pd.DataFrame(product_meta_data)
        metadata_df.to_csv('sph_product_metadata_all.csv', index=None)
        return metadata_df
    

