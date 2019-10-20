from __future__ import print_function, absolute_import
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
    """
    summary

    Arguments:
        Browser {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    base_url="https://www.sephora.com"
    info = tldextract.extract(base_url)
    source = info.registered_domain

    @classmethod
    def update_base_url(cls, url):
        cls.base_url = url
        cls.info = tldextract.extract(cls.base_url)
        cls.source = cls.info.registered_domain

    def __init__(self, driver_path, logs=True, path = Path.cwd(), show=True):
        """[summary]
        
        Arguments:
            driver_path {[type]} -- [description]
        
        Keyword Arguments:
            logs {bool} -- [description] (default: {True})
            path {[type]} -- [description] (default: {Path.cwd()})
            show {bool} -- [description] (default: {True})
        """
        super().__init__(driver_path, show)
        self.path = Path(path)
        self.data_path = path/'sephora/metadata'
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.currnet_progress_path = self.data_path/'current_progress'
        self.currnet_progress_path.mkdir(parents=True, exist_ok=True)
        self.logs = logs
    
    def get_product_type_urls(self):
        """ pass """
        if self.logs:
            self.logger, self.log_file_name = Logger("sph_site_structure_url_extraction", path=self.data_path).start_log()
        drv  = self.create_driver(url=self.base_url)
        cats = drv.find_elements_by_class_name("css-1t5gbpr")
        cat_urls = []
        for c in cats:
            cat_urls.append((c.get_attribute("href").split("/")[-1], c.get_attribute("href")))
            self.logger.info(str.encode(f'Category:- name:{c.get_attribute("href").split("/")[-1]}, \
                                          url:{c.get_attribute("href")}', "utf-8", "ignore"))
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
                    self.logger.info(str.encode(f'SubCategory:- name:{s.get_attribute("href").split("/")[-1]},\
                                                  url:{s.get_attribute("href")}', "utf-8", "ignore"))
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
                    product_type_urls.append((cat_name, sub_cat_name, item.get_attribute("href").split("/")[-1], 
                                              item.get_attribute("href")))
                    self.logger.info(str.encode(f'ProductType:- name:{item.get_attribute("href").split("/")[-1]},\
                                                  url:{item.get_attribute("href")}', "utf-8", "ignore"))
            else:
                product_type_urls.append((cat_name, sub_cat_name, sub_cat_url.split('/')[-1], sub_cat_url))
        df = pd.DataFrame(product_type_urls, columns = ['category_raw', 'sub_category_raw', 'product_type', 'url'])
        df['scraped'] = 'N'
        df.to_feather(self.data_path/'sph_product_cat_subcat_structure') 
        drv.close() 
        df.drop_duplicates(subset='product_type', inplace=True)
        df.reset_index(inplace=True, drop=True)
        df.to_feather(self.data_path/'sph_product_type_urls_to_extract') 
        self.logger.handlers.clear()
        self.logger.stop_log()
        return df

    def download_metadata(self,fresh_start):
        """ pass """
        if self.logs:
            self.logger, self.log_file_name = Logger("sph_prod_metadata_extraction", path=self.data_path).start_log()

        product_meta_data = []  

        if fresh_start:
            product_type_urls = self.get_product_type_urls()
            self.logger.info('Starting Fresh Extraction.')
        else:
            if Path(self.data_path/'sph_product_type_urls_to_extract').exists():
                product_type_urls = pd.read_feather(self.data_path/'sph_product_type_urls_to_extract')
                if sum(product_type_urls.scraped=='N')>0:
                    self.logger.info('Continuing Metadata Extraction From Last Run.')         
                else:
                    product_type_urls = self.get_product_type_urls()
                    self.logger.info('Previous Run Was Complete. Starting Fresh Extraction.')

        drv  = self.create_driver(url=self.base_url)
        for pt in product_type_urls.index:
            scraped = product_type_urls.loc[pt,'scraped']
            if scraped in ['NA', 'Y']:
                continue
            cat_name = product_type_urls.loc[pt,'category_raw']
            #sub_cat_name = product_type_urls.loc[pt,'sub_category_raw']
            product_type = product_type_urls.loc[pt,'product_type']
            product_type_link = product_type_urls.loc[pt,'url']

            if 'best-selling' in product_type or 'new' in product_type:
                product_type_urls.loc[pt,'scraped'] = 'NA'
                continue
            drv.get(product_type_link)
            time.sleep(5)
            #click and close welcome forms 
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
            except NoSuchElementException:
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType {product_type} has\
                only one page of products.(page link: {product_type_link})', 'utf-8', 'ignore'))
                one_page = True 
                current_page = 1 
            else:
                #get a list of all avilable pages 
                pages =  []
                for page in drv.find_elements_by_class_name('css-1f9ivf5'):
                    pages.append(page.text)
            
            #start getting product form each page 
            while True: 
                cp = 0
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type}\
                                  getting product from page {current_page}.(page link: {product_type_link})','utf-8', 'ignore')) 
                time.sleep(3)
                products = drv.find_elements_by_class_name('css-12egk0t')
                for p in products:
                    time.sleep(3)
                    try:
                       product_name = p.find_element_by_class_name('css-ix8km1').get_attribute('aria-label')
                    except NoSuchElementException or StaleElementReferenceException:
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                     product {products.index(p)} metadata extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                        continue
                    try:    
                        product_page = p.find_element_by_class_name('css-ix8km1').get_attribute('href')
                    except NoSuchElementException or StaleElementReferenceException:
                        product_page = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                     product {products.index(p)} product_page extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                    try:
                        brand = p.find_element_by_class_name('css-ktoumz').text
                    except NoSuchElementException or StaleElementReferenceException:
                        brand = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                     product {products.index(p)} brand extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                    try:
                        rating = p.find_element_by_class_name('css-1adflzz').get_attribute('aria-label')
                    except NoSuchElementException or StaleElementReferenceException:
                        rating = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                     product {products.index(p)} rating extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                    try:
                        price = p.find_element_by_class_name('css-68u28a').text
                    except NoSuchElementException or StaleElementReferenceException:
                        price = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                      product {products.index(p)} price extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                
                    d = {"product_name":product_name,"product_page":product_page,"brand":brand,"price":price,"rating":rating,
                         "category":cat_name,"product_type": product_type, "timestamp": time.strftime("%Y-%m-%d-%H-%M"),"complete_scrape_flag":"N"}
                    cp += 1
                    self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                 Product: {product_name} - {cp} extracted sucessfully.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                    product_meta_data.append(d)
                
                if one_page:
                    break
                else:
                    if int(current_page) == int(pages[-1]):
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} extraction complete.\
                                                    (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                        break
                    else:
                        try:
                            time.sleep(2)
                            drv.find_element_by_css_selector('body > div.css-o44is > div.css-138ub37 > div > div > div >\
                                                            div.css-1o80i28 > div > main > div.css-1aj5qq4 > div > div.css-1cepc9v >\
                                                            div.css-6su6fj > nav > ul > button').click()
                            time.sleep(3)
                            self._scroll_down_page(drv)
                            current_page = drv.find_element_by_class_name('css-x544ax').text
                        except:
                            self.logger.info(str.encode(f'Page navigation issue occured for Category: {cat_name} - \
                                                          ProductType: {product_type} (page_link: {product_type_link} \
                                                          - page_no: {current_page})', 'utf-8', 'ignore'))
                            break  
                                 
            if len(product_meta_data)>0:
                product_meta_df = pd.DataFrame(product_meta_data)
                product_meta_df.to_feather(self.currnet_progress_path/f'sph_prod_meta_extract_progress_{time.strftime("%Y-%m-%d-%H%M%S")}')  
                self.logger.info(f'Completed till IndexPosition: {pt} - ProductType: {product_type}. (URL:{product_type_link})') 
                product_type_urls.loc[pt,'scraped'] = 'Y'
                product_type_urls.to_feather(self.data_path/'sph_product_type_urls_to_extract')
                product_meta_data = [] 
        self.logger.info('Metadata Extraction Complete')
        print('Metadata Extraction Complete')
        #self.progress_monitor.info('Metadata Extraction Complete')
        drv.close() 

    def extract(self, fresh_start=False):
        """ call the extraction functions here """
        self.download_metadata(fresh_start)
        self.logger.info('Creating Combined Metadata File')
        files = [f for f in self.currnet_progress_path.glob("sph_prod_meta_extract_progress_*")]
        li = [pd.read_feather(file) for file in files]
        metadata_df = pd.concat(li, axis=0, ignore_index=True)
        metadata_df.reset_index(inplace=True, drop=True)
        metadata_df.to_feather('sph_product_metadata_all')
        self.logger.info(f'Metadata file created. Please look for file sph_product_metadata_all in path {self.data_path}')
        self.logger.handlers.clear()
        self.logger.stop_log()
        print(f'Metadata file created. Please look for file sph_product_metadata_all in path {self.data_path}')
        return metadata_df
    

