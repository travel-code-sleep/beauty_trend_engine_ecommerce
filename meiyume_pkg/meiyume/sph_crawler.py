
"""[summary]
"""
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
from .utils import Logger, Browser, MeiyumeException
import shutil

class Metadata(Browser):
    """[summary]

    Arguments:
        Browser {[type]} -- [description]
    """
    base_url="https://www.sephora.com"
    info = tldextract.extract(base_url)
    source = info.registered_domain

    @classmethod
    def update_base_url(cls, url):
        """[summary]

        Arguments:
            url {[type]} -- [description]
        """
        cls.base_url = url
        cls.info = tldextract.extract(cls.base_url)
        cls.source = cls.info.registered_domain

    def __init__(self, driver_path, logs=True, path = Path.cwd(), show=True):
        """
        [summary]

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
        self.curnet_progress_path = self.data_path/'current_progress'
        self.curnet_progress_path.mkdir(parents=True, exist_ok=True)
        if logs:
            self.prod_meta_log = Logger("sph_prod_metadata_extraction", path=self.data_path)
            self.logger, _ = self.prod_meta_log.start_log()

    def get_product_type_urls(self):
        """[summary]
        """
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
        df.to_feather(self.data_path/f'sph_product_type_urls_to_extract')
        return df

    def download_metadata(self,fresh_start):
        """[summary]
        
        Arguments:
            fresh_start {[type]} -- [description]
        """
        
        product_meta_data = []
       
        def fresh_ext():
            product_type_urls = self.get_product_type_urls()
            # progress tracker: captures scraped and error desc 
            progress_tracker = pd.DataFrame(index=product_type_urls.index, columns=['product_type', 'scraped', 'error_desc'])
            progress_tracker.scraped = 'N'
            return product_type_urls, progress_tracker
        
        if fresh_start:
            self.logger.info('Starting Fresh Extraction.')
            product_type_urls, progress_tracker = fresh_ext()
        else:
            if Path(self.data_path/'sph_product_type_urls_to_extract').exists():
                try:
                    progress_tracker = pd.read_feather(self.data_path/'sph_metadata_progress_tracker')
                except ArrowIOError:
                    raise MeiyumeException(f"File sph_product_type_urls_to_extract can't be located in the path {self.data_path}.\
                                             Please put progress file in the correct path or start fresh extraction.")
                product_type_urls = pd.read_feather(self.data_path/'sph_product_type_urls_to_extract')
                if sum(progress_tracker.scraped=='N')>0:
                    self.logger.info('Continuing Metadata Extraction From Last Run.')
                    product_type_urls = product_type_urls[product_type_urls.index.isin(progress_tracker.index[progress_tracker.scraped=='N'].values.tolist())]
                else:
                    self.logger.info('Previous Run Was Complete. Starting Fresh Extraction.')
                    product_type_urls, progress_tracker = fresh_ext()
            else:
                self.logger.info('URL File Not Found. Starting Fresh Extraction.')
                product_type_urls, progress_tracker = fresh_ext()
                
        drv  = self.create_driver(url=self.base_url)
        for pt in product_type_urls.index:
            cat_name = product_type_urls.loc[pt,'category_raw']
            #sub_cat_name = product_type_urls.loc[pt,'sub_category_raw']
            product_type = product_type_urls.loc[pt,'product_type']
            product_type_link = product_type_urls.loc[pt,'url']

            progress_tracker.loc[pt,'product_type'] = product_type

            # if 'best-selling' in product_type or 'new' in product_type:
            #     progress_tracker.loc[pt,'scraped'] = 'NA'
            #     continue
            
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
            self.scroll_down_page(drv)
            #check whether on the first page of product type
            try:
                current_page = drv.find_element_by_class_name('css-x544ax').text
            except NoSuchElementException:
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType {product_type} has\
                only one page of products.(page link: {product_type_link})', 'utf-8', 'ignore'))
                one_page = True
                current_page = 1
            except:
                product_type_urls.loc[pt,'scraped'] = 'NA'
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType {product_type} page not found.(page link: {product_type_link})', 'utf-8', 'ignore'))
            else:
                #get a list of all available pages
                one_page = False
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
                                                 Product: {product_name} - {cp} extracted successfully.\
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
                            self.scroll_down_page(drv)
                            current_page = drv.find_element_by_class_name('css-x544ax').text
                        except:
                            self.logger.info(str.encode(f'Page navigation issue occured for Category: {cat_name} - \
                                                          ProductType: {product_type} (page_link: {product_type_link} \
                                                          - page_no: {current_page})', 'utf-8', 'ignore'))
                            break

            if len(product_meta_data)>0:
                product_meta_df = pd.DataFrame(product_meta_data)
                product_meta_df.to_feather(self.curnet_progress_path/f'sph_prod_meta_extract_progress_{product_type}_{time.strftime("%Y-%m-%d-%H%M%S")}')
                self.logger.info(f'Completed till IndexPosition: {pt} - ProductType: {product_type}. (URL:{product_type_link})')
                progress_tracker.loc[pt,'scraped'] = 'Y'
                progress_tracker.to_feather(self.data_path/'sph_metadata_progress_tracker')
                product_meta_data = []
        self.logger.info('Metadata Extraction Complete')
        print('Metadata Extraction Complete')
        #self.progress_monitor.info('Metadata Extraction Complete')
        drv.close()

    def extract(self, fresh_start=False, delete_progress=True):
        """[summary]

        Keyword Arguments:
            fresh_start {bool} -- [description] (default: {False})
            delete_progress {bool} -- [description] (default: {True})
        Returns: 
        """
        self.download_metadata(fresh_start)
        self.logger.info('Creating Combined Metadata File')
        files = [f for f in self.curnet_progress_path.glob("sph_prod_meta_extract_progress_*")]
        li = [pd.read_feather(file) for file in files]
        metadata_df = pd.concat(li, axis=0, ignore_index=True)
        metadata_df.reset_index(inplace=True, drop=True)
        metadata_df.to_feather(self.data_path/f'sph_product_metadata_all_{time.strftime("%Y-%m-%d")}')
        self.logger.info(f'Metadata file created. Please look for file sph_product_metadata_all in path {self.data_path}')
        print(f'Metadata file created. Please look for file sph_product_metadata_all in path {self.data_path}')
        if delete_progress:
            print('Deleting Progress Files')
            shutil.rmtree(f'{data_path}\\current_progress', ignore_errors=True)
            self.logger.info('Progress files deleted')
        self.logger.handlers.clear()
        self.prod_meta_log.stop_log()
        return metadata_df


class Details(Browser):
    """
    [summary]

    Arguments:
        Browser {[type]} -- [description]
    """
    pass
    # define steps
    # first step
    # second step
    # third step