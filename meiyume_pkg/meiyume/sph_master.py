import datetime
import logging
import time
import pandas as pd
import numpy as np
import os

import ast
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly as ply
import re
from ast import literal_eval
import seaborn as sns
plt.style.use('fivethirtyeight')
np.random.seed(42)

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc,accuracy_score,average_precision_score,\
recall_score, roc_auc_score, roc_curve, recall_score, classification_report, f1_score, precision_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from .utils import nan_equal, show_missing_value
np.random.seed(42)

class MeiyumeException(Exception):
    """class to define custom exceptions in runtime"""
    pass

class Meiyume(object):
    """parent class. the methods in this class will
       be inherited by all the child classes. 
       define global methods only"""    
    def __init__(self):
        self.nan_equal = nan_equal
        self.show_missing_value = show_missing_value

    
class Cleaner(Meiyume): ## add category translation to meta data cleaning 
    """ pass """
    def __init__(self, file_name, file_path=os.getcwd()):
        """pass"""
        super(Cleaner, self).__init__()
        self.file_name = file_name
        self.file_path = file_path
        self.file_type = file_name.split('.')[-1].lower()
        
        if 'sephora' in self.file_name.lower() or 'sph' in  self.file_name.lower():
            if 'meta' in self.file_name.lower():
                self.data_defintion = 'sph_meta'
            elif 'detail' in self.file_name.lower():
                self.data_defintion = 'sph_detail'
            elif 'item' in self.file_name.lower():
                self.data_defintion = 'sph_item'
            elif 'review' in self.file_name.lower():
                self.data_defintion = 'sph_review'
        else: 
            raise MeiyumeException("Unable to determine data definition. Please provide correct file names.")
                
        self.file = os.path.join(file_path, file_name)
                
    def read_data(self):
        """pass"""
        if self.file_type == 'xlsx':
            return pd.read_excel(self.file)
        elif self.file_type == 'csv':
            return pd.read_csv(self.file)
    
    def clean_data(self, return_type='no_cats'):
        """pass"""
        if self.data_defintion == 'sph_meta':
            cleaned_metadata = self.sph_meta_cleaner(return_type)
            return cleaned_metadata
        if self.data_defintion == 'sph_detail':
            cleaned_detail = self.sph_detail_cleaner(return_type='None')
            return cleaned_detail
        if self.data_defintion == 'sph_review':
            cleaned_review = self.sph_review_cleaner()
            return cleaned_review
        
    
    def sph_review_cleaner(self):
        """pass"""
        self.sph_review = self.read_data()
        #remove blank review and date fields
        self.sph_review = self.sph_review[~self.sph_review.review.isna()] 
        self.sph_review = self.sph_review[~self.sph_review.review_date.isna()]
        self.sph_review.reset_index(inplace=True, drop=True)
        
        #clean and separate helpful
        self.sph_review.helpful[self.sph_review.helpful.astype(str)== "['', '']"] = np.nan
        
        def helpful_split(x, extract):
            if not self.nan_equal(x,np.nan):
                if extract == 'helpful':
                    helpful = literal_eval(x)[1].split('(')[-1].replace(')','')
                    return helpful
                elif extract == 'not_helpful':
                    not_helpful = literal_eval(x)[0].split('(')[-1].replace(')','')
                    return not_helpful
            else:
                return np.nan
        self.sph_review['helpful_Y'] = self.sph_review.helpful.apply(lambda x: helpful_split(x,extract='helpful'))
        self.sph_review['helpful_N'] = self.sph_review.helpful.apply(lambda x: helpful_split(x,extract='not_helpful'))
        self.sph_review.helpful_Y[self.sph_review.helpful_Y.isna()] = 0
        self.sph_review.helpful_N[self.sph_review.helpful_N.isna()] = 0
        self.sph_review.helpful_Y[self.sph_review.helpful_Y ==''] = 0
        self.sph_review.helpful_N[self.sph_review.helpful_N ==''] = 0
        self.sph_review.drop('helpful', inplace=True, axis=1)
        self.sph_review.helpful_Y = self.sph_review.helpful_Y.astype(int)
        self.sph_review.helpful_N = self.sph_review.helpful_N.astype(int)
        
        #convert ratings to numbers
        rem_rat = re.compile('(\s*)stars|star|No(\s*)')
        self.sph_review.review_rating = self.sph_review.review_rating.apply(lambda x: rem_rat.sub('',x))
        self.sph_review.review_rating = self.sph_review.review_rating.astype(int)
        
        #clean date and convrt to pandas datetime¶
        def convert_to_date(x):
            if 'ago' in x.lower():
                if 'd' in x.lower():
                    days = 10 + int(x.split()[0])
                    date = datetime.today() - timedelta(days=days)
                    return date.strftime('%d %b %Y')
                elif 'm' in x.lower():
                    mins = int(x.split()[0])
                    date = datetime.today() - timedelta(days=10) - timedelta(minutes=mins)
                    return date.strftime('%d %b %Y')
                elif 'h' in x.lower():
                    hours = int(x.split()[0])
                    date = datetime.today() - timedelta(days=10) - timedelta(hours=hours)
                    return date.strftime('%d %b %Y')
            else:
                return x
        self.sph_review.review_date = self.sph_review.review_date.apply(convert_to_date)
        self.sph_review.review_date = pd.to_datetime(self.sph_review.review_date, infer_datetime_format=True)
        
        #clean and convert recommendation
        #### if rating is 5 then it is assumed that the person recommends
        #### id rating is 1 or 2 then it is assumed that the person does not recomend
        #### for all the other cases data is not available
        self.sph_review.recommend[(self.sph_review.recommend=='Recommends this product') | (self.sph_review.review_rating==5)] = 'Yes'
        self.sph_review.recommend[(self.sph_review.recommend!='Yes') & (self.sph_review.review_rating.isin([1,2]))] = 'No'
        self.sph_review.recommend[(self.sph_review.recommend!='Yes') & (self.sph_review.review_rating.isin([3,4]))] = 'not_avlbl'
        
        #clean and separate user attributes
        self.sph_review.user_attribute[self.sph_review.user_attribute.isin(["['', '', '', '']", "['', '', '']","[]"])] = np.nan
        # this part of the code depends on multi-level regular expression search and quite slower 
        def separate_user_attributes(x):
            """pass"""
            rmv = re.compile("(\s*)]|\(|\)|'|,(\s*)")
            if not self.nan_equal(x, np.nan):
                age_srch = re.search('age(.+?),', x.lower())
                if age_srch is None:
                    age_srch = re.search('age(.+?)]', x.lower())
                if age_srch:
                    age = age_srch.group(0)
                    age = rmv.sub('', age.split()[-1])
                else: 
                    age = 'not_avlbl'

                eye_srch = re.search('eye color(.+?),', x.lower())
                if eye_srch is None:
                    eye_srch = re.search('eye color(.+?)]', x.lower())
                if eye_srch:
                    eye_color = eye_srch.group(0)
                    eye_color = rmv.sub('', eye_color.split('color')[-1])
                else:
                    eye_color = 'not_avlbl'

                hair_col_srch = re.search('hair color(.+?),', x.lower())
                if hair_col_srch is None:
                    hair_col_srch = re.search('hair color(.+?)]', x.lower())
                if hair_col_srch:
                    hair_color = hair_col_srch.group(0)
                    hair_color = rmv.sub('', hair_color.split('color')[-1])
                else:
                    hair_color = 'not_avlbl'

                hair_cond_srch = re.search('hair condition(.+?),', x.lower())
                if hair_cond_srch is None:
                    hair_cond_srch = re.search('hair condition(.+?)]', x.lower())
                if hair_cond_srch:
                    hair_cond = hair_cond_srch.group(0)
                    hair_cond = rmv.sub('', hair_cond.split('condition')[-1])
                else:
                    hair_cond = 'not_avlbl'

                skin_tone_srch = re.search('skin tone(.+?),', x.lower())
                if skin_tone_srch is None:
                    skin_tone_srch = re.search('skin tone(.+?)]', x.lower())
                if skin_tone_srch:
                    skin_tone = skin_tone_srch.group(0)
                    skin_tone = rmv.sub('', skin_tone.split('tone')[-1])
                else:
                    skin_tone = 'not_avlbl'

                skin_type_srch = re.search('skin type(.+?)]', x.lower())
                if skin_type_srch is None:
                    skin_type_srch = re.search('skin type(.+?),', x.lower())
                if skin_type_srch:
                    skin_type = skin_type_srch.group(0)
                    skin_type = rmv.sub('', skin_type.split('type')[-1])
                else:
                    skin_type = 'not_avlbl'

                return({'age':age, 'eye_color':eye_color, 'hair_color':hair_color, 
                    'hair_cond':hair_cond, 'skin_tone':skin_tone, 'skin_type':skin_type})
            else:
                return({'age':'not_avlbl', 'eye_color':'not_avlbl', 'hair_color':'not_avlbl', 
                    'hair_cond':'not_avlbl', 'skin_tone':'not_avlbl', 'skin_type':'not_avlbl'}) 
        self.sph_review['user_attrs'] = self.sph_review.user_attribute.apply(separate_user_attributes)
        user_attrs = pd.DataFrame(self.sph_review.user_attrs.values.tolist())
        self.sph_review = pd.concat([self.sph_review, user_attrs], axis=1)
        self.sph_review.drop(['user_attribute', 'user_attrs'], axis=1, inplace=True)
        self.sph_review.reset_index(drop=True, inplace=True)
        return self.sph_review       
        
        
    def sph_detail_cleaner(self, return_type):
        """pass"""
        self.sph_detail = self.read_data()
        #self.sph_detail.drop_duplicates(subset='sephora_product_id', inplace=True) ##uncomment once the new detail with id is downloaded and augmented
        self.sph_detail.drop('product_attributes', inplace=True, axis=1)
        self.sph_detail.reset_index(drop=True, inplace=True)
        
        def convert_votes_to_number(x):
            """pass"""
            if not self.nan_equal(np.nan, x):
                if 'K' in x:
                     return int(x.replace('K',''))*1000
                else:
                    return x.replace('K','')
            else:
                return np.nan
            
        self.sph_detail.votes_count = self.sph_detail.votes_count.apply(lambda x: convert_votes_to_number(x))
        
        import ast
        def split_rating_dist(x, stars):
            if x is not np.nan:
                ratings = ast.literal_eval(x)
                if stars == 5:
                    return ratings[1]            
                elif stars == 4:
                    return ratings[3]
                elif stars == 3:
                    return ratings[5]
                elif stars == 2:
                    return ratings[7]
                elif stars == 1:
                    return ratings[9]
            else:
                return np.nan
        self.sph_detail['five_star'] = self.sph_detail.rating_distribution.apply(lambda x: split_rating_dist(x,5))
        self.sph_detail['four_star'] = self.sph_detail.rating_distribution.apply(lambda x: split_rating_dist(x,4))
        self.sph_detail['three_star'] = self.sph_detail.rating_distribution.apply(lambda x: split_rating_dist(x,3))
        self.sph_detail['two_star'] = self.sph_detail.rating_distribution.apply(lambda x: split_rating_dist(x,2))
        self.sph_detail['one_star'] = self.sph_detail.rating_distribution.apply(lambda x: split_rating_dist(x,1))
        self.sph_detail.drop('rating_distribution', axis=1, inplace=True)
        
        self.sph_detail.would_recommend = self.sph_detail.would_recommend.str.replace('%','').astype(float)
        self.sph_detail.rename({'would_recommend':'would_recommend_percentage'}, inplace=True, axis=1)
               
        
        if return_type == 'None':
            return self.sph_detail
        #else:
         #   return self.sph_meta
    
    def calculate_ratings(self, x):
            """pass"""
            if self.nan_equal(np.nan, x['ratings']):
                return (x['five_star']*5 + x['four_star']*4 + x['three_star']*3 + x['two_star']*2 + x['one_star'])\
                        /(x['five_star'] + x['four_star'] + x['three_star'] + x['two_star'] + x['one_star'])
            else:
                return x['ratings']
            #self.sph_meta.ratings =  round(self.sph_detail.apply(lambda x: calculate_ratings(x), axis=1),1)
            
    def sph_meta_cleaner(self, return_type):
        """pass"""
        self.sph_meta = self.read_data()
        #self.sph_meta.drop_duplicates(inplace=True)
        #self.sph_meta.reset_index(drop=True, inplace=True)
        self.sph_meta['price'] = self.sph_meta['price'].str.replace('$', '')
        self.sph_meta['price'] = self.sph_meta['price'].str.replace('(', '/ ')
        self.sph_meta['price'] = self.sph_meta['price'].str.replace(')', '')
        self.sph_meta['price'] = self.sph_meta['price'].str.replace('value','')
    
        sephora_product_ids = self.sph_meta.product_page.str.split(':', expand=True)
        sephora_product_ids.columns = ['a', 'b', 'id']
        self.sph_meta['sephora_product_id'] = sephora_product_ids.id
        
        rem_rat = re.compile('(\s*)stars|star|No(\s*)')
        self.sph_meta.ratings = self.sph_meta.ratings.apply(lambda x: rem_rat.sub('',x))
        self.sph_meta.ratings[self.sph_meta.ratings==''] = np.nan
        
        low_price = []
        high_price = []
        actual_price = []
        self.sph_meta_missing_price = self.sph_meta[self.sph_meta.price.isna()]
        self.sph_meta_missing_price.drop_duplicates(inplace=True)
        self.sph_meta_missing_price.reset_index(drop=True, inplace=True)
        
        self.sph_meta = self.sph_meta[~self.sph_meta.price.isna()]
        for i in self.sph_meta.price[:]:
            if '/' in i:
                prices = i.split('/')
                low_price.append(prices[0])
                actual_price.append(prices[-1])
                high_price.append(np.nan)
            elif '-' in i:
                prices = i.split('-')
                low_price.append(prices[0])
                high_price.append(prices[-1])  
                actual_price.append(np.nan)
            else:
                low_price.append(i)
                high_price.append(np.nan)
                actual_price.append(np.nan)
        prices = pd.DataFrame(list(zip(low_price, high_price, actual_price)), columns=['low_price', 'high_price', 'actual_price'])
        prices.reset_index(drop=True, inplace=True)
        self.sph_meta.reset_index(drop=True, inplace=True)
        self.sph_meta = pd.concat([self.sph_meta, prices], axis=1)
        self.sph_meta.drop('price', axis=1, inplace=True)
        
        self.sph_meta_no_cats = self.sph_meta.loc[:, self.sph_meta.columns.difference(['category', 'sub_category'])]     
        self.sph_meta_no_cats.drop_duplicates(subset='sephora_product_id', inplace=True)
        self.sph_meta_no_cats.reset_index(drop=True, inplace=True)
        
        self.sph_meta.reset_index(drop=True, inplace=True)
        
        if return_type == 'no_cats':
            return self.sph_meta_no_cats
        else:
            return self.sph_meta
        
class DataPrep(Meiyume):
    """pass"""
    def __init__(self, meta_file=None, detail_file=None, review_file=None, file_path=os.getcwd(), 
                 clean_data={'meta':True, 'detail':True, 'review':True}):
        super(DataPrep, self).__init__()
        self.clean_data = clean_data
        if meta_file is not None:
            self.meta_file = Cleaner(file_name=meta_file) # don't need cleaner objet in ranker once clean data is pushed to database and 
            if clean_data['meta']:
                 self.meta = self.meta_file.clean_data() # a combined table for both meta and details in the final storage area while two datasets in the table connections are used 
            else:
                self.meta = self.meta_file.read_data()
        if detail_file is not None:
            self.detail_file = Cleaner(file_name=detail_file) # for meta and details data fetching or staging period to be used in cleaner object
            if clean_data['detail']:                    
                self.detail = self.detail_file.clean_data() 
            else:
                self.detail = self.detail_file.read_data()   
        if review_file is not None:
            self.review_file = Cleaner(file_name=review_file)
            if clean_data['review']:
                self.review = self.review_file.clean_data()
            else:
                self.review = self.review_file.read_data()         
                   
    def prepare_data_for_ranker(self):
        """pass"""
        vc = self.detail.product_name.value_counts() # remove or align all these code after second level verification with 
        vc = pd.DataFrame(vc[vc>1])                   # newly downloaded details data 
        vc.reset_index(inplace=True)                  # new details data need a sanity check for about the brand 
        vc.columns = ['name', 'count']                # and how to use. might need to borrow data from old download
        
        names = vc.name.values.tolist()
        self.meta = self.meta[~self.meta.product_name.isin(names)]
        self.detail = self.detail[~self.detail.product_name.isin(names)]
        self.meta.set_index('product_name', inplace=True)
        self.detail.set_index('product_name', inplace=True)
        self.meta_detail = self.meta.join(self.detail, how='inner')
        self.meta.reset_index(inplace=True)
        self.detail.reset_index(inplace=True)
        self.meta_detail.reset_index(inplace=True)
        self.meta_detail = self.meta_detail[self.meta_detail.low_price.apply(len) < 9] # for the time being removing two or three # prices value in lower price
        
        self.cat = self.meta_file.sph_meta[['sephora_product_id','category','sub_category']].groupby('sephora_product_id').first().reset_index()
        self.cat.set_index('sephora_product_id', inplace=True)
        self.meta_detail.set_index('sephora_product_id', inplace=True)
        self.meta_detail = self.meta_detail.join(self.cat, how='left', rsuffix='_right')
        self.meta_detail.reset_index(inplace=True)
        self.cat.reset_index(inplace=True)
        
        self.meta_detail_rank = self.meta_detail[['ratings', 'reviews_count', 'votes_count', # next step split the value into a list and # take the lowest integer value as current price 
                                          'would_recommend_percentage', 'low_price', 'high_price',
                                          'actual_price', 'five_star', 'four_star', 'three_star', 
                                          'two_star', 'one_star']].apply(pd.to_numeric)     
        
        self.meta_detail_rank.reset_index(inplace=True, drop=True)
        self.meta_detail.reset_index(inplace=True, drop=True)
        
        self.meta_detail_rank['product_name'] = self.meta_detail.product_name
        self.meta_detail_rank['sephora_product_id'] = self.meta_detail.sephora_product_id
        self.meta_detail_rank['brand'] = self.meta_detail.brand
        self.meta_detail_rank['category'] = self.meta_detail.category
        self.meta_detail_rank['sub_category'] =  self.meta_detail.sub_category
        
        def calculate_ratings(x):
            """pass"""
            if self.meta_file.nan_equal(np.nan, x['ratings']):
                return (x['five_star']*5 + x['four_star']*4 + x['three_star']*3 + x['two_star']*2 + x['one_star'])\
                        /(x['five_star'] + x['four_star'] + x['three_star'] + x['two_star'] + x['one_star'])
            else:
                return x['ratings']
            
        self.meta_detail_rank.ratings =  round(self.meta_detail_rank.apply(lambda x: calculate_ratings(x), axis=1),1)
        self.meta_detail_rank = self.meta_detail_rank[~self.meta_detail_rank.ratings.isna()]
        self.meta_detail_rank = self.meta_detail_rank[~self.meta_detail_rank.reviews_count.isna()]
        self.meta_detail_rank.reset_index(inplace=True, drop=True)
        
        return self.meta_detail_rank 

    def prepare_data_for_review_data_analysis(self):
        """pass"""
        if self.clean_data['review']:
            _ = self.prepare_data_for_ranker()
            cat_brand = self.meta_detail[['brand', 'product_name', 'category', 'sub_category', 'low_price']]
            self.review.set_index('product_name', inplace=True)
            cat_brand.set_index('product_name', inplace=True)
            self.review = self.review.join(cat_brand, how='inner')
            cat_brand.reset_index(inplace=True)
            self.review.reset_index(inplace=True)        
            #convert price to float
            self.review.low_price = self.review.low_price.str.replace(' ','')
            self.review = self.review[self.review.low_price.apply(len) != 0]
            self.review.low_price = self.review.low_price.astype(float)
            return self.review   
        else:
            return self.review      

    def prepare_data_for_review_classification_first_stage(self):
        """pass"""
        #combine title and review where titel exists
        self.review = self.prepare_data_for_review_data_analysis()
        def combine_title_review(x):
            if not self.review_file.nan_equal(x.review_title, np.nan):
                full_review = x.review_title + " " + x.review
                return full_review
            else:
                return x.review
        self.review.review = self.review.apply(lambda x: combine_title_review(x), axis=1)

        #Select a Good Repesentative Unlabelled Sephora Test Set
        product_rating_count = self.review.groupby('product_name').review_rating.value_counts().rename(columns={'review_rating':'rating'}).reset_index()
        product_rating_count.columns =['product_name', 'review_rating', 'count']
        product_name_counts = pd.DataFrame(product_rating_count.product_name.value_counts())
        product_name_counts.reset_index(inplace=True)
        product_name_counts.columns = ['product_name', 'name_count']
        product_name_counts.set_index('product_name', inplace=True)
        self.review.set_index('product_name', inplace=True)
        self.review = self.review.join(product_name_counts)
        self.review.reset_index(inplace=True)
        self.review.review_date = pd.to_datetime(self.review.review_date, infer_datetime_format=True)
        
        np.random.seed(42)
        self.unlabelled_chosen = self.review[self.review.name_count==5].sample(500000, random_state=42)
        self.unlabelled_not_chosen = self.review[~self.review.review.isin(self.unlabelled_chosen.review.values.tolist())]
        return self.unlabelled_chosen


    def prepare_data_for_review_classification_second_stage(self, review_data=None):
        """pass"""
        self.unlabelled_not_chosen.product_name = self.unlabelled_not_chosen.product_name.astype(str)
        self.unlabelled_not_chosen.recommend = self.unlabelled_not_chosen.recommend.astype(str)
        self.unlabelled_not_chosen.review =self.unlabelled_not_chosen.review.astype(str)
        self.unlabelled_not_chosen.review_rating = self.unlabelled_not_chosen.review_rating.astype(int)
        self.unlabelled_not_chosen.review_title = self.unlabelled_not_chosen.review_title.astype(str) 
        self.unlabelled_not_chosen.helpful_Y = self.unlabelled_not_chosen.helpful_Y.astype(int)
        self.unlabelled_not_chosen.helpful_N = self.unlabelled_not_chosen.helpful_N.astype(int)
        self.unlabelled_not_chosen.age = self.unlabelled_not_chosen.age.astype(str)
        self.unlabelled_not_chosen.eye_color = self.unlabelled_not_chosen.eye_color.astype(str)
        self.unlabelled_not_chosen.hair_color = self.unlabelled_not_chosen.hair_color.astype(str)
        self.unlabelled_not_chosen.hair_cond = self.unlabelled_not_chosen.hair_cond.astype(str)
        self.unlabelled_not_chosen.skin_tone = self.unlabelled_not_chosen.skin_tone.astype(str)
        self.unlabelled_not_chosen.skin_type = self.unlabelled_not_chosen.skin_type.astype(str)
        self.unlabelled_not_chosen.brand = self.unlabelled_not_chosen.brand.astype(str)
        self.unlabelled_not_chosen.category = self.unlabelled_not_chosen.category.astype(str)
        self.unlabelled_not_chosen.sub_category = self.unlabelled_not_chosen.sub_category.astype(str)
        self.unlabelled_not_chosen.low_price =self.unlabelled_not_chosen.low_price.astype(int)
        self.unlabelled_not_chosen.name_count = self.unlabelled_not_chosen.name_count.astype(int)
        return self.unlabelled_not_chosen

class TextDataViz(Meiyume):
    """pass"""

class Ranker(Meiyume):
    """pass"""
    def __init__(self, meta_file, detail_file, file_path=os.getcwd()):
        super(Ranker, self).__init__()  
        self.meta_file = meta_file  
        self.detail_file = detail_file  
        self.file_path = file_path
    
    def bayesian_ranker(self):
        """pass"""
        data_prep = DataPrep(self.meta_file, self.detail_file, file_path=self.file_path)
        self.meta_detail_rank = data_prep.prepare_data_for_ranker()
        self.review_conf_no = self.meta_detail_rank.groupby(by=['category','sub_category'])['reviews_count'].mean().reset_index()
        self.prior_rating = self.meta_detail_rank.groupby(by=['category','sub_category'])['ratings'].mean().reset_index()
        
        def total_stars (x):
            return x.reviews_count * x.ratings

        def bayesian_estimate (x):
            c = int(round(self.review_conf_no['reviews_count'][(self.review_conf_no.category==x.category) & (self.review_conf_no.sub_category==x.sub_category)].values[0]))
            prior = int(round(self.prior_rating['ratings'][(self.prior_rating.category==x.category) & (self.prior_rating.sub_category==x.sub_category)].values[0]))
            return (c * prior + x.ratings * x.reviews_count	) / (c + x.reviews_count)
        
        self.meta_detail_rank['total_star'] = self.meta_detail_rank.apply(lambda x: total_stars(x), axis=1).reset_index(drop=True)
        self.meta_detail_rank['bayesian_estimate'] = self.meta_detail_rank.apply(bayesian_estimate, axis=1)
        self.meta_detail_rank.reset_index(drop=True,inplace=True)
        
        def ratio(x, which='+ve-ve'):
            """pass"""
            if which == '+ve-ve':
                return ((x.five_star + x.four_star)+1) / ((x.two_star+1 + x.one_star+1)+1)
            elif which == '+ve_total':
                return (x.five_star + x.four_star) / (x.reviews_count)
            
        self.meta_detail_rank['postive_to_negative_star_ratio'] = self.meta_detail_rank.apply(lambda x: ratio(x), axis=1)
        self.meta_detail_rank['postive_to_total_ratio'] = self.meta_detail_rank.apply(lambda x: ratio(x, which='+ve_total'), axis=1)
        
        self.meta_detail_rank = self.meta_detail_rank[['brand','sephora_product_id','product_name','low_price','category','sub_category',
                                                       'bayesian_estimate','ratings','reviews_count','votes_count','total_star',
                                                       'postive_to_negative_star_ratio','postive_to_total_ratio','five_star',
                                                       'four_star','three_star','two_star','one_star','high_price', 'actual_price',
                                                       'would_recommend_percentage']]
        self.meta_detail_rank.sort_values(by=['category','sub_category','bayesian_estimate','postive_to_negative_star_ratio',
                                              'postive_to_total_ratio','ratings','reviews_count','votes_count','total_star'],
                                          ascending=False, inplace=True)
        self.meta_detail_rank.reset_index(drop=True, inplace=True)
        
        self.meta_detail_rank_by_cat = self.meta_detail_rank.groupby(by=['category','sub_category'])
        self.grouped_ranked_description_by_cat = self.meta_detail_rank_by_cat.describe()
        
        return self.meta_detail_rank 

class BinaryPerfMetrics(Meiyume):
    '''Custom class that creates an object using model, data features and label'''
    def __init__(self,model,features,label, model_module='sklearn', y_pred=None, y_prob=None):
        self.model = model
        self.features = features
        self.label = label
        self.model_module = model_module
        if y_pred is not None:
            self.y_pred = y_pred
        if y_prob is not None:
            self.y_prob = y_prob

    def getModel(self):
        return self.model
    def getFeatures(self):
        return self.features
    def getLabel(self):
        return self.label
    def __str__(self):
        return 'Model '+ str(self.model) + '\n\n' + 'features\n'+ str(pd.DataFrame(self.features))\
                 + '\n\nLabel \n' + str(pd.DataFrame(self.label)) 
    
    def pred_prob(self,k=3,method=None):
        """pass"""
        if self.model_module == 'sklearn':
            if method=='cross_val':
                y_pred = cross_val_predict(self.model,self.features,self.label.ravel(),cv=k)
                y_prob = cross_val_predict(self.model,self.features,self.label.ravel(),cv=k, method='predict_proba')  
                self.scores = cross_val_score(estimator=self.model,X=self.features,y=self.label.ravel(),cv=k)
                return y_pred, y_prob, self.scores
            else:
                y_pred = self.model.predict(self.features)
                y_prob = self.model.predict_proba(self.features)
                return y_pred, y_prob
        elif self.model_module == 'bert':
            return self.y_pred, self.y_prob
    
    def plot_prec_recall_vs_tresh(self,precisions, recalls, thresholds):
        """pass"""
        plt.plot(thresholds, precisions[:-1], 'b--', label='precision')
        plt.plot(thresholds, recalls[:-1], 'g--', label = 'recall')
        plt.xlabel('Threshold')
        plt.legend(loc='upper left')
        plt.ylim([0,1])

    def show_confusion(self,pred):
        """pass"""
        self.cm = confusion_matrix(self.label,pred)
        ax= plt.subplot()
        sns.heatmap(self.cm, annot=True, ax = ax, fmt=".1f"); #annot=True to annotate cells
        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['False', 'True']); ax.yaxis.set_ticklabels(['False', 'True']);
        
    def show_roc(self,prob):
        """pass"""
        if self.model_module == 'bert':
            fpr, tpr, threshold = roc_curve(self.label, prob)
        else:
            fpr, tpr, threshold = roc_curve(self.label, prob[:,1])
        self.roc_auc = auc(fpr, tpr)
        #Plot ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % self.roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right"); plt.show()
        return self.roc_auc
    
    def show_prec_rec(self, roc_auc, prob, pred):
        """pass"""
        self.recall = recall_score(self.label, pred)
        print('recall', self.recall)
        self.macro_Precision = precision_score(self.label,pred, average='macro')
        self.micro_Precision = precision_score(self.label,pred, average='micro')
        self.weighted_Precision = precision_score(self.label,pred, average='weighted')
        print('macro_Precision={} micro_Precision={} weighted_Precision={}'\
              .format(self.macro_Precision, self.micro_Precision, self.weighted_Precision))
        self.f1 = f1_score(self.label, pred)
        
        if self.model_module=='bert':
            pre, rec, tre = precision_recall_curve(self.label, prob)
            self.average_precision = average_precision_score(self.label, prob)
        else:
            pre, rec, tre = precision_recall_curve(self.label.ravel(),prob[:,1])
            self.average_precision = average_precision_score(self.label, prob[:,1])
        #Plot Precision-Recall Curve
        plt.step(rec, pre, color='b', alpha=0.2, where='post')
        plt.fill_between(rec, pre, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0]);
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(self.average_precision))
        print('f1_score=%.3f auc=%.3f average_precision=%.3f' % (self.f1, roc_auc, self.average_precision))
        # plot no skill
        plt.plot([0, 1], [0.5, 0.5], linestyle='--')
        # plot the roc curve for the model
        plt.plot(rec, pre, marker='.')
        fig1 = plt.figure()
        self.plot_prec_recall_vs_tresh(pre, rec, tre)
        plt.show()
        
    def show_metrics(self):
        """pass"""
        y_pred, y_prob = self.pred_prob()
        print(classification_report(self.label,y_pred))
        self.accuracy =  accuracy_score(self.label, y_pred)
        print('accuracy', self.accuracy)
        self.show_confusion(y_pred)
        roc_auc = self.show_roc(y_prob) 
        self.show_prec_rec(roc_auc, prob=y_prob, pred=y_pred)       
    
    def cross_val_metrics(self,k=5):
        '''Accpets k as folds'''
        y_pred, y_prob, scores = self.pred_prob(method='cross_val',k=k)
        print('cross_validation_scores',scores)
        print('\n')
        print(classification_report(self.label,y_pred))
        self.accuracy_cv =  accuracy_score(self.label,y_pred)
        print('accuracy', self.accuracy_cv)
        self.show_confusion(y_pred)
        roc_auc = self.show_roc(y_prob) 
        self.show_prec_rec(roc_auc, prob=y_prob, pred=y_pred)


class ConsensusAggregationEngine(Meiyume):
    """pass"""
