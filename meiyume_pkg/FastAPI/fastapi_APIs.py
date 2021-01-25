import os
import pandas as pd
from dotenv import load_dotenv
import boto3
import io
from fastapi import FastAPI
import json


class S3FileManager(object):
    def __init__(self, bucket: str = "meiyume-datawarehouse-prod"):
        self.bucket = bucket

    def get_matching_s3_objects(self, prefix: str = "", suffix: str = ""):
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")

        kwargs = {"Bucket": self.bucket}

        # We can pass the prefix directly to the S3 API.  If the user has passed
        # a tuple or list of prefixes, we go through them one by one.
        if isinstance(prefix, str):
            prefixes = (prefix,)
        else:
            prefixes = prefix

        for key_prefix in prefixes:
            kwargs["Prefix"] = key_prefix

            for page in paginator.paginate(**kwargs):
                try:
                    contents = page["Contents"]
                except KeyError:
                    break

                for obj in contents:
                    key = obj["Key"]
                    if key.endswith(suffix):
                        yield obj

    def get_matching_s3_keys(self, prefix: str = "", suffix: str = ""):
        for obj in self.get_matching_s3_objects(prefix, suffix):
            yield obj  # obj["Key"]


file_manager = S3FileManager()
s3_keys = file_manager.get_matching_s3_keys(
    prefix="Feeds/BeautyTrendEngine/WebAppData/"
)
s3_keys = list(s3_keys)
s3_keys_name_list = [x["Key"].split("/")[-1] for x in s3_keys]

load_dotenv()
S3_BUCKET = "meiyume-datawarehouse-prod"
S3_PREFIX = "Feeds/BeautyTrendEngine"
S3_REGION = "ap-southeast-1"
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")


def get_s3_client(region: str, access_key_id: str, secret_access_key: str):
    if region == "":
        print("*ERROR: S3 client information not set*")
        return sys.exit(1)
    else:
        try:
            client = boto3.client(
                "s3",
                region,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
            )
        except Exception as ex:
            client = boto3.client("s3")
    return client


def read_file_s3(
    filename: str,
    prefix: str = f"{S3_PREFIX}/WebAppData",
    bucket: str = S3_BUCKET,
    file_type: str = "feather",
) -> pd.DataFrame:
    key = prefix + "/" + filename
    s3 = get_s3_client(S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    obj = s3.get_object(Bucket=bucket, Key=key)
    if file_type == "feather":
        df = pd.read_feather(io.BytesIO(obj["Body"].read()))
    elif file_type == "pickle":
        df = pd.read_pickle(io.BytesIO(obj["Body"].read()))
    return df


app = FastAPI()


@app.get("/data/{s3_key}")
def get_data_by_s3_bucket_key(s3_key):
    df = read_file_s3(filename=s3_key, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/category_page_distinct_brands_products")
def get_data_by_s3_key_category_page_distinct_brands_products():
    df = read_file_s3(
        filename=category_page_distinct_brands_products, file_type="feather"
    )
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/category_page_item_package_oz")
def get_data_by_s3_key_category_page_item_package_oz():
    df = read_file_s3(filename=category_page_item_package_oz, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/category_page_item_variations_price")
def get_data_by_s3_key_category_page_item_variations_price():
    df = read_file_s3(filename=category_page_item_variations_price, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/category_page_new_ingredients")
def get_data_by_s3_key_category_page_new_ingredients():
    df = read_file_s3(filename=category_page_new_ingredients, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/category_page_new_products_count")
def get_data_by_s3_key_category_page_new_products_count():
    df = read_file_s3(filename=category_page_new_products_count, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/category_page_new_products_details")
def get_data_by_s3_key_category_page_new_products_details():
    df = read_file_s3(filename=category_page_new_products_details, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/category_page_pricing_data")
def get_data_by_s3_key_category_page_pricing_data():
    df = read_file_s3(filename=category_page_pricing_data, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/category_page_reviews_by_user_attributes")
def get_data_by_s3_key_category_page_reviews_by_user_attributes():
    df = read_file_s3(
        filename=category_page_reviews_by_user_attributes, file_type="feather"
    )
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/category_page_top_products")
def get_data_by_s3_key_category_page_top_products():
    df = read_file_s3(filename=category_page_top_products, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/ing_page_ing_data")
def get_data_by_s3_key_ing_page_ing_data():
    df = read_file_s3(filename=ing_page_ing_data, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/landing_page_data")
def get_data_by_s3_key_landing_page_data():
    df = read_file_s3(filename=landing_page_data, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/meta_product_launch_intensity_category_month")
def get_data_by_s3_key_meta_product_launch_intensity_category_month():
    df = read_file_s3(
        filename=meta_product_launch_intensity_category_month, file_type="feather"
    )
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/meta_product_launch_trend_category_month")
def get_data_by_s3_key_meta_product_launch_trend_category_month():
    df = read_file_s3(
        filename=meta_product_launch_trend_category_month, file_type="feather"
    )
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/meta_product_launch_trend_product_type_month")
def get_data_by_s3_key_meta_product_launch_trend_product_type_month():
    df = read_file_s3(
        filename=meta_product_launch_trend_product_type_month, file_type="feather"
    )
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/new_ingredient_trend_category_month")
def get_data_by_s3_key_new_ingredient_trend_category_month():
    df = read_file_s3(filename=new_ingredient_trend_category_month, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/new_ingredient_trend_product_type_month")
def get_data_by_s3_key_new_ingredient_trend_product_type_month():
    df = read_file_s3(
        filename=new_ingredient_trend_product_type_month, file_type="feather"
    )
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/prod_page_ing_data")
def get_data_by_s3_key_prod_page_ing_data():
    df = read_file_s3(filename=prod_page_ing_data, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/prod_page_item_data")
def get_data_by_s3_key_prod_page_item_data():
    df = read_file_s3(filename=prod_page_item_data, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/prod_page_product_review_summary")
def get_data_by_s3_key_prod_page_product_review_summary():
    df = read_file_s3(filename=prod_page_product_review_summary, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/prod_page_review_sentiment_influence")
def get_data_by_s3_key_prod_page_review_sentiment_influence():
    df = read_file_s3(
        filename=prod_page_review_sentiment_influence, file_type="feather"
    )
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/prod_page_review_talking_points")
def get_data_by_s3_key_prod_page_review_talking_points():
    df = read_file_s3(filename=prod_page_review_talking_points, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/prod_page_reviews_attribute")
def get_data_by_s3_key_prod_page_reviews_attribute():
    df = read_file_s3(filename=prod_page_reviews_attribute, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/product_page_metadetail_data")
def get_data_by_s3_key_product_page_metadetail_data():
    df = read_file_s3(filename=product_page_metadetail_data, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/review_trend_by_marketing_category_month")
def get_data_by_s3_key_review_trend_by_marketing_category_month():
    df = read_file_s3(
        filename=review_trend_by_marketing_category_month, file_type="feather"
    )
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/review_trend_by_marketing_product_type_month")
def get_data_by_s3_key_review_trend_by_marketing_product_type_month():
    df = read_file_s3(
        filename=review_trend_by_marketing_product_type_month, file_type="feather"
    )
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/review_trend_category_month")
def get_data_by_s3_key_review_trend_category_month():
    df = read_file_s3(filename=review_trend_category_month, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/review_trend_product_type_month")
def get_data_by_s3_key_review_trend_product_type_month():
    df = read_file_s3(filename=review_trend_product_type_month, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response
