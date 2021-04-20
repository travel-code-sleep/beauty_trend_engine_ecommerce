import io
import json
import os

import boto3
import pandas as pd
from fastapi import FastAPI
from typing import Optional

S3_BUCKET = "meiyume-datawarehouse-prod"
S3_PREFIX = "Feeds/BeautyTrendEngine"
S3_REGION = "ap-southeast-1"


def get_s3_client(region: str):
    if region == "":  # or access_key_id == '' or secret_access_key == '':
        print("*ERROR: S3 client information not set*")
        return sys.exit(1)
    else:
        client = boto3.client("s3")
    return client


def read_file_s3(
    filename: str,
    prefix: str = f"{S3_PREFIX}/WebAppData",
    bucket: str = S3_BUCKET,
    file_type: str = "feather",
    number_of_records: str = "all",
) -> pd.DataFrame:
    print("number_of_records", number_of_records)
    key = prefix + "/" + filename
    s3 = get_s3_client(S3_REGION)
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_feather(io.BytesIO(obj["Body"].read()))
    if number_of_records != "all" and number_of_records.isdigit():
        json_response = df[: int(number_of_records)].to_json(
            orient="records", date_format="iso"
        )
    else:
        json_response = df.to_json(orient="records", date_format="iso")
    json_response = json.loads(json_response)
    return json_response


app = FastAPI()


# @app.get("/data/{s3_key}/{number_of_records}")
# def get_data_by_s3_bucket_key(s3_key, number_of_records):
#     json_response = read_file_s3(
#         filename=s3_key, file_type="feather", number_of_records=number_of_records
#     )
#     return json_response


@app.get("/data/category_page_distinct_brands_products/{number_of_records}")
def get_data_by_s3_key_category_page_distinct_brands_products(number_of_records):
    json_response = read_file_s3(
        filename="category_page_distinct_brands_products",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/category_page_item_package_oz/{number_of_records}")
def get_data_by_s3_key_category_page_item_package_oz(number_of_records):
    json_response = read_file_s3(
        filename="category_page_item_package_oz",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/category_page_item_variations_price/{number_of_records}")
def get_data_by_s3_key_category_page_item_variations_price(number_of_records):
    json_response = read_file_s3(
        filename="category_page_item_variations_price",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/category_page_new_ingredients/{number_of_records}")
def get_data_by_s3_key_category_page_new_ingredients(number_of_records):
    json_response = read_file_s3(
        filename="category_page_new_ingredients",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/category_page_new_products_count/{number_of_records}")
def get_data_by_s3_key_category_page_new_products_count(number_of_records):
    json_response = read_file_s3(
        filename="category_page_new_products_count",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/category_page_new_products_details/{number_of_records}")
def get_data_by_s3_key_category_page_new_products_details(number_of_records):
    json_response = read_file_s3(
        filename="category_page_new_products_details",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/category_page_pricing_data/{number_of_records}")
def get_data_by_s3_key_category_page_pricing_data(number_of_records):
    json_response = read_file_s3(
        filename="category_page_pricing_data",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/category_page_reviews_by_user_attributes/{number_of_records}")
def get_data_by_s3_key_category_page_reviews_by_user_attributes(number_of_records):
    json_response = read_file_s3(
        filename="category_page_reviews_by_user_attributes",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/category_page_top_products/{number_of_records}")
def get_data_by_s3_key_category_page_top_products(number_of_records):
    json_response = read_file_s3(
        filename="category_page_top_products",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/ing_page_ing_data/{number_of_records}")
def get_data_by_s3_key_ing_page_ing_data(number_of_records):
    json_response = read_file_s3(
        filename="ing_page_ing_data",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/landing_page_data/{number_of_records}")
def get_data_by_s3_key_landing_page_data(number_of_records):
    json_response = read_file_s3(
        filename="landing_page_data",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


# http://localhost:8000/data/meta_product_launch_intensity_category_month?source=us&start-date=2020-01-01&end-date=2020-12-31&category=bath-body,hair-products&subcategory=body
@app.get("/data/meta_product_launch_intensity_category_month/{number_of_records}")
def get_data_by_s3_key_meta_product_launch_intensity_category_month(
    number_of_records,
    source: Optional[str] = None,
    startdate: Optional[str] = None,
    enddate: Optional[str] = None,
    category: Optional[str] = None,
):
    filename = "meta_product_launch_intensity_category_month"
    file_type = "feature"
    prefix = f"{S3_PREFIX}/WebAppData"
    bucket = S3_BUCKET
    key = prefix + "/" + filename
    number_of_records = number_of_records if number_of_records else "all"
    s3 = get_s3_client(S3_REGION)
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_feather(io.BytesIO(obj["Body"].read()))
    if startdate and enddate:
        df = df[(df["meta_date"] > startdate) & (df["meta_date"] < enddate)]
    categories = category.split(",")
    sources = source.split(",")
    if len(categories) > 0:
        df = df[df["category"].isin(categories)]
    if len(sources) > 0:
        df = df[df["source"].isin(sources)]
    # print(df)
    if number_of_records != "all" and number_of_records.isdigit():
        print("here")
        json_response = df[: int(number_of_records)].to_json(
            orient="records", date_format="iso"
        )
    else:
        json_response = df.to_json(orient="records", date_format="iso")
    json_response = json.loads(json_response)
    return json_response


@app.get("/data/meta_product_launch_trend_category_month/{number_of_records}")
def get_data_by_s3_key_meta_product_launch_trend_category_month(number_of_records):
    json_response = read_file_s3(
        filename="meta_product_launch_trend_category_month",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/meta_product_launch_trend_product_type_month/{number_of_records}")
def get_data_by_s3_key_meta_product_launch_trend_product_type_month(number_of_records):
    json_response = read_file_s3(
        filename="meta_product_launch_trend_product_type_month",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/new_ingredient_trend_category_month/{number_of_records}")
def get_data_by_s3_key_new_ingredient_trend_category_month(number_of_records):
    json_response = read_file_s3(
        filename="new_ingredient_trend_category_month",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/new_ingredient_trend_product_type_month/{number_of_records}")
def get_data_by_s3_key_new_ingredient_trend_product_type_month(number_of_records):
    json_response = read_file_s3(
        filename="new_ingredient_trend_product_type_month",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/prod_page_ing_data/{number_of_records}")
def get_data_by_s3_key_prod_page_ing_data(number_of_records):
    json_response = read_file_s3(
        filename="prod_page_ing_data",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/prod_page_item_data/{number_of_records}")
def get_data_by_s3_key_prod_page_item_data(number_of_records):
    json_response = read_file_s3(
        filename="prod_page_item_data",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/prod_page_product_review_summary/{number_of_records}")
def get_data_by_s3_key_prod_page_product_review_summary(number_of_records):
    json_response = read_file_s3(
        filename="prod_page_product_review_summary",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/prod_page_review_sentiment_influence/{number_of_records}")
def get_data_by_s3_key_prod_page_review_sentiment_influence(number_of_records):
    json_response = read_file_s3(
        filename="prod_page_review_sentiment_influence",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/prod_page_review_talking_points/{number_of_records}")
def get_data_by_s3_key_prod_page_review_talking_points(number_of_records):
    json_response = read_file_s3(
        filename="prod_page_review_talking_points",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/prod_page_reviews_attribute/{number_of_records}")
def get_data_by_s3_key_prod_page_reviews_attribute(number_of_records):
    json_response = read_file_s3(
        filename="prod_page_reviews_attribute",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/product_page_metadetail_data/{number_of_records}")
def get_data_by_s3_key_product_page_metadetail_data(number_of_records):
    json_response = read_file_s3(
        filename="product_page_metadetail_data",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/review_trend_by_marketing_category_month/{number_of_records}")
def get_data_by_s3_key_review_trend_by_marketing_category_month(number_of_records):
    json_response = read_file_s3(
        filename="review_trend_by_marketing_category_month",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/review_trend_by_marketing_product_type_month/{number_of_records}")
def get_data_by_s3_key_review_trend_by_marketing_product_type_month(number_of_records):
    json_response = read_file_s3(
        filename="review_trend_by_marketing_product_type_month",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/review_trend_category_month/{number_of_records}")
def get_data_by_s3_key_review_trend_category_month(number_of_records):
    json_response = read_file_s3(
        filename="review_trend_category_month",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response


@app.get("/data/review_trend_product_type_month/{number_of_records}")
def get_data_by_s3_key_review_trend_product_type_month(number_of_records):
    json_response = read_file_s3(
        filename="review_trend_product_type_month",
        file_type="feature",
        number_of_records=number_of_records,
    )
    return json_response
