import os
import pandas as pd
from dotenv import load_dotenv
from meiyume.utils import RedShiftReader, S3FileManager
import boto3
import io
from fastapi import FastAPI
import json


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
def get_data(s3_key):
    df = read_file_s3(filename=s3_key, file_type="feather")
    json_response = df.to_json(orient="records")
    json_response = json.loads(json_response)
    return json_response
