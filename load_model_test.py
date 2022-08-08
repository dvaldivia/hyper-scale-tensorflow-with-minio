import tensorflow as  tf
import tensorflow_io
import os

datasets_bucket = "datasets"
# How to access MinIO
minio_address = "localhost:9000"
minio_access_key = "minioadmin"
minio_secret_key = "minioadmin"

os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
os.environ["AWS_REGION"] = "us-east-1"
os.environ["S3_ENDPOINT"] = minio_address
os.environ["S3_USE_HTTPS"] = "0"
os.environ["S3_VERIFY_SSL"] = "0"

print(os.getenv("S3_ENDPOINT"))
embed = tf.saved_model.load(f"s3://datasets/universal-sentence-encoder/4")