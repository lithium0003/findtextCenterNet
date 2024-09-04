import boto3
import datetime

dt_now = datetime.datetime.now()

client = boto3.client('s3', endpoint_url='https://s3.isk01.sakurastorage.jp')
Bucket = '20240830'
subkey = dt_now.strftime('%Y%m%d-%H%M%S')

def upload(filename, key):
    client.upload_file(filename, Bucket, '%s/%s'%(subkey, key))
