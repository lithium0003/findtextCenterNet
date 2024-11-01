import boto3

client = boto3.client('s3', endpoint_url='https://s3.isk01.sakurastorage.jp', region_name='jp-north-1')
Bucket = '20240830'

def download():
    log = client.list_objects_v2(Bucket=Bucket)
    if 'Contents' in log:
        Contents = log['Contents']
        keys = [c['Key'] for c in Contents]
    else:
        return

    directory = sorted(set([k.split('/')[0] for k in keys]))
    if len(directory) > 0:
        last_dir = directory[-1]
        print(last_dir)

        modelpt = sorted([k for k in keys if k.startswith('%s/epoch'%last_dir)])
        print('\n'.join(modelpt))
        if len(modelpt) > 0:
            last_cp = modelpt[-1]
            print('output',last_cp)
            response = client.get_object(Bucket=Bucket, Key=last_cp)
            body = response['Body'].read()
            with open('result1/model.pt','wb') as wf:
                wf.write(body)
