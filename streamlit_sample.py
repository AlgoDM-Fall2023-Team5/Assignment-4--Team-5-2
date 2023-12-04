import streamlit as st
import s3fs
import boto3 

# Set AWS credentials as environment variables
# import os
# os.environ['AWS_ACCESS_KEY_ID'] = 'AKIASGVOLPG2BPBLER6T'
# os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
# os.environ['AWS_DEFAULT_REGION'] = 'us-east-2'

s3_client = boto3.client(service_name = 's3',
                aws_access_key_id='AKIASGVOLPG2BPBLER6T',
                aws_secret_access_key='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',
                region_name = 'us-east-2')

img = s3_client.get_object(Bucket='assignment4admt5', Key='dataset/id_00000001_02_1_front.jpg')


image_bytes = img['Body'].read()

# Display the image in Streamlit
st.image(image_bytes)