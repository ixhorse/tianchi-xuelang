import pandas as pd
from pandas import DataFrame
import os

csv = 'output/results.csv'
image_path = '/home/mcc/data/xuelang/test'

test_images = os.listdir(image_path)

df = pd.read_csv(csv, sep=',')
print(type(df['filename']))
for name in df['filename']:
    if not os.path.exists(os.path.join(image_path, name)):
        print(name)

for name in test_images:
    if not name in list(df['filename']):
        print(name, 1)