
# coding: utf-8

# In[1]:


file_path = 'C:/Users/liuhongm/Desktop/download images/test.csv'
workspace_dir = 'C:/Users/liuhongm/Desktop/download images'


# In[24]:


import shutil
shutil.rmtree(os.path.join(workspace_dir, 'images'))


# In[10]:


from download_data import download_data
import pandas as pd
import os
from glob import glob

try:
    from tqdm import tqdm
    has_tqdm = True
except:
    has_tqdm = False

df = pd.read_csv(file_path, sep=',')
df.drop_duplicates('physical_id', keep='first', inplace=True)
print(df.head())

if not os.path.isdir(workspace_dir):
    os.makedirs(workspace_dir)
    
# download images

download_data(workspace_dir, df, url_column_name = 'physical_id')


# In[11]:


def rename_files(folder, input_file):
    files = glob(os.path.join(folder, '*.jpg'))
    pids = [s.split('\\')[-1][:-4] for s in files]
    df = pd.read_csv(file_path, sep=',')
    df.drop_duplicates('physical_id', keep='first', inplace=True)
    # compute dictionary mapping physicalid to asin
    asin_dict = df.set_index('physical_id').to_dict()['asin']
    for p, old_path in zip(pids, files):
        new_path = os.path.join(folder, str(asin_dict[p]) + '.PT12.jpg')
        os.rename(old_path, new_path)


# In[12]:


rename_files(os.path.join(workspace_dir, 'images'), file_path)

