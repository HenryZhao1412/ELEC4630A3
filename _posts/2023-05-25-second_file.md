# My learning of the 00-is-it-a-bird-creating-a-model-from-your-own-data.ipynb model

```python
import socket,warnings
try:
    socket.setdefaulttimeout(1)
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('1.1.1.1', 53))
except socket.error as ex: raise Exception("STOP: No internet. Click '>|' in top right and set 'Internet' switch to on")
```
This part of would check whether the Internet is connected. If this can not recive a response in 1 second, it would warn you "*STOP: No internet. Click '>|' in top right and set 'Internet' switch to on*"



```python
import os
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

if iskaggle:
    !pip install -Uqq fastai
```

This part of code is to check whether it is in the kaggle environment, if it is, then the syetem would execute "*!pip install -Uqq fastai*", it would update the fastai to the latest version.

```python
!pip install -Uqq duckduckgo_search
```
This is used to install the duckduckgo.

```python
from duckduckgo_search import ddg_images
from fastcore.all import *

def search_images(term, max_images=200): 
    return L(ddg_images(term, max_results=max_images)).itemgot('image')
```

import two libraries and give us images' URL, the default number of images it would search is 200, if there are now special sepcification, it would search 200 images.

```python
urls = search_images('bird photos', max_images=1)
urls[0]
```

Search one image of bird, and show off the URL of the searched image.


```python
from fastdownload import download_url
dest = 'bird.jpg'
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(256,256)
```
Download an image and store it in "*dest*", and it would not show the progress of the downloading.<br>

Then show downloaded image would be show off, the image would be resized to 256x256.

```python
download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256,256)
```
Search one image of forest and store it in the "*forest.jpg*" (That is download an image and name it 'forest.jpg'). After the download, the image would be resized to 256x256.

```python 
searches = 'forest','bird'
path = Path('bird_or_not')
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} sun photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} shade photo'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)
```

This piece of code is to download images of forests and birds, it each group would download 600 images, we take forest as example, it would download 200 images of forest, 200 images of forest sun and 200 images of forest shade. The downloaded images of bird are the same.

```python
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)
```
Tell us how many images can not be used for the training. And it can delete those useless images.

```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)

dls.show_batch(max_n=6)
```
This part of code is similar to the DataLoader code I written in my first blog.
















