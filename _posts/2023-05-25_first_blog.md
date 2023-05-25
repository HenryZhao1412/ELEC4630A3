
# My learning of Fastai

## The basis of fastai:
Every fastai's application all use the same basic steps and code：<br>
They are:  
1. Create DataLoaders.<br>
2. Create a Learner.<br>
3. Call a *fit* method.<br>
4. Make predictions.<br>

We can take the example shown on the source page as an example:<br>

Firstly, we need to import the needed libraries:<br>

```python
from fastai.vision.all import *
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *
```
The next line of code is:

```python
path = untar_data(URLs.PETS)/'images'
```
This line of code uses *'untar_data'* function to download dataset from URLs.PETS, and it told us the path is the *'path'*
variable. The following code " */'image'* " means that under this path, the images would be saved in a folder called *'images'*

```python
def is_cat(x):
    return x[0].isupper()
```

This line means that we defined a function by ourselves. It would take string as an input, I think the string would be the name of the files. And this function would check whether the animal in this fucntion is a cat, the method is to check wether the first character of the image's name is upper format, if that is upper, it would return the bool value *True*, otherwise it would return *False*.


```python
dls = ImageDataLoaders.from_name_func(
    path,
    get_image_files(path),
    valid_pct=0.2,
    seed=42,
    label_func=is_cat,
    item_tfms=Resize(224))
```

This part of code created a DataLoader <br>
|Parameters|Explanation|
|-|-|
|*path*| This is the path for the dataset.|
|*get_image_files(path)*|This is to get all the images from the given path|
|*valid_pct=0.2*|About 20% of data would be used to confirm.|
|*seed=42*|Set a random to ensure it can be repeated.|
|*label_function=is_cat*|Tell us the function to be used is *is_cat*.|
|*item_tfms=Resize(224)*|Do a transform to every images, resize these images to 224x224 pixels|

```python
learn = vision.learner(dls,resnet34,metrics=error_rate)
```

|Parameters|Explantion|
|-|-|
|*dls*|The DataLoader|
|*resnet34*|The architecture of the pretrained model used, here we use Resnet34|
|*metrics=error_rate*|The metrics we used to evalutate the model, here we use the rate of error.|

```python
learn.fine_tune(1)
```
This line of code tell us this model would be trained for one time.<br>
<br>


















source: https://docs.fast.ai/quick_start.html


