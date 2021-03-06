# Superresolution 
A super-resolution mobile app built with Flutter. The superresolution algorithm can be trained using [fastai's notebook](https://github.com/fastai/fastai2/blob/master/nbs/course/lesson7-superres.ipynb), and is to my knowledge one of the best available today. 

## Description

* A web API code that can be deployed on any cloud, which performs the superresolution. 
* A mobile app built with [Flutter](https://github.com/flutter/flutter) that enables you to take a picture (from the library or with the camera), enhance it by calling the model above, and then compare it with the initial one.

If you want the app without compiling it or use the API running the algorithm, please contact me.

## To deploy locally
```
docker build -t superres .
docker run -p 80:80 superres
```
You can then test the API using the notebook and by setting http://localhost/ as the app URL.   

## Known issue

Unfortunately, the web API doesn't support heavy images, which unfortunately is what most smartphones' camera provides. For this reason, it was particularly hard for me to use the algorithm directly on images taken with my smartphone. If someone has any idea on how to fix this, please let me know!

## Examples

![img_input](https://github.com/sebderhy/superres/blob/master/images/superres-2b-results.gif "GIF Results") 

![img_input](https://github.com/sebderhy/superres/blob/master/images/superres-2b-results-2.gif "GIF Results 2") 

## App screenshots

<img src="https://github.com/sebderhy/superres/blob/master/images/flutter_screenshot_1.jpg" width="256" /> <img src="https://github.com/sebderhy/superres/blob/master/images/flutter_screenshot_2.jpg" width="256" />
<img src="https://github.com/sebderhy/superres/blob/master/images/flutter_screenshot_3.jpg" width="512" />
<img src="https://github.com/sebderhy/superres/blob/master/images/flutter_screenshot_4.jpg" width="512" />

