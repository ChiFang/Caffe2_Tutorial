# load up the caffe2 workspace
from caffe2.python import workspace
# choose your model here (use the downloader first)
from caffe2.python.models import squeezenet as mynet
# helper image processing functions
import caffe2.python.tutorials.helpers as helpers

import os
import time

# load the pre-trained model
print "load the pre-trained model....."
init_net = mynet.init_net
predict_net = mynet.predict_net

# you must name it something
predict_net.name = "squeezenet_predict"
workspace.RunNetOnce(init_net)
workspace.CreateNet(predict_net)
p = workspace.Predictor(init_net.SerializeToString(), predict_net.SerializeToString())

# use whatever image you want (local files or urls)
img = "https://cdn.pixabay.com/photo/2015/02/10/21/28/flower-631765_1280.jpg"
# average mean to subtract from the image
mean = 128
# the size of images that the model was trained with
input_size = 227

# use the image helper to load the image and convert it to NCHW
img = helpers.loadToNCHW(img, mean, input_size)


os.system('clear')
# submit the image to net and get a tensor of results
print "submit the image to net and get a tensor of results....."

tStart = time.time()
results = p.run([img])
tEnd = time.time()

response = helpers.parseResults(results)
# and lookup our result from the list
print "\n---------------------"
print "It cost %f sec to predict\n" % (tEnd - tStart)
print response
print "---------------------\n"

print "save URL Image..."
import urllib2
mp3file = urllib2.urlopen("https://cdn.pixabay.com/photo/2015/02/10/21/28/flower-631765_1280.jpg")
with open('test.jpg','wb') as output:
  output.write(mp3file.read())
  
print "Log result..."
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
img = Image.open("test.jpg")
draw = ImageDraw.Draw(img)
# font = ImageFont.truetype(<font-file>, <font-size>)
#font = ImageFont.truetype("sans-serif.ttf", 16)

font_path=os.environ.get("FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
ttFont = ImageFont.truetype(font_path, 20)

# draw.text((x, y),"Sample Text",(r,g,b))
# draw.text((0, 0),"Sample Text",(255,255,255),font=font)
draw.text((0, 0),response,(0,0,0),font=ttFont)
img.save('test-out.jpg')


