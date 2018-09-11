from PIL import Image

basewidth = 300
im = Image.open('notwork.jpg')
img = im.convert('RGB')
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img.save('output.jpg') 
