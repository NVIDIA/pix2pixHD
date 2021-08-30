from PIL import Image,ImageDraw
import numpy as np
import math


def drawShadow(imageNp,colorCode,angle,size):
    # imageNp = np.array(image)
    mask = imageNp == colorCode
    mask = Image.fromarray(mask)
    doIndex = np.where(imageNp == colorCode)
    Shadow = Image.new("RGBA",mask.size)
    draw = ImageDraw.Draw(Shadow)
    angle = math.radians(angle)*-1

    for i,v in enumerate(doIndex[0]):
        dest = (doIndex[1][i]+size*math.cos(angle),doIndex[0][i]+size*math.sin(angle))
        draw.line([(doIndex[1][i],doIndex[0][i]),dest],fill=(50,70,110,125),width=1)
    
    # for i,v in enumerate(doIndex[0]):
    #     Shadow.putpixel((doIndex[1][i],doIndex[0][i]),(0,0,0,0))

    newImage = Image.new("RGBA",image.size,(255,255,255,0))
    newImage.paste(image,(0,0),Shadow)
    im = Image.composite(newImage,Shadow,mask).convert("RGBA")
    return im

if __name__ == "__main__":
    import time
    start = time.time()
    image = Image.open("1-1.png")
    drawShadow(image,18,45,120)
    print("time :", time.time() - start)
    