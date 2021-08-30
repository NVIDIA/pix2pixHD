from PIL import Image,ImageDraw
import numpy as np
import math

ImageSize = (1024,1024)
#nputImage = Image.open('C:\\Users\\aswww\\Desktop\\planning\\LandscapeMaker\\colorOutput\\10.jpg')
#terRate = 50

def shadow4Building(inputImage,angle,size):
    tempImage = Image.new("RGBA",inputImage.size,(255,255,255,0))
    draw = ImageDraw.Draw(tempImage)
    angle = math.radians(angle)*-1
    for j in range(inputImage.size[1]):
        for i in range(inputImage.size[0]):
            if inputImage.getpixel((i,j)) == 18:
                dest = (i+size*math.cos(angle),j+size*math.sin(angle))
                # if tempImage.getpixel(dest)[3] != 0:
                #     break
                draw.line([(i,j),dest],fill=(50,70,110,125),width=1)
    for j in range(inputImage.size[1]):
        for i in range(inputImage.size[0]):
            if inputImage.getpixel((i,j)) == 18:
                tempImage.putpixel((i,j),(0,0,0,0))
    return tempImage

def shadow4Tree(inputImage,angle,size):
    tempImage = Image.new("RGBA",ImageSize,(255,255,255,0))
    angle = math.radians(angle)*-1
    for j in range(ImageSize[1]):
        for i in range(ImageSize[0]):
            if inputImage.getpixel((i,j))[0] < filterRate and 255-filterRate < inputImage.getpixel((i,j))[1]and 255-filterRate <inputImage.getpixel((i,j))[2]:
                dest = (int(i+size*math.cos(angle)),int(j+size*math.sin(angle)))
                # if inputImage.getpixel(dest)[0] < filterRate and 255-filterRate < inputImage.getpixel(dest)[1]and 255-filterRate <inputImage.getpixel(dest)[2]:
                #     break
                tempImage.putpixel(dest,(50,70,110,125))
    for j in range(ImageSize[1]):
        for i in range(ImageSize[0]):
            if inputImage.getpixel((i,j))[0] < filterRate and 255-filterRate < inputImage.getpixel((i,j))[1]and 255-filterRate <inputImage.getpixel((i,j))[2]:
                tempImage.putpixel((i,j),(0,0,0,0))
    return tempImage

if __name__ == "__main__":
    tempImage = shadow4Building(inputImage,120,120)
    tempImage2 = shadow4Tree(inputImage,120,15)    
    newImage = Image.new("RGBA",ImageSize,(255,255,255,0))
    newImage.paste(inputImage,(0,0))
    # newImage.paste(tempImage,(0,0),tempImage)
    newImage.paste(tempImage2,(0,0),tempImage2)
    newImage.save("sample.png")
     
