from logging import basicConfig
from PIL import Image
import cv2
import numpy as np
import time
import math

def overlay_transparent(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    Y = y
    X = x

    if y < 0:
        Y = 0
        overlay = overlay[-1*y:,:]
    
    if x < 0:
        X = 0
        overlay = overlay[:,-1*x:]
     
    overlay_image = overlay[..., :4]
    mask = overlay[..., 3:] / 255.0
    background[Y:y+h, X:x+w] = (1.0 - mask) * background[Y:y+h, X:x+w] + mask * overlay_image

    return background

    

def plantTreeCV(imageNpArray,positionList,LandscapeNp,angle):
    """ plant the tree at the position which in the position list
        need Tree image array that conclude with the three different color
        it will put the tree shadow too

        Args:
            imageNpArray : Tree Image array (cv2 numpy image array)
            positionList : list of the x,y point lists that want to plant the tree(mid point) 
            LandscapeNp : Back ground image (cv2 numpy image)
            angle : the degree angle of the tree shadow. 3'o clock is 0 degree

        Return:
            numpy image (cv2)
    """
    i=0
    w,h,c = imageNpArray[0].shape
    positionList = np.array(positionList)
    positionList -= [w//2,h//2]

    angle = math.radians(angle)*-1
    size = w//8+h//8
    for pos in positionList:
        shadowDest = (int(pos[0]+size*math.cos(angle)),int(pos[1]+size*math.sin(angle)))
        # LandscapeNp = pasteImage(LandscapeNp,makeTreeShadow(imageNpArray[i%3]),shadowDest)
        # LandscapeNp = pasteImage(LandscapeNp,imageNpArray[i%3],pos)
        LandscapeNp = overlay_transparent(LandscapeNp,makeTreeShadow(imageNpArray[i%3]),shadowDest[0],shadowDest[1])
        LandscapeNp = overlay_transparent(LandscapeNp,imageNpArray[i%3],pos[0],pos[1])
        i+=1
    return LandscapeNp

def makeThree(imageNp,weight):
    """ make the tree more various color(more greener only)
        use weight for the greener amount

        Args:
            imageNp : tree image (cv2 numpy image)
            weight : weight of the greener 

        Return
            cv2 numpy image list
    """
    change = np.array([0,weight,0,0],dtype="uint8")
    ImageArray = []
    for w in range(3):
        ImageArray.append(imageNp+change*w)
    return ImageArray

def makeTreeShadow(tree):
    b,g,r,a = cv2.split(tree)
    maskA = a != 0
    b = np.where(maskA,110,b)
    g = np.where(maskA,70,g)
    r = np.where(maskA,50,r)
    a = np.where(maskA,125,a)
    shadow = cv2.merge((b,g,r,a))
    return shadow
    

if __name__ =="__main__":
    posList = [[0,0],
                [130,130],
                [130,100],
                [100,700],
                [100,900],
                [300,900],
                [500,900],
                [100,1000]]


    Tnp = cv2.imread('TreePlantS.png',-1)
    Bnp = cv2.imread('0.png',-1)

    ImArray = makeThree(Tnp,15)
    new2 = plantTreeCV(ImArray,posList,Bnp,45)
    cv2.imwrite("TreePlantTest.png",new2)



