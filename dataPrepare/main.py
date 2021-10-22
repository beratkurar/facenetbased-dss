import pandas as pd
import os
from PIL import Image

def matches( val ):
    def innerMatches( toMatch ):
        return toMatch.startswith(val)
    return innerMatches

#ok/105-Fg001-R-C01-R01-D03012012-T133309-LR924_012_F.jpg
def getManu( imageToManu, imagename ):
    name = imagename.replace("ok/", "")
    val = imageToManu[imageToManu["filename"].apply( matches( name[:11] ) )]
    if len(val):
        return val["manuscript"].to_list()[0]
    return None

def getLetterToManuDB(letterToImage, imageToManu):
    d = {}
    for image in letterToImage["imagename"].unique():
        #image =  "ok/P1109-Fg016-R-C01-R01-D01072013-T084539-LR924_012.jpg"
        manu = getManu(imageToManu, image)
        d[image] = manu

    letterToImage["manuscript"] = letterToImage["imagename"].map(d)
    return letterToImage




def getAllLetters(letter, fullDf):
    return fullDf[ fullDf ["ocr"] == letter ].copy()


def getAllHeh( imageNameshort, manuscript ):
    imageNameshort = imageNameshort.replace("ok/", "")
    boxes = imageNameshort.replace(".jpg", ".csv")

    imageName = os.path.join(r"/home/olya/Documents/fragmentsData/newImages", imageNameshort)


    boxes = os.path.join(r"/home/olya/Documents/fragmentsData/newImages", boxes)
    if not os.path.exists(imageName) or not os.path.exists(boxes):
        return

    pathToManu = r"/home/olya/Documents/fragmentsData/perManu/{}".format(manuscript)
    if not os.path.exists(pathToManu):
        os.mkdir(pathToManu)

    boxes = pd.read_csv(boxes)
    boxes = boxes[ boxes[ "cls"] == 5]
    boxes = boxes[boxes.columns[:4]]
    boxes.columns = ['x', 'y', 'w', 'h']
    boxes['w'] = boxes['x'] + boxes['w']
    boxes['h'] = boxes['y'] + boxes['h']
    boxes = boxes.values
    boxes = boxes.astype('float32')
    image = Image.open(imageName)
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        try:
            curr =  image.crop((box[0], box[1],box[2], box[3]))
            #curr.show()
            newName =  r"{}/{}".format(pathToManu, imageNameshort.replace( ".jpg", "{}.jpg".format(i) ))
            curr.save(newName)
        except:
            pass








def createAllLetters(df):
    for m in df.groupby("manuscript"):
        for im in m[1].groupby("imagename"):
            getAllHeh( im[0], m[0] )






def main():
    recreateLettersDF = False
    if recreateLettersDF:
        # 871 manuscripts, number of images per manuscript varies between 4 and 9302
        imageToManu = pd.read_csv(r"/home/olya/Documents/scrollPeriods/imageToScroll.csv")
        #1726 heh in 364 images
        dfLetters = pd.read_csv(r"/home/olya/Documents/scrollPeriods/0_line_table03.csv")
        hehDF = getAllLetters( u'\N{Hebrew Letter He}', dfLetters)
        hehDF = getLetterToManuDB( hehDF, imageToManu)
        hehDF.to_csv("hehDF.csv", index = False)
    else:
        hehDF = pd.read_csv("hehDF.csv")


    createAllLetters(hehDF)



if __name__ == "__main__":
    main()