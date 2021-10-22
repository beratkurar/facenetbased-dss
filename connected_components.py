import cv2
from PIL import Image
import os
import pandas as pd

def binariseImage(imPath):
    img = cv2.imread(imPath, 0)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 10)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im = Image.fromarray(th3)
    return im


def runConnected(imPath, showComps = False):
    import numpy as np

    # image = cv2.imread(imPath)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 0, 255,
    #                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]




    img = cv2.imread(imPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5), 10)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)



    temp = np.copy(thresh)
    temp[thresh > 127] = 255
    temp[thresh <= 127] = 0
    thresh = temp


    output = cv2.connectedComponentsWithStats(
        thresh, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    validCmp = 1
    comps = None
    componentMask = None
    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(
                i + 1, numLabels)
        # otherwise, we are examining an actual connected component
        else:
            text = "examining component {}/{}".format(i + 1, numLabels)
        # print a status message update for the current connected
        # component
        #print("[INFO] {}".format(text))
        # extract the connected component statistics and centroid for
        # the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        if area > 1000 and area <10000  :
            if showComps:
                labels[labels == i] = validCmp
                validCmp+=1
            if comps is not None:
                df2 = pd.DataFrame([[x, y, w, h]], columns=['x', 'y', 'w', 'h'])
                pd.concat([comps, df2])
            else:
                comps = pd.DataFrame([[x, y, w, h]], columns=['x', 'y', 'w', 'h'])
        elif showComps:
             labels[labels==i]=0

        #cv2.imshow("Connected Component", componentMask)
       # cv2.waitKey(0)

    if showComps:

        copyLabels = np.copy(labels)

        labelsVals = range(1, np.max(labels))

        firstLabels = range(1, np.max(labels)//2)
        secLabels = range(np.max(labels) // 2, np.max(labels))
        toSwitch = {}
        currFist =0
        currSec =0
        i=1
        while i <  np.max(labels) :
            if currFist < len(firstLabels):
                toSwitch[i] = firstLabels[currFist]
                currFist+=1
                i+=1
            if currSec < len(secLabels):
                toSwitch[i] = secLabels[currSec]
                currSec +=1
                i+=1

        for origLabel, newLabel in toSwitch.items():
            copyLabels[ labels == origLabel ] = newLabel

        labels = copyLabels
        # Map component labels to hue val, 0-179 is the hue range in OpenCV
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        # Converting cvt to BGR
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        # set bg label to black
        labeled_img[label_hue == 0] = 0
        i = Image.fromarray(labeled_img)
        i.save(imPath[:-4]+"comps.jpg")

    if comps is not None:
        comps.to_csv(imPath[:-3]+"csv")




def main():
    #imPath = r'/home/olya/Documents/fragmentsData/DSS_Joins_bw/2Q13/P741-Fg004-R-C01-R01-D02062013-T145645-LR924_012.jpg'
    #runConnected(imPath, showComps=True)
    import os
    for root, dirs, files in os.walk("/home/olya/Documents/fragmentsData/DSS_Joins_bw/", topdown=False):
        if root.endswith('bin') or root.endswith('bin1') or root.endswith('bin2') or root.endswith('bin2'):
            continue
        for name in files:
            curr = os.path.join(root, name)
            if curr.endswith(".jpg"):
                print(curr)
                runConnected(curr, showComps = True)

if __name__ == "__main__":
    main()