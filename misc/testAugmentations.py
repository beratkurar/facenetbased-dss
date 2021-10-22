
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image


def main():
    i = Image.open(r"/home/olya/Documents/fragmentsData/DSS_Joins/4Q27/P1080-Fg006-R-C01-R01-D04082013-T104759-LR445_ColorCalData_IAA_Both_CC110304_110702.png")
    h,w = i.size
    i  = i.resize((h//6, w//6))
    i.show()
    arrI = np.array(i)

    seq = iaa.Sequential([
        #iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
        #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        iaa.Multiply((0.5, 1.5), per_channel=0.2),
        #should not succeed on blured images
        #iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
        #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),



        #Try only grayscale
        #iaa.Grayscale(alpha=(0.0, 1.0)),

    ])



    arrAug = seq( images= [arrI, arrI,arrI,arrI,arrI])

    augmented = [ Image.fromarray(im) for im in arrAug]
    #augmented.show()

    images = [i] + augmented
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.show()


if __name__ == "__main__":
    main()