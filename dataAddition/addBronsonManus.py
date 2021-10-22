import os
import pandas as pd
import cv2
import numpy as np

def downloadImage(plate, imageName, manuscript, sftp):



    folder = plate

    imagesFolder = r"/home/olya/Documents/fragmentsData/DSS_Joins_Test_bw"


    if  os.path.exists(os.path.join(imagesFolder, manuscript, imageName)):
        return

    remote_folder = r"/specific/netapp5_wolf/nachumd/home/nachumd/DSS/fragments_w"

    try:
        allImages = sftp.listdir(r"{}/{}".format(remote_folder, folder))

        maxLen = 0
        imageToGet = ""
        for n in allImages:
            currLen = len(os.path.commonprefix([n, imageName]))
            if currLen >= maxLen:
                imageToGet = n
                maxLen = currLen

        with sftp.open('{}/{}/{}'.format(remote_folder, folder, imageToGet)) as f:
            print("download {}/{}/{}".format(remote_folder, folder, imageToGet))
            img = cv2.imdecode(np.fromstring(f.read(), np.uint8), 1)
            if not os.path.exists(os.path.join(imagesFolder, manuscript)):
                os.mkdir(os.path.join(imagesFolder, manuscript))
            cv2.imwrite(os.path.join(imagesFolder, manuscript, imageToGet), img)
    except:
        print("could not get {}".format(imageName))
        pass


def downloadBoundary(plate, imageName, manuscript, sftp):



    folder = plate

    imagesFolder = r"/home/olya/Documents/fragmentsData/DSS_Joins_Test_boundaries"


    if  os.path.exists(os.path.join(imagesFolder, manuscript, imageName)):
        return

    remote_folder = r"/specific/netapp5_wolf/nachumd/home/nachumd/DSS/DSS_Fragments/boundaries/fragments_nojp"

    try:
        allImages = sftp.listdir(r"{}/{}".format(remote_folder, folder))

        maxLen = 0
        imageToGet = ""
        for n in allImages:
            currLen = len(os.path.commonprefix([n, imageName]))
            if currLen >= maxLen:
                imageToGet = n
                maxLen = currLen

        with sftp.open('{}/{}/{}'.format(remote_folder, folder, imageToGet)) as f:
            print("download {}/{}/{}".format(remote_folder, folder, imageToGet))

            if not os.path.exists(os.path.join(imagesFolder, plate)):
                os.mkdir(os.path.join(imagesFolder, plate))
            toWrite = open(os.path.join(imagesFolder, plate, imageToGet), 'wb')
            toWrite.write(f.read())
            toWrite.close()
    except:
        print("could not get {}".format(imageName))
        pass



def main():
    import paramiko
    s = paramiko.SSHClient()
    s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    s.connect("rack-nachum1.cs.tau.ac.il", 22, username="olyasirkin", password='1QAZ2wsx', timeout=4)
    sftp = s.open_sftp()


    for manu in os.listdir(r'/home/olya/Documents/fragmentsData/DSS_Joins_Test'):
        manusDir = os.path.join(r'/home/olya/Documents/fragmentsData/DSS_Joins_Test', manu)
        for f in os.listdir(manusDir):
            splited = f.split("-")
            plate = splited[0]
            # namestart = splited[0]+"-"+splited[1]+"-"+splited[2]
            #downloadImage(plate, f, manu, sftp)
            downloadBoundary(plate, f, manu, sftp)


    # s = "/home/olya/Documents/fragmentsData/SQE_image_to_manuscript_list.csv"
    # df = pd.read_csv(s)
    # filenames = df["filename"].tolist()
    # manuscript = df["manuscript"].tolist()
    # for f, m in zip(filenames, manuscript):
    #     if "924" not in f or "-R-" not in f:
    #         continue
    #     splited= f.split("-")
    #     plate = splited[0]
    #     #namestart = splited[0]+"-"+splited[1]+"-"+splited[2]
    #     downloadImage(plate, f, m, sftp)



    print("")


if __name__ == "__main__":
    main()