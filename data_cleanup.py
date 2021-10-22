import os



def getNumberOfLetters(filename):
    filename = filename.split("/")[-1]
    plate = filename.split("-")[0]

    textFolder = r'/home/olya/Documents/fragmentsData/texts'
    folderPath = os.path.join(textFolder, plate)
    if not os.path.exists(folderPath):
        return -1
    allImages = os.listdir(folderPath)

    maxLen = 0
    imageToGet = ""
    for n in allImages:
        currLen = len(os.path.commonprefix([n, filename]))
        if currLen >= maxLen:
            imageToGet = n
            maxLen = currLen

    if maxLen < 11:
        return -1
    with open(os.path.join( folderPath, imageToGet), 'r') as f:
        allData = f.read()
    return len(allData)


def manually_remove():
    from PIL import Image
    import psutil
    allManuDir = r"/home/olya/Documents/fragmentsData/DSS_Joins"
    idx =0
    start = False
    for f in reversed(os.listdir(allManuDir)):
        # if f == '4Q22':
        #     start = True
        # if not start:
        #     continue

        fullManuath = os.path.join( allManuDir, f)
        if os.path.isdir(os.path.join( fullManuath)):
            # if len(os.listdir(fullManuath))==0:
            #     os.rmdir(fullManuath)
            #     continue

            for f1 in os.listdir(fullManuath):
                fullPath = os.path.join(fullManuath, f1)
                idx+=1
                with Image.open(fullPath) as image:
                    image = image.resize((500,500))
                    image.show()
                nb = input('D for delete, N for next')
                for proc in psutil.process_iter():
                    if proc.name() == "display":
                        proc.kill()
                if nb == 'd':
                    os.remove(fullPath)
                    print("removed {}".format(fullPath))
                else:
                    print("{} keep {} ".format(idx, fullPath))


def  findSameManus():
    from collections import defaultdict
    import shutil
    import random
    file2Manu = defaultdict(list)
    manu2Files = defaultdict(list)
    allManuDir = r"/home/olya/Documents/fragmentsData/DSS_Joins"

    allManus = os.listdir(allManuDir)
    randomTest = random.sample(allManus, 100)
    for t in randomTest:
        shutil.move( os.path.join(allManuDir, t), os.path.join(r"/home/olya/Documents/fragmentsData/DSS_Joins_Test", t) )



    for f in os.listdir(allManuDir):

        fullManuath = os.path.join(allManuDir, f)
        if os.path.isdir(os.path.join(fullManuath)):
            for f1 in os.listdir(fullManuath):
                file2Manu[f1].append(f)
                manu2Files[f].append(f1)
    #sameFolders = set()
    for k,v in file2Manu.items():
         # if len(v)>1:
         #     print(k, v)
            for m in v:
                for m2 in v:
                    if m==m2:
                        continue
                    l1 = manu2Files[m]
                    l2 =manu2Files[m2]
                    if set(l1).issubset(set(l2)):
                        if os.path.exists(os.path.join(allManuDir, m)):
                            print("removing {} because of {}".format(os.path.join(allManuDir, m), m2))
                            shutil.rmtree(os.path.join(allManuDir, m))
                    elif set(l2).issubset(set(l1)):
                        if os.path.exists(os.path.join(allManuDir, m2)):
                            print("removing {} because of {}".format(os.path.join(allManuDir, m2), m))
                            shutil.rmtree(os.path.join(allManuDir, m2))



def main():

    findSameManus()

    #manually_remove()
    allManuDir = r"/home/olya/Documents/fragmentsData/DSS_Joins"
    for f in os.listdir(allManuDir):
        fullManuath = os.path.join( allManuDir, f)
        if os.path.isdir(os.path.join( fullManuath)):
            if len(os.listdir(fullManuath))==0:
                os.rmdir(fullManuath)
                continue

            # for f1 in os.listdir(fullManuath):
            #     lettersNum = getNumberOfLetters(f1)
            #     fullPath = os.path.join(fullManuath, f1)
            #     if lettersNum < 5 and lettersNum !=-1:
            #         if not os.path.exists(fullManuath.replace("DSS_Joins_New", "DSS_Joins_New_empty")):
            #             os.mkdir(fullManuath.replace("DSS_Joins_New", "DSS_Joins_New_empty"))
            #         os.replace(fullPath, fullPath.replace("DSS_Joins_New", "DSS_Joins_New_empty"))


if __name__ == "__main__":
    main()