import sys
import os
import shutil
import subprocess



def main():
    allManuDir = r"/home/olya/Documents/fragmentsData/DSS_Joins_Test_bw"
    forBinPath = r'/home/olya/Documents/fragmentsData/DSS_Joins/forBin/tmp'

    for f in os.listdir(allManuDir):

        fullManuath = os.path.join(allManuDir, f)
        if not os.path.exists(fullManuath.replace(f, f+"_bin")):
            os.mkdir(fullManuath.replace(f, f+"_bin"))
        if os.path.isdir(os.path.join(fullManuath)):
            for f1 in os.listdir(fullManuath):
                fullPath = os.path.join(fullManuath, f1)
                finalOutPath = fullPath.replace(f, f + "_bin")
                finalOutPath = finalOutPath[:-3] + "tif"
                if os.path.exists(finalOutPath):
                    continue

                dest = os.path.join(forBinPath, f1 )
                shutil.copy(fullPath, dest)
                process = subprocess.Popen(['matlab', '-nodisplay', '-nosplash', '-nodesktop', '-r', "run('/home/olya/Documents/TL_DSS/segmentation_matlab/apply_binarization.m');exit;" ],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                print(stdout)
                print(stderr)


                binOutput = '/home/olya/Documents/fragmentsData/DSS_Joins/forBin_Bin/tmp/binaries/tmp_sauvola_binary.tif'

                if os.path.exists(binOutput):
                    shutil.copy(binOutput, finalOutPath)
                    os.remove(binOutput)
                    print("done, file output: {}".format(finalOutPath))
                else:
                    print("Error processing {}".format(fullPath))
                if os.path.exists(dest):
                    os.remove(dest)
                if os.path.exists(r"/home/olya/Documents/fragmentsData/DSS_Joins/forBin_Bin/tmp/label/tmp_labels2.bmp"):
                    os.remove(r"/home/olya/Documents/fragmentsData/DSS_Joins/forBin_Bin/tmp/label/tmp_labels2.bmp")

                print("")







if __name__=="__main__":


    main()