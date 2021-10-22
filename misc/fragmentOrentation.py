



def main():
    from PIL import Image
    import curses
    import psutil
    import sys
    folder= r"/home/olya/Documents/fragmentsData/DSS_Joins_Test"

    import pandas as pd

    import os
    import numpy as np
    allMap = {}
    #allMap1 = pd.read_csv("fragmentOrentTrain.csv").transpose().to_dict()
    #for k,v in allMap1.items():
    #    allMap[v["Unnamed: 0"]] = v["orentation"]
    i=0
    cnt =0
    for dir in os.listdir(folder):
        #if cnt>3:
        #    break
        currDir  =os.path.join(folder, dir)
        for f in os.listdir(currDir):
            #if cnt >3:
            #    break
            if f.endswith("npy") :
                continue
            if f in allMap:
                cnt+=1
                continue
            try:
                p = os.path.join(currDir, f)
                with Image.open(p) as i:
                    i.resize((500,500)).show()

                    #stdscr = curses.initscr()
                    c = input()
                    #curses.endwin()
                    allMap[f] = c
                    cnt+=1
                    print( "{}_{}_{}".format( cnt, dir, f))
                for proc in psutil.process_iter():
                    if proc.name() == "display":
                        proc.kill()
            except Exception as e:
                print("error in ".format(f))
                df = pd.DataFrame(allMap, index=["orentation"]).transpose()
                df.to_csv("fragmentOrentTest.csv")
                raise  e

    df = pd.DataFrame(allMap, index=["orentation"]).transpose()
    df.to_csv("fragmentOrentTest.csv")



if __name__ =="__main__":
    main()