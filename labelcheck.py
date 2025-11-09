
import os, glob
root="/home/chandemonium/Downloads/Dataset"
for split in ("train","valid","test"):
    imgs=glob.glob(os.path.join(root,split,"images","*.*"))
    miss=[]
    for im in imgs:
        b=os.path.splitext(os.path.basename(im))[0]
        lbl=os.path.join(root,split,"labels",b+".txt")
        if not os.path.exists(lbl): miss.append(im)
    print(split, "missing_label:", len(miss))
    [print("  ",x) for x in miss[:5]]
