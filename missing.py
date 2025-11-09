
import os, glob
root="/home/chandemonium/Downloads/Dataset"; NC=36
for split in ("train","valid","test"):
    bad=[]
    for lbl in glob.glob(os.path.join(root,split,"labels","*.txt")):
        with open(lbl) as f:
            for i,line in enumerate(f,1):
                s=line.strip().split()
                if not s: continue
                if len(s)!=5: bad.append((lbl,i,"len!=5",line.strip())); continue
                try:
                    c=int(s[0]); x,y,w,h=map(float,s[1:])
                except: bad.append((lbl,i,"parse",line.strip())); continue
                if not (0<=c<NC): bad.append((lbl,i,"class",c))
                if not all(0<=v<=1 for v in (x,y,w,h)): bad.append((lbl,i,"range",(x,y,w,h)))
    print(split,"bad lines:",len(bad))
    for r in bad[:5]: print("  ",r)

