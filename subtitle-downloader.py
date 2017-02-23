#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Name      : subtitle downloader
# Purpose   : One step subtitle download
#
# Authors   : manoj m j, arun shivaram p, Valentin Vetter, niroyb
# Edited by : Valentin Vetter
# Created   :
# Copyright : (c) www.manojmj.com
# Licence   : GPL v3
#-------------------------------------------------------------------------------

# TODO: use another DB if subs are not found on subDB
import hashlib
import os
import shutil 
import sys
import logging
import requests,time,re,zipfile
from bs4 import BeautifulSoup
PY_VERSION = sys.version_info[0]
if PY_VERSION == 2:
    import urllib2
if PY_VERSION == 3:
    import urllib.request


def get_hash(file_path):
    read_size = 64 * 1024
    with open(file_path, 'rb') as f:
        data = f.read(read_size)
        f.seek(-read_size, os.SEEK_END)
        data += f.read(read_size)
    return hashlib.md5(data).hexdigest()


def sub_downloader(file_path):
    # Put the code in a try catch block in order to continue for other video files, if it fails during execution
    try:
        # Skip this file if it is not a video
        root, extension = os.path.splitext(file_path)
        if extension not in [".avi", ".mp4", ".mkv", ".mpg", ".mpeg", ".mov", ".rm", ".vob", ".wmv", ".flv", ".3gp",".3g2"]:
            return

        if not os.path.exists(root + ".srt"):
            headers = {'User-Agent': 'SubDB/1.0 (subtitle-downloader/1.0; http://github.com/manojmj92/subtitle-downloader)'}
            url = "http://api.thesubdb.com/?action=download&hash=" + get_hash(file_path) + "&language=en"
            if PY_VERSION == 3:
                req = urllib.request.Request(url, None, headers)
                response = urllib.request.urlopen(req).read()
            if PY_VERSION == 2:
                req = urllib2.Request(url, '', headers)
                response = urllib2.urlopen(req).read()

            with open(root + ".srt", "wb") as subtitle:
                subtitle.write(response)
                logging.info("Subtitle successfully downloaded for " + file_path)
    except:
        #download subs from subscene if not found in subdb  
        sub_downloader2(file_path)
def sub_downloader2(file_path):
    try:
        root, extension = os.path.splitext(file_path)
        if extension not in [".avi", ".mp4", ".mkv", ".mpg", ".mpeg", ".mov", ".rm", ".vob", ".wmv", ".flv", ".3gp",".3g2"]:
            return  
        if os.path.exists(root + ".srt"):
            return
        j=-1
        root2=root
        for i in range(0,len(root)):
            if(root[i]=="\\" or root[i] =="/"):
                j=i
                break
        root=root2[j+1:]
        root2=root2[:j+1]
        r=requests.get("http://subscene.com/subtitles/release?q="+root);
        soup=BeautifulSoup(r.content,"lxml")
        atags=soup.find_all("a")
        href=""
        for i in range(0,len(atags)):
            spans=atags[i].find_all("span")
            if(len(spans)==2 and spans[0].get_text().strip()=="English"):
                href=atags[i].get("href").strip()               
        if(len(href)>0):
            r=requests.get("http://subscene.com"+href);
            soup=BeautifulSoup(r.content,"lxml")
            lin=soup.find_all('a',attrs={'id':'downloadButton'})[0].get("href")
            r=requests.get("http://subscene.com"+lin);
            soup=BeautifulSoup(r.content,"lxml")
            subfile=open(root2+".zip", 'wb')
            for chunk in r.iter_content(100000):
                subfile.write(chunk)
                subfile.close()
                time.sleep(1)
                zip=zipfile.ZipFile(root2+".zip")
                zip.extractall(root2)
                zip.close()
                os.unlink(root2+".zip")
                shutil.move(root2+zip.namelist()[0], os.path.join(root2, root + ".srt"))
    except:
        #Ignore exception and continue
        print("Error in fetching subtitle for " + file_path)
        print("Error", sys.exc_info())
        logging.error("Error in fetching subtitle for " + file_path + str(sys.exc_info()))


def main():
    root, _ = os.path.splitext(sys.argv[0])
    logging.basicConfig(filename=root + '.log', level=logging.INFO)
    logging.info("Started with params " + str(sys.argv))

    if len(sys.argv) == 1:
        print("This program requires at least one parameter")
        sys.exit(1)

    for path in sys.argv:
        if os.path.isdir(path):
            # Iterate the root directory recursively using os.walk and for each video file present get the subtitle
            for dir_path, _, file_names in os.walk(path):
                for filename in file_names:
                    file_path = os.path.join(dir_path, filename)
                    sub_downloader(file_path)
        else:
            sub_downloader(path)

if __name__ == '__main__':
    main()
