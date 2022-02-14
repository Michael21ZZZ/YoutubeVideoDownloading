# coding: utf-8
from pytube import YouTube
import os

BASE_URL = 'https://www.youtube.com/watch?v='


def get_downloadURLSet(filename):
    URL_set = set()
    with open(filename) as f:
        for eachURL in f.readlines():
            eachURL = eachURL.rstrip('\n')
            URL_set.add(BASE_URL + eachURL)
            # print(eachURL)
    return URL_set


def download_video(downloadURLSet):
    if len(downloadURLSet) == 0:
        print("no video to download")
        return
    current_number = 0
    for downloadURL in downloadURLSet:
        print(downloadURL)
        try:
            yt = YouTube(downloadURL)
        except:
            print("Some thing wrong about the authority!")
            continue
        name = yt.title
        print("Now is loading %s------------>" % name)
        stream = yt.streams.filter(file_extension='mp4').first()
        stream.download('./videos')
        os.rename(os.path.join('./videos', yt.streams.first().default_filename), os.path.join('./videos', downloadURL.strip(BASE_URL)+'.mp4'))
        print("--------------->%s is loaded!" % name)
        current_number += 1
        # if current_number == 5:
        #     break


def main():
    URL_set = get_downloadURLSet('./videoIDlist.txt')
    download_video(URL_set)


if __name__ == '__main__':
    main()