import urllib.parse
import os
from pymongo import MongoClient


def get_reid_txt_file():

    MONGO_DB = "mydb"


    url = "mongodb://localhost:27017/"
    client = MongoClient(url)
    db = client[MONGO_DB]
    
    global_id = db['gid']
    #Creating a pymongo client
    for video in os.listdir('./result/result_txt'):
        if video[-5] != 'w':
            f = open('./result/result_txt/' + video[:-4] + '.txt', 'r')
            f2 = open('./result/result_txt/' + video[:-4] +'_new.txt', 'w')
            lines = f.readlines()
            for line in lines:
                try:
                    x = line.split(',')
                    old_id = int(x[1])
                    if old_id == 0:
                        new_id = 0
                    else:
                        new_id = global_id.find_one({"old key": old_id})["new_key"]
                    f2.write(x[0] + ',' + str(new_id) + ',' + x[2] + ',' + x[3] + ',' + x[4] + ',' + x[5] + ', 1, 1, 1')
                    f2.write('\n')
                except:
                    continue
                