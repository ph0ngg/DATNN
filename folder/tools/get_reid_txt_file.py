import urllib.parse
import os
from pymongo import MongoClient

#url = "mongodb://{}:{}@{}:{}/{}?authSource=admin".format(MONGO_USER, MONGO_PASS, MONGO_HOST, MONGO_PORT, MONGO_DB)
url = 'mongodb://localhost:27017/'
client = MongoClient(url)
db = client['mydb']

global_id = db['Campus4_gid']
#Creating a pymongo client
#url = "mongodb://{}:{}@{}:{}/{}?authSource=admin".format(MONGO_USER, MONGO_PASS, MONGO_HOST, MONGO_PORT, MONGO_DB)

for cam in os.listdir('/home/phongnn/test/test/Campus4/result_reid'):
    if cam[-5] != 'w':
        f = open('/home/phongnn/test/test/Campus4/result_reid/' + cam[:-4] + '.txt', 'r')
        f2 = open('/home/phongnn/test/test/Campus4/result_reid/' + cam[:-4] +'_new.txt', 'w')
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
            