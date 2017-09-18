import MySQLdb
import matplotlib.pyplot as plt

db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="root",         # your username
                     passwd="password",  # your password
                     db="dejavu")        # name of the data base

# you must create a Cursor object. It will let
#  you execute all the queries you need
cur = db.cursor()

# Use all the SQL you like
cur.execute("SELECT HEX(hash),offset FROM fingerprints where song_id=4")

# print all the first cell of all the rows
hashes=[]
offets=[]
for row in cur.fetchall():
	k=row[0]
	bb=int(k,16)
	hashes.append(bb)
	offets.append(int(row[1]))
print hashes[:10],offets[:10]
plt.scatter(offets,hashes)
plt.show()
db.close()