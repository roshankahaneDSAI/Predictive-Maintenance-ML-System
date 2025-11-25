""" In this code we are trying to check whether there is a successful connection to mongodb or not"""


from pymongo import MongoClient

uri = "mongodb://localhost:27017"

client = MongoClient(uri)

try:
    client.admin.command("ping")
    print("Ping successful! Connected to local MongoDB.")
except Exception as e:
    print("Error:", e)
