from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["MACHINE"]
collection = db["machineData"]

print("Total docs:", collection.count_documents({}))
print("One sample doc:", collection.find_one())
