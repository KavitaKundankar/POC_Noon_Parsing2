import pika

credentials = pika.PlainCredentials("user", "password")

parameters = pika.ConnectionParameters(
    host="localhost",
    port=5672,  
    virtual_host="myvhost",
    credentials=credentials
)

try:
    connection = pika.BlockingConnection(parameters)
    print("✅ Connected successfully to RabbitMQ!")
    connection.close()
except Exception as e:
    print("❌ Connection failed:", e)