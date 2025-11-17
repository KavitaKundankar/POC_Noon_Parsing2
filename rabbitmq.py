import pika
import json
import time

RABBITMQ_HOST = "localhost"
RABBITMQ_PORT = 5672
QUEUE_NAME = "NOON_PARSER_DATA_QUEUE"
USERNAME = "user"
PASSWORD = "password"


def process_message(message: dict):
    """Process the received message."""
    print("Processing Message:", message)
    time.sleep(1)
    print("Message processed successfully\n")


def callback(ch, method, properties, body):
    """Callback triggered when a new message arrives."""
    try:
        message = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        message = {"raw_data": body.decode("utf-8")}
    
    print(f"Received new message from RabbitMQ: {message}")
    process_message(message)

    # ch.basic_ack(delivery_tag=method.delivery_tag)


def start_subscriber():
    """Connect to RabbitMQ and start consuming messages."""
    credentials = pika.PlainCredentials(USERNAME, PASSWORD)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        virtual_host="myvhost",
        credentials=credentials,
        heartbeat=60,
        blocked_connection_timeout=300
    )

    try:
        # Create Connection and consume message

        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        # print(f"channel:" , channel)

        print(f"Waiting for messages in queue: '{QUEUE_NAME}'. To exit press CTRL+C\n")

        channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)

        channel.start_consuming()

    except pika.exceptions.AMQPConnectionError as e:
        print(f"Failed to connect to RabbitMQ: {e}")
    except KeyboardInterrupt:
        print("Stopping subscriber...")
    finally:
        try:
            connection.close()
        except Exception:
            pass


if __name__ == "__main__":
    start_subscriber()
