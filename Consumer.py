from time import sleep

from kafka import KafkaConsumer
import numpy as np

bootstrap_servers = ['localhost:9092']
topic_name = 'test2'

def consume(consumer):
    
    for msg in consumer:
        sensor = msg.key

        wave_file = np.loads(msg.value)
        print type(msg)
        print type(msg.value)
        #print msg.value.decode('utf-8')
        print sensor
        print wave_file

if __name__ == '__main__':
    
    consumer = KafkaConsumer(topic_name, bootstrap_servers = bootstrap_servers, consumer_timeout_ms=1000)
    while consumer is not None:
        consume(consumer)
    consumer.close()

