#import sys
#from pyspark import SparkContext
#from pyspark.streaming import StreamingContext
#from pyspark.streaming.kafka import KafkaUtils
from kafka import KafkaProducer
from scipy.io import wavfile


bootstrap_servers=['localhost:9092']
topic = 'test2'

def publish_message(producer_instance, topic_name, value):
    try:
        producer_instance.send(topic_name, key='Sensor1', value=value.dumps())
        producer_instance.flush()
        print('Message published successfully.')
    except Exception as ex:
        print('Exception in publishing message')
        print ex


def connect_kafka_producer():
    _producer = None
    try:
        _producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
    except Exception as ex:
        print('Exception while connecting Kafka')
        print(ex)
    finally:
        return _producer

    
if __name__ == "__main__":

    kafka_producer = connect_kafka_producer()

    wave_file = '/Users/pranavlal/Documents/Big_Data/Project/dhishoom/VGG/wavefiles/gunshots/Adwb_rxQRXI_590.000.wav'
    sr, wav_data = wavfile.read(wave_file)
    publish_message(kafka_producer, topic, wav_data)
    print sr



    

