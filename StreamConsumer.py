import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import numpy as np
from AudioClassifier.parse_file import process_data
def classify(rdd):
    wav_data = np.asarray(rdd.take(1), dtype=np.int16)
    if len(wav_data) > 0:
        wav_data = wav_data.reshape(220500,2)
        prediction = process_data(wav_data)
        print prediction

    

	
if __name__ == "__main__":
    sc = SparkContext(appName="PythonStreamingDirectKafkaWordCount")
    ssc = StreamingContext(sc, 2)
    brokers = 'localhost:9092'
    topic = 'test2'
    kafkaParams = {"metadata.broker.list": brokers}
    audioStream = KafkaUtils.createDirectStream(ssc, [topic], kafkaParams, valueDecoder=lambda x: x)
    wav_data = audioStream.map(lambda x: np.loads(x[1]))
    pred = wav_data.foreachRDD(lambda rdd: classify(rdd))
    #predictions = process_data(wav_data)
    ssc.start()
    ssc.awaitTermination()
