import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import numpy as np
if __name__ == "__main__":
    sc = SparkContext(appName="PythonStreamingDirectKafkaWordCount")
    ssc = StreamingContext(sc, 2)
    brokers = 'localhost:9092'
    topic = 'test2'
    kafkaParams = {"metadata.broker.list": brokers}
    kvs = KafkaUtils.createDirectStream(ssc, [topic], kafkaParams, valueDecoder=lambda x: x)
    wav_data = kvs.map(lambda x: np.loads(x[1]))
    print type(wav_data)
    print 'asdfsf'
    wav_data.pprint()
    ssc.start()
    ssc.awaitTermination()
