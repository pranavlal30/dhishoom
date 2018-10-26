#!/usr/bin/env python
# coding: utf-8

# In[10]:


import tensorflow as tf


# In[11]:


#Initialize a reader
reader = tf.TFRecordReader()

#Initialize a filename queue
filename_queue = tf.train.string_input_producer(['/Users/pranavlal/Documents/Big Data /Project/Data/audioset_v1_embeddings/eval/_l.tfrecord'])

#Read the file
serialized_keys, serialized_example = reader.read(filename_queue)

#Create dictionary for context and features list

context_ftrs = {
    'video_id': tf.FixedLenFeature([], dtype=tf.string),
    'start_time_seconds': tf.FixedLenFeature([], dtype=tf.float32),
    'end_time_seconds': tf.FixedLenFeature([], dtype=tf.float32),
    'labels': tf.VarLenFeature(dtype=tf.int64)}

features_list_features = {
    'audio_embedding': tf.VarLenFeature(dtype=tf.string)}

#Read the context and feautures list based on above dictionary
read_context, read_sequence = tf.parse_single_sequence_example(serialized=serialized_example,
                                                               context_features=context_ftrs,
                                                               sequence_features=features_list_features)


# In[12]:


#Create a batch to read through all contexts in the file

context_batch = dict(zip(read_context.keys(),
    tf.train.batch(read_context.values(), batch_size=1)))


# In[13]:


#TESTING WITH batch of 1, printing just the first context

with tf.Session() as sess:
  sess.run(tf.initialize_local_variables())
  tf.train.start_queue_runners()
  i = 0 

  while i < 1:
        data_batch = sess.run(context_batch.values())
        print(data_batch)
        i = i + 1
      # process data
  pass


# In[ ]:


#Printing all contexts:

with tf.Session() as sess:
  sess.run(tf.initialize_local_variables())
  tf.train.start_queue_runners()
  try:
    while True:
      data_batch = sess.run(context_batch.values())
      # process data
      print(data_batch)
  except tf.errors.OutOfRangeError:
    pass


# In[ ]:




