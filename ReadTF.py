#!/usr/bin/env python
# coding: utf-8

# # EVAL DATASET

# In[2]:


import tensorflow as tf


# In[13]:


sess = tf.InteractiveSession()


# In[14]:


reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['/Users/pranavlal/Documents/Big Data /Project/Data/audioset_v1_embeddings/eval/_l.tfrecord'])

_, serialized_example = reader.read(filename_queue)


# In[15]:


serialized_example


# In[16]:


context_ftrs = {
    'video_id': tf.FixedLenFeature([], dtype=tf.string),
    'start_time_seconds': tf.FixedLenFeature([], dtype=tf.float32),
    'end_time_seconds': tf.FixedLenFeature([], dtype=tf.float32),
    'labels': tf.VarLenFeature(dtype=tf.int64)}
features_list_features = {
    'audio_embedding': tf.VarLenFeature(dtype=tf.string)}


# In[17]:


read_context, read_sequence = tf.parse_single_sequence_example(serialized=serialized_example,
                                                               context_features=context_ftrs,
                                                               sequence_features=features_list_features)


# In[18]:




tf.train.start_queue_runners(sess)

# Print features
for name, tensor in read_context.items():
    print('{}: {}'.format(name, tensor.eval()))


# In[19]:


tf.train.start_queue_runners(sess)

# Print features
for name, tensor in read_sequence.items():
    print('{}: {}'.format(name, tensor.eval()))


# In[ ]:




