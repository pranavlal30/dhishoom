#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


data = {
    'Age': 29,
    'Movie': ['The Shawshank Redemption', 'Fight Club'],
    'Movie Ratings': [9.0, 9.7],
    'Suggestion': 'Inception',
    'Suggestion Purchased': 1.0,
    'Purchase Price': 9.99
}

print(data)


# In[3]:


example = tf.train.Example(features=tf.train.Features(feature={
    'Age': tf.train.Feature(
        int64_list=tf.train.Int64List(value=[data['Age']])),
    'Movie': tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[m.encode('utf-8') for m in data['Movie']])),
    'Movie Ratings': tf.train.Feature(
        float_list=tf.train.FloatList(value=data['Movie Ratings'])),
    'Suggestion': tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[data['Suggestion'].encode('utf-8')])),
    'Suggestion Purchased': tf.train.Feature(
        float_list=tf.train.FloatList(
            value=[data['Suggestion Purchased']])),
    'Purchase Price': tf.train.Feature(
        float_list=tf.train.FloatList(value=[data['Purchase Price']]))
}))


# In[4]:


print(example)


# In[5]:


with tf.python_io.TFRecordWriter('/Users/pranavlal/Documents/customer_1.tfrecord') as writer:
    writer.write(example.SerializeToString())


# In[6]:


sess = tf.InteractiveSession()

# Read TFRecord file
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['/Users/pranavlal/Documents/customer_1.tfrecord'])


# In[7]:


print(filename_queue)


# In[8]:


_, serialized_example = reader.read(filename_queue)


# In[9]:


print(serialized_example)


# In[10]:


read_features = {
    'Age': tf.FixedLenFeature([], dtype=tf.int64),
    'Movie': tf.VarLenFeature(dtype=tf.string),
    'Movie Ratings': tf.VarLenFeature(dtype=tf.float32),
    'Suggestion': tf.FixedLenFeature([], dtype=tf.string),
    'Suggestion Purchased': tf.FixedLenFeature([], dtype=tf.float32),
    'Purchase Price': tf.FixedLenFeature([], dtype=tf.float32)}


# In[11]:


read_data = tf.parse_single_example(serialized=serialized_example,
                                    features=read_features)


# In[12]:


tf.train.start_queue_runners(sess)

# Print features
for name, tensor in read_data.items():
    print('{}: {}'.format(name, tensor.eval()))


# In[13]:


movie_1_actors = tf.train.Feature(
  bytes_list=tf.train.BytesList(
    value=[b'Tim Robbins', b'Morgan Freeman']))
movie_2_actors = tf.train.Feature(
  bytes_list=tf.train.BytesList(
    value=[b'Brad Pitt', b'Edward Norton', b'Helena Bonham Carter']))
movie_actors_list = [movie_1_actors, movie_2_actors]       
movie_actors = tf.train.FeatureList(feature=movie_actors_list)

# Short form
movie_names = tf.train.FeatureList(feature=[
    tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[b'The Shawshank Redemption', b'Fight Club']))
])
movie_ratings = tf.train.FeatureList(feature=[
        tf.train.Feature(float_list=tf.train.FloatList(
            value=[9.7, 9.0]))
])


# In[14]:


movies_dict = {
  'Movie Names': movie_names,
  'Movie Ratings': movie_ratings,
  'Movie Actors': movie_actors
}

movies = tf.train.FeatureLists(feature_list=movies_dict)


# In[15]:


customer = tf.train.Features(feature={
    'Age': tf.train.Feature(int64_list=tf.train.Int64List(value=[19])),
})

example = tf.train.SequenceExample(
    context=customer,
    feature_lists=movies)


# In[16]:


print(example)


# In[17]:


with tf.python_io.TFRecordWriter('/Users/pranavlal/Documents/customer_2.tfrecord') as writer:
    writer.write(example.SerializeToString())


# In[20]:


sess.close()


# In[21]:


sess = tf.InteractiveSession()

# Read TFRecord file
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['/Users/pranavlal/Documents/customer_2.tfrecord'])


# In[22]:


_, serialized_example = reader.read(filename_queue)


# In[38]:


context_ftrs = {
    'Age': tf.FixedLenFeature([], dtype=tf.int64)}
features_list_features = {
    'Movie Actors': tf.VarLenFeature(dtype=tf.string),
    'Movie Names': tf.VarLenFeature(dtype=tf.string),
    'Movie Ratings': tf.VarLenFeature(dtype=tf.float32)}



# In[46]:


read_context, read_sequence = tf.parse_single_sequence_example(serialized=serialized_example,context_features=context_ftrs,sequence_features=features_list_features)


# In[47]:


tf.train.start_queue_runners(sess)

# Print features
for name, tensor in read_context.items():
    print('{}: {}'.format(name, tensor.eval()))


# In[48]:


tf.train.start_queue_runners(sess)

# Print features
for name, tensor in read_sequence.items():
    print('{}: {}'.format(name, tensor.eval()))


# # EVAL DATASET

# In[49]:


reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['/Users/pranavlal/Documents/eval.tfrecord'])

_, serialized_example = reader.read(filename_queue)


# In[50]:


serialized_example


# In[58]:


context_ftrs = {
    'video_id': tf.FixedLenFeature([], dtype=tf.string),
    'start_time_seconds': tf.FixedLenFeature([], dtype=tf.float32),
    'end_time_seconds': tf.FixedLenFeature([], dtype=tf.float32),
    'labels': tf.VarLenFeature(dtype=tf.int64)}
features_list_features = {
    'audio_embedding': tf.VarLenFeature(dtype=tf.string)}


# In[59]:


read_context, read_sequence = tf.parse_single_sequence_example(serialized=serialized_example,
                                                               context_features=context_ftrs,
                                                               sequence_features=features_list_features)


# In[60]:


tf.train.start_queue_runners(sess)

# Print features
for name, tensor in read_context.items():
    print('{}: {}'.format(name, tensor.eval()))


# In[61]:


tf.train.start_queue_runners(sess)

# Print features
for name, tensor in read_sequence.items():
    print('{}: {}'.format(name, tensor.eval()))


# In[ ]:




