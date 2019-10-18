
# coding: utf-8

# In[1]:


import pandas as pd

from tensorflow import flags
flags.DEFINE_string("prediction_file", "",
                  	"Prediction file in csv format.")

flags.DEFINE_string("submission_file", "",
                  	"Submission file in csv format.")
FLAGS = flags.FLAGS

# In[17]:


sample_sub = pd.read_csv('../../sub/sample_submission.csv')
pred = pd.read_csv(FLAGS.prediction_file, header=None)


# In[18]:


pred.columns = sample_sub.columns


# In[19]:


sample_sub.set_index('Class', inplace=True)
pred.set_index('Class', inplace=True)


# In[20]:


for cls in sample_sub.index:
    if cls in pred.index:
        sample_sub.loc[cls, 'Segments'] = pred.loc[cls, 'Segments']


# In[24]:


sample_sub.shape


# In[26]:


sample_sub.reset_index().to_csv(FLAGS.submission_file, index=None)

