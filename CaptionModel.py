import  pandas as pd
import  numpy as np
import  tensorflow as tf
from    tensorflow.keras import layers,Model
from    tqdm import tqdm
from    nltk.translate.bleu_score import sentence_bleu
import  pickle5 as pickle
from    tensorflow.keras.activations import tanh
from    tensorflow.keras.activations import softmax
import  matplotlib.pyplot as plt
from    tensorflow.keras.preprocessing.text import Tokenizer
import  time

# ### Encoder
class Encoder(Model):
    def __init__(self,embed_dim):
        super(Encoder, self).__init__()
        # build your Dense layer with relu activation
        self.dense = tf.keras.layers.Dense(embed_dim, activation='relu')
        
    def call(self, features):
        # extract the features from the image shape: (batch, 8*8, embed_dim)
        features = self.dense(features)
        return features    

# ### Attention model
class Attention_model(Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.units=units
        # build your Dense layer
        self.W1 = tf.keras.layers.Dense(units)
        # build your Dense layer
        self.W2 = tf.keras.layers.Dense(units)
        # build your final Dense layer with unit 1
        # self.V = tf.keras.layers.Dense(1, activation='softmax')
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features shape: (batch_size, 8*8, embedding_dim)
        # hidden shape: (batch_size, hidden_size)
        
        # Expand the hidden shape to shape: (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # build your score funciton to shape: (batch_size, 8*8, units)
        score = tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        # extract your attention weights with shape: (batch_size, 8*8, 1)
        score = self.V(score)
        attention_weights = softmax(score, axis=1)

        # shape: create the context vector with shape (batch_size, 8*8,embedding_dim)
        context_vector = attention_weights * features
        # reduce the shape to (batch_size, embedding_dim)
        # context_vector = tf.reduce_sum(context_vector, axis=1)
        context_vector = tf.reduce_mean(context_vector, axis=1)

        return context_vector, attention_weights        



# In[38]:

# ### Decoder
class Decoder(Model):
    def __init__(self, embed_dim, units, vocab_size, embedding_dim):
        super(Decoder, self).__init__()
        self.units = units
        # iniitalise your Attention model with units
        self.attention = Attention_model(self.units)
        # build your Embedding layer
        self.embed = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        # build your Dense layer
        self.d1 = tf.keras.layers.Dense(self.units)
        # build your Dense layer
        self.d2 = tf.keras.layers.Dense(vocab_size)


    def call(self,x,features, hidden):
        #create your context vector & attention weights from attention model
        context_vector, attention_weights = self.attention(features, hidden)
        # embed your input to shape: (batch_size, 1, embedding_dim)
        embed = self.embed(x)
        # Concatenate your input with the context vector from attention layer. 
        # Shape: (batch_size, 1, embedding_dim + embedding_dim)
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis=-1)
        # Extract the output & hidden state from GRU layer. 
        # Output shape : (batch_size, max_length, hidden_size)
        output, state = self.gru(embed)
        output = self.d1(output)
        # shape : (batch_size * max_length, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2])) 
        # shape : (batch_size * max_length, vocab_size)
        output = self.d2(output)

        return output, state, attention_weights

    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


# # In[34]:


# # In[4]:

# total_test_images = 100
# workspace_path = '/home/pi/WorkSpace'

# text_file = workspace_path + '/lists/captions_' + str(total_test_images) + '.txt'
# list_file = workspace_path + '/lists/images_' + str(total_test_images) + '.txt'

# # In[18]:

# max_tokenized_words = 20000
# MAX_SEQ_LENGTH = 25
# batch_size = 32
# embedding_dim = 256 
# units = 512

# # In[ ]:



# In[ ]:


class CaptionModel(Model):
    def __init__(self, image_size=250, model_path=None):
        self.image_size = image_size
        embedding_dim = 256
        super(CaptionModel, self).__init__()
        if(model_path == None):
            model_path = './saved_model' + '/' + 'caption_i_%d' % (self.image_size)

        with open(model_path + '/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        max_tokenized_words = 5000
        self.MAX_SEQ_LENGTH = 25
        batch_size = 32
        embedding_dim = 256 
        units = 512
        vocab_size = max_tokenized_words + 1

        s = tf.zeros([32, 64, 2048], tf.int32)
        self.encoder=Encoder(embedding_dim)
        self.decoder=Decoder(embedding_dim, units, vocab_size, embedding_dim)
        features = self.encoder(s)
        hidden = self.decoder.init_state(batch_size=batch_size)
        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * batch_size, 1)
        predictions, hidden_out, attention_weights= self.decoder(dec_input, features, hidden)

        self.decoder.load_weights(model_path + "/decoder.h5" )
        self.encoder.load_weights(model_path + "/encoder.h5" )
        self.decoder.summary()
        self.encoder.summary()

    def call(self, features):
        # extract the features from the image shape: (batch, 8*8, embed_dim)
        features = self.dense(features)
        return features    

    def evaluate(self, features):
        attention_features_shape = 64
        attention_plot = np.zeros((self.MAX_SEQ_LENGTH, attention_features_shape))

        hidden = self.decoder.init_state(batch_size=1)

        features = self.encoder(features)

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(self.MAX_SEQ_LENGTH):
            # get the output from decoder
            predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)

            # attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

            #extract the predicted id(embedded value) which carries the max value
            predicted_id = tf.argmax(tf.transpose(predictions))
            predicted_id = predicted_id.numpy()[0]
            # map the id to the word from tokenizer and append the value to the result list
            result.append(self.tokenizer.index_word[predicted_id])

            if self.tokenizer.index_word[predicted_id] == '<end>':
                return result

            dec_input = tf.expand_dims([predicted_id], 0)

        # attention_plot = attention_plot[:len(result), :]
        # return result, attention_plot,predictions
        # print(tf.shape(caption_tensor))
        result=' '.join(result).rsplit(' ', 1)[0]    

        return result
