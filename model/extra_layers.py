# used for model 1

# get the avg length of words between the sentences 
from dataset.data import *
sentences = [len(sentence) for sentence in X_train]

output_sequence_len_sentences = int(np.mean(sentences))

# import text vectorization layer 
from keras.layers import TextVectorization
vectorization = TextVectorization(
    max_tokens=1000,
    output_sequence_length= output_sequence_len_sentences, # add the avg number of words in the sentences of the training data
    pad_to_max_tokens=True, # add 0 as padding in the sentences (to match the output_sequence_length)

)

# adapt the training data
vectorization.adapt(X_train)


# get the vocabulary found in the vectorization layer  
vocabulary = vectorization.get_vocabulary()

# get the vocabulary size 
vocab_size = len(vocabulary)

# import Embedding layer 
from keras.layers import Embedding
embedding = Embedding(
    input_dim=vocab_size,
    output_dim=256
)








