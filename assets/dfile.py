import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense

# ==========================================
#    CONFIGURATION
# ==========================================
EMBEDDING_DIM = 100
UNITS = 64

print("\n=============================================")
print("      NLP BRAIN DIAGNOSTIC TEST (FIXED)")
print("=============================================\n")

# 1. LOAD TOKENIZERS
print("[1/4] Loading Tokenizers...")
try:
    with open('assets/gloss_tokenizer.pkl', 'rb') as f:
        gloss_tokenizer = pickle.load(f)
    with open('assets/target_tokenizer.pkl', 'rb') as f:
        target_tokenizer = pickle.load(f)
    
    # DETECT START TOKEN
    if '<sos>' in target_tokenizer.word_index:
        start_token = '<sos>'
    elif '<start>' in target_tokenizer.word_index:
        start_token = '<start>'
    else:
        start_token = list(target_tokenizer.word_index.keys())[0]
        
    print(f"   ✅ SUCCESS. Start Token: '{start_token}'")
    
except Exception as e:
    print(f"   ❌ FAILED to load tokenizers. Error: {e}")
    exit()

# 2. DEFINE MODEL
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W1 = Dense(units, name='W1')
        self.W2 = Dense(units, name='W2')
        self.V  = Dense(1, name='V')

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_with_time_axis)))
        attn = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(attn * values, axis=1)
        return context, attn

class Encoder(tf.keras.Model):
    def __init__(self, vocab, embed, units):
        super().__init__()
        self.embedding = Embedding(vocab, embed, name='embedding')
        self.lstm = LSTM(units, return_sequences=True, return_state=True, 
                        dropout=0.2, recurrent_dropout=0.2, name='lstm') 
    def call(self, x):
        x = self.embedding(x)
        out, h, c = self.lstm(x)
        return out, h, c

class Decoder(tf.keras.Model):
    def __init__(self, vocab, embed, units):
        super().__init__()
        self.embedding = Embedding(vocab, embed, name='embedding')
        self.attention = BahdanauAttention(units, name='attention') 
        self.lstm = LSTM(units, return_sequences=True, return_state=True, 
                        dropout=0.2, recurrent_dropout=0.2, name='lstm')
        self.projection = Dense(units, name='dense') 
        self.fc = Dense(vocab, name='fc')

    def call(self, x, enc_out, h, c):
        context, _ = self.attention(h, enc_out)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        x = self.projection(x)
        out, h, c = self.lstm(x, initial_state=[h, c])
        out = tf.reshape(out, (-1, out.shape[2]))
        x = self.fc(out)
        return x, h, c, _

# 3. BUILD & LOAD WEIGHTS
print("[2/4] Building Model & Loading Weights...")
tf.keras.backend.clear_session()
encoder = Encoder(len(gloss_tokenizer.word_index)+1, EMBEDDING_DIM, UNITS)
decoder = Decoder(len(target_tokenizer.word_index)+1, EMBEDDING_DIM, UNITS)

# Dummy call
encoder(tf.zeros((1, 5)))
decoder(tf.zeros((1, 1)), tf.zeros((1, 5, UNITS)), tf.zeros((1, UNITS)), tf.zeros((1, UNITS)))

try:
    encoder.load_weights('assets/best_encoder.weights.h5')
    decoder.load_weights('assets/best_decoder.weights.h5')
    print("   ✅ SUCCESS: Weights loaded.")
except Exception as e:
    print(f"   ❌ FAILED to load weights. Error: {e}")
    exit()

# 4. INFERENCE FUNCTION (FIXED with +1 Padding)
def evaluate(gloss_text):
    gloss_text = str(gloss_text).lower().strip()
    if not gloss_text: return ""

    inputs = []
    for w in gloss_text.split():
        idx = gloss_tokenizer.word_index.get(w)
        if idx: inputs.append(idx)
    
    if not inputs: return "..."

    # ====================================================
    # THE FIX: Add +1 to length to force one '0' padding
    # This gives the Attention mechanism a "stop" signal
    # ====================================================
    inputs = pad_sequences([inputs], maxlen=len(inputs)+1, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    enc_out, enc_h, enc_c = encoder(inputs)
    dec_h, dec_c = enc_h, enc_c

    start_idx = target_tokenizer.word_index.get(start_token)
    dec_input = tf.expand_dims([start_idx], 0)
    
    result = []
    for t in range(20):
        predictions, dec_h, dec_c, _ = decoder(dec_input, enc_out, dec_h, dec_c)
        predicted_id = tf.argmax(predictions[0]).numpy()
        word = target_tokenizer.index_word.get(predicted_id, '?')

        if word == '<eos>' or word == '<end>':
            break
            
        result.append(word)
        dec_input = tf.expand_dims([predicted_id], 0)

    return " ".join(result)

# 5. RUN TESTS
print("\n[3/4] Running Translation Tests...")
test_glosses = [
    "monday sunday",
    "friday",
    "hello yesterday",
    "tomorrow tuesday",
    "mnday tusday",
    "hello  fridy",
    "today is  saturdy",
    "hello today wednesday",
    "monday tuesday friday",
    "hello today monday"
]

print(f"{'INPUT':<25} | {'PREDICTION':<30}")
print("-" * 60)

for gloss in test_glosses:
    try:
        pred = evaluate(gloss)
        print(f"{gloss:<25} | {pred:<30}")
    except Exception as e:
        print(f"{gloss:<25} | ERROR: {e}")

print("\n=============================================")
print("DIAGNOSTIC COMPLETE")