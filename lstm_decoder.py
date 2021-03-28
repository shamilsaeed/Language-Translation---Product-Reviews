class Decoder(Model):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = Embedding(vocab_size, embedding_size)
        self.lstm = LSTM(hidden_size, return_sequences=True, return_state=True)
        self.dense = Dense(vocab_size, activation='softmax')

    def call(self,inputs,hidden):
        embedding = self.embedding(inputs)
        output, hstate, cstate = self.lstm(embedding,hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        output = self.dense(output)
        return output, hstate, cstate

    def init_states(self, batch_size):
        hinitial = tf.zeros([batch_size, self.hidden_size])
        cinitial = tf.zeros([batch_size, self.hidden_size])
      return (hinitial,cinitial)
