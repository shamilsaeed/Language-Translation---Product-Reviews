class Encoder(Model):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Encoder,self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = Embedding(vocab_size,embedding_size)
        self.dropout = Dropout(0.2)
        self.lstm = LSTM(hidden_size, return_sequences=True, return_state=True)

    def call(self,inputs,hidden):
        embedding = self.embedding(inputs)
        embedding = self.dropout(embedding)
        output, hstate, cstate = self.lstm(embedding,initial_state=hidden)
        return output, hstate, cstate

    def init_states(self, batch_size):
        hinitial = tf.zeros([batch_size, self.hidden_size])
        cinitial = tf.zeros([batch_size, self.hidden_size])
      return (hinitial,cinitial)
