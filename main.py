import tensorflow as tf
import regex as re
import unicodedata
from tensorflow.keras.layers import Embedding, LSTM, Dense,Dropout
from tensorflow.keras import Model
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os



#Define model parameters
BATCHSIZE = 64
EMBEDDINGSIZE = 256
NUMHIDDEN = 1024
EPOCHS = 45

DATASET_SIZE = len(data_eng)
dataset = tf.data.Dataset.from_tensor_slices((data_eng,data_fre)).shuffle(buffer_size=DATASET_SIZE)

#use 80/20/20 split for train/validation/test
train_size = int(0.8 * DATASET_SIZE)
val_size = int(0.1 * DATASET_SIZE)

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)
val_dataset = test_dataset.take(val_size)
test_dataset = test_dataset.skip(val_size)

#create batches of English-French pair defined by batchsize for LSTM input
train_dataset = train_dataset.batch(BATCHSIZE, drop_remainder=True)
val_dataset = val_dataset.batch(BATCHSIZE, drop_remainder=True)

#define vocabulary size for English and French using our tokenization method
eng_vocab_size = len(tokenizer_eng.word_index)+1
fre_vocab_size = len(tokenizer_fre.word_index)+1

#Instantiate Encoder and Decoder
encoder = Encoder(eng_vocab_size, EMBEDDINGSIZE, NUMHIDDEN)
decoder = Decoder(fre_vocab_size, EMBEDDINGSIZE, NUMHIDDEN)


optimizer = tf.keras.optimizers.Adam()

Num_Batches = int(train_size/BATCHSIZE)
Num_val_Batches = int(val_size/BATCHSIZE)

train_losses = []
train_epoch_losses = []
val_losses = []
val_epoch_losses = []

#setup model checkpoint weights
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


if __name__ == "__main__":

    for e in range(EPOCHS):
        eng_initial_states = encoder.init_states(BATCHSIZE)

        total_loss = 0
        val_total_loss = 0

        for batch, (input_seq, label_seq) in enumerate(train_dataset):
            loss = 0
            with tf.GradientTape() as tape:
                encoder_outputs, hstates, cstates = encoder(input_seq, eng_initial_states)
                encoder_states = [hstates,cstates]
                decoder_states = encoder_states

                decoder_input = tf.expand_dims([tokenizer_fre.word_index['<start>']] * BATCHSIZE, 1)

                for t in range(1,label_seq.shape[1]):
                    predictions,_,_ = decoder(decoder_input, decoder_states)

                    loss += loss_func(label_seq[:, t], predictions)

                    #use teacher forcing
                    decoder_input = tf.expand_dims(label_seq[:, t], 1)

            batch_loss = (loss / int(label_seq.shape[1]))

            #store training batch losses to plot
            train_losses.append(batch_loss.numpy())

            total_loss += batch_loss

                
            variables = encoder.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))


            if batch % 10 == 0:
                print('Epoch {} Batch {} Training Loss {:.4f}'.format(e + 1, batch, batch_loss.numpy()))

        print('Epoch {} Loss {:.4f}'.format(e + 1,total_loss / Num_Batches))

        #store epoch losses
        train_epoch_losses.append((total_loss / Num_Batches).numpy())      

        #save model checkpoint every epoch
        checkpoint.save(file_prefix = checkpoint_prefix)    

    #QUICK VALIDATION

        for batch, (val_input_seq, val_label_seq) in enumerate(val_dataset):

            loss=0

            encoder_outputs, hstates, cstates = encoder(val_input_seq, eng_initial_states)
            encoder_states = [hstates,cstates]
            decoder_states = encoder_states

            decoder_input = tf.expand_dims([tokenizer_fre.word_index['<start>']] * BATCHSIZE, 1)


            for t in range(1,val_label_seq.shape[1]):
                predictions,_,_ = decoder(decoder_input, decoder_states)

                loss += loss_func(val_label_seq[:, t], predictions)

                decoder_input = tf.expand_dims(val_label_seq[:, t], 1)


            batch_loss = (loss / int(val_label_seq.shape[1]))

            val_losses.append(batch_loss.numpy())

            val_total_loss += batch_loss

            if batch % 10 == 0:
                print('Epoch {} Batch {} Validation Loss {:.4f}'.format(e + 1, batch, batch_loss.numpy()))


        print('Epoch {} Validation Loss {:.4f}'.format( e + 1, val_total_loss/Num_val_Batches))
        val_epoch_losses.append((val_total_loss / Num_val_Batches).numpy())


    #Load latest checkpoint Trained Model --if internet goes off, just run from this line to reclaim latest model
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # load encoder and decoder from checkpoint
    encoder = checkpoint.encoder
    decoder = checkpoint.decoder

    #test translations on few statements

    test_inputs = (
    "Hi.",
    "What are you doing?",
    "Can I have a few minutes, please?",
    "Could you close the door, please?",
    "What is this?",
    "He is amazing.",
    "I am sure he is okay.",
    "Tom, you need a life.",
    "I am going to the shop.",
    "I mean this needs to be fixed."
        )


    for i, test_sent in enumerate(test_inputs):
        print(test_sent)
        test_sequence = normalize(test_sent,IsPredict=False)
        predictFrench(test_sequence)



    #Initialize Cumulative BLEU scores
    BLEU1 = []
    BLEU2 = []
    BLEU3 = []
    BLEU4 = []

    #evaluate model using Bleu scores

    for eng,real_fre in test_dataset:
        yactual = getFrenchLabel(real_fre)
        print(yactual)
        ypredict = evaluate(eng)
        print(ypredict)
        print(sentence_bleu(yactual, ypredict,weights=(1, 0, 0, 0)))
        print(sentence_bleu(yactual, ypredict,weights=(0.5, 0.5, 0, 0)))
        print(sentence_bleu(yactual, ypredict,weights=(0.33, 0.33, 0.33, 0)))
        print(sentence_bleu(yactual, ypredict,weights=(0.25, 0.25, 0.25, 0.25)))

        #keep appending bleu scores
        BLEU1.append((sentence_bleu(yactual, ypredict,weights=(1, 0, 0, 0))))
        BLEU2.append((sentence_bleu(yactual, ypredict,weights=(0.5, 0.5, 0, 0))))
        BLEU3.append((sentence_bleu(yactual, ypredict,weights=(0.33, 0.33, 0.33, 0))))
        BLEU4.append((sentence_bleu(yactual, ypredict,weights=(0.25, 0.25, 0.25, 0.25))))
