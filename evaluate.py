#Use the evaluate to compare target translation with predicted translation

def evaluate(inputs):
    
    #need to reshape input tensor first..
    inputs = tf.reshape(inputs,(1,inputs.shape[0]))

    #run FF
    enc_initial_states = encoder.init_states(1)

    encoder_outputs, hstates, cstates = encoder(inputs, enc_initial_states)
    encoder_states = [hstates,cstates]
    decoder_states = encoder_states

    decoder_input = tf.expand_dims([tokenizer_fre.word_index['<start>']], 0)
    translation = []


    for t in range(max_len_target):
        predictions, state_h, state_c = decoder(decoder_input, decoder_states)
        predicted_id = tf.argmax(predictions[0]).numpy()
        translation.append(tokenizer_fre.index_word[predicted_id])

        if translation[-1] == '<end>' or len(translation) > max_len_target:
          translation = translation[:-1]
          break
        
        decoder_input = tf.expand_dims([predicted_id], 0)

    

    word_translation =  translation


    return word_translation




def getFrenchLabel(tensor):
  fre_label = []
  for j in tensor:
    if j.numpy() != 0:
      fre_label.append(tokenizer_fre.index_word[j.numpy()])
  fre_label = [fre_label[1:-1]]
  return fre_label

