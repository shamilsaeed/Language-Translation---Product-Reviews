def predictFrench(engtext):

    #preprocess incoming text
    test_seq = tokenizer_eng.texts_to_sequences([engtext])

    test_sequence = tf.keras.preprocessing.sequence.pad_sequences(test_seq, padding='post',maxlen=max_len_input)
    test_sequence = tf.convert_to_tensor(test_sequence)


    #apply FF
    enc_initial_states = encoder.init_states(1)

    encoder_outputs, hstates, cstates = encoder(test_sequence, enc_initial_states)
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

    
    print(' '.join(translation))

    return None
