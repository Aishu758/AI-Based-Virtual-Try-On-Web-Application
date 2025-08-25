from keras.models import load_model
from sklearn.externals import joblib
from keras.preprocessing.sequence import pad_sequences

# Load the tokenizer object
tokenizer = joblib.load('tokenizer.joblib')
modelmain = load_model("hup_sentimental_lstm.h5")
print(modelmain.summary())


twt = ['it is bad']
#vectorizing the tweet by the pre-fitted tokenizer instance
twt = tokenizer.texts_to_sequences(twt)
#padding the tweet to have exactly the same shape as `embedding_2` input
twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
print(twt)
sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")
