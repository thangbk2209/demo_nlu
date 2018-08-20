from flask import Flask, abort, request, redirect, url_for
from flask import render_template
from data_cleaner import DataCleaner
import tensorflow as tf
import pickle as pk 
import numpy as np 
from sklearn_crf import NerCrf
from pyvi import ViPosTagger,ViTokenizer
import json
def read_trained_data(file_trained_data):
    with open(file_trained_data,'rb') as input_file :
        vectors = pk.load(input_file)
        word2int = pk.load(input_file)
        int2word = pk.load(input_file)
    return vectors, word2int, int2word

app = Flask(__name__)
def text_classify(content):
    content = content.lower()
    input_size = 16
    window_size = 2
    embedding_dim = 32
    batch_size_word2vec = 4
    file_to_save_word2vec_data = 'word2vec_ver2/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size-' + str(batch_size_word2vec) + '.pkl'
    data_cleaner = DataCleaner(content)
    all_words = data_cleaner.separate_sentence()     
    vectors, word2int, int2word = read_trained_data(file_to_save_word2vec_data)
    data_x_raw = []
    for word in all_words:
        data_x_raw.append(vectors[word2int[word]])
    for k in range(input_size - len(data_x_raw)):
        padding = np.zeros(embedding_dim)
        data_x_raw.append(padding)
    data_x =[]
    data_x.append(data_x_raw)
    int2intent = {0: 'end', 1: 'trade', 2: 'cash_balance', 3: 'advice', 4: 'order_status', 5: 'stock_balance', 6: 'market',7: 'cancel'}
    with tf.Session() as sess:
        #First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph('ANN_2layers_320-64/ws-2-embed-32batch_size_w2c-4batch_size_cl16.meta')
        saver.restore(sess,tf.train.latest_checkpoint('ANN_2layers_320-64/'))
        # Access and create placeholders variables and
        # print (sess.run ('x:0'))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        # print (sess.graph.get_operations())
        # Access the op that you want to run. 
        prediction = graph.get_tensor_by_name("prediction/Softmax:0")
        pred = (sess.run(prediction,{x:data_x}))
        corr_pred = tf.reduce_max(pred)
        index = tf.argmax(pred, axis=1, name=None)
        # print (sess.run(pred))
        print (sess.run(corr_pred))
        print (sess.run(index))
        intent = int2intent[sess.run(index)[0]]
    return intent

def named_entity_reconignition(content,intent):
    content = content.lower()
    ner = read_ner_model()
    y_pred,y_test = ner.test(content)
    s = ViPosTagger.postagging(ViTokenizer.tokenize(content))[0]
    print (s)
    data = []
    side = ""
    price = "" 
    quantity = ""
    symbol = ""

    for i in range(len(s)):
        if y_pred[0][i] == 'side-B':
            side = 'B'
        elif y_pred[0][i] == 'side-S':
            side = 'S'
        elif y_pred[0][i] == 'price':
            price = s[i]
        elif y_pred[0][i] == 'quantity':
            quantity = s[i]
            
        elif y_pred[0][i] == 'symbol':
            symbol = s[i]
        data.append([s[i],y_pred[0][i]])
    
    json_data = {
        
        "entities":{
            "price":price,
            "quantity":quantity,
            "side":side,
            "symbol":symbol,
        },
        "intent": intent,
        "text" : content
    }  
    return json_data
@app.route('/')
def index():
    # name = request.args.get('name')
    return render_template('home.html')
@app.route('/postContent', methods = ['POST'])
def postContent():
    # content = request.form['content']
    # print (content)
    print ('call API OK')
    data = json.loads(request.data.decode())
    content = data["content"]
    intent = text_classify(content)
    json_data = named_entity_reconignition(content,intent)
    json_data = str(json_data)
    print("sd: ",json_data)
    # print (entities)
    

    return json_data
def read_ner_model():
    ner = pk.load(open('./ner/crf_model.pkl','rb'))
    return ner
   


if __name__ == '__main__':
    app.run(debug=True)