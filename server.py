from flask import Flask, abort, request, redirect, url_for
from flask import render_template,jsonify
from data_cleaner import DataCleaner
from vn_gen import tokenize_tunning
import tensorflow as tf
import pickle as pk 
import numpy as np 
from sklearn_crf import NerCrf
from pyvi import ViPosTagger,ViTokenizer
import json
from flask_pymongo import PyMongo
from pymongo import MongoClient
import datetime
now = datetime.datetime.now()
unknown_file_name = './unknown/' + str(now.day) + str(now.month) + str(now.year) +'.txt'
fail_file_name = './fail/' + str(now.day) + str(now.month) + str(now.year) +'.txt'
"""
read word2vec trained model
"""
def read_trained_data(file_trained_data):
    with open(file_trained_data,'rb') as input_file :
        vectors = pk.load(input_file)
        word2int = pk.load(input_file)
        int2word = pk.load(input_file)
    return vectors, word2int, int2word
# client = MongoClient('mongodb://localhost:27017/')
# mydb = client.test_database_1
app = Flask(__name__)
# app.config['MONGO_DBNAME'] = 'FinancialBotDb'
# app.config["MONGO_URI"] = "mongodb://localhost:27017/FinancialBotDb"
# mongo = PyMongo(app)
def text_classify(content):
    content = content.lower()
    input_size = 16
    window_size = 2
    embedding_dim = 50
    batch_size_word2vec = 8
    file_to_save_word2vec_data = 'word2vec_ver6/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size-' + str(batch_size_word2vec) + '.pkl'
    data_cleaner = DataCleaner(content)
    print("data_cleaner",data_cleaner)
    all_words = data_cleaner.separate_sentence()     
    print("all_words",all_words)
    vectors, word2int, int2word = read_trained_data(file_to_save_word2vec_data)
    
    data_x_raw = []
    for word in all_words:
        if word in word2int:
            data_x_raw.append(vectors[word2int[word]])
        else:
            intent = 'unknown'
            return intent,all_words
    for k in range(input_size - len(data_x_raw)):
        padding = np.zeros(embedding_dim)
        data_x_raw.append(padding)
    data_x =[]
    data_x.append(data_x_raw)
    int2intent = {0: 'end', 1: 'trade', 2: 'cash_balance', 3: 'advice', 4: 'order_status', 5: 'stock_balance', 6: 'market',7: 'cancel'}
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        #First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph('ANN_ver6/ws-2-embed-50batch_size_w2c-8batch_size_cl8.meta')
        saver.restore(sess,tf.train.latest_checkpoint('ANN_ver6/'))
        # Access and create placeholders variables and
        # print (sess.run ('x:0'))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        print (sess.graph.get_operations())
        # Access the op that you want to run. 
        prediction = graph.get_tensor_by_name("prediction/Softmax:0")
        pred = (sess.run(prediction,{x:data_x}))
        corr_pred = tf.reduce_max(pred)
        index = tf.argmax(pred, axis=1, name=None)
        # print (sess.run(pred))
        print (sess.run(corr_pred))
        print (sess.run(index))
        intent = int2intent[sess.run(index)[0]]
    return intent,all_words

def named_entity_reconignition(content,intent):
    content = content.lower()
   # print("content = ",content)
    check_sw = DataCleaner()
    ner = read_ner_model()
    raw = check_sw.remove_stopword_sent(content)
    tokens = tokenize_tunning(raw)
    y_pred,y_test = ner.test(tokens)
    data = []
    side = ""
    price = "" 
    quantity = ""
    symbol = ""

    for i in range(len(tokens[0])):
        if y_pred[0][i] == 'side-B':
            side = 'B'
        elif y_pred[0][i] == 'side-S':
            side = 'S'
        elif y_pred[0][i] == 'price':
            price = tokens[0][i]
        elif y_pred[0][i] == 'quantity':
            quantity = tokens[0][i]
            
        elif y_pred[0][i] == 'symbol':
            symbol = tokens[0][i]
       # print(1)
        data.append([tokens[0][i],y_pred[0][i]])
    print("data",data)
    
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
    return json_data,data
    # return data
@app.route('/')
def index():
    # name = request.args.get('name')
    return render_template('home.html')
@app.route('/check/unknown')
def check_unknown():    
    print ('call API check OK')
    # print (request.json)
    # texts = read_error()
    return render_template('unknown.html')
@app.route('/check/fail')
def check_fail():    
    print ('call API check OK')
    # print (request.json)
    # texts = read_error()
    return render_template('fail.html')
"""
get all word not in vocab in sentences that have unknown intent
"""
@app.route('/word',methods = ['GET'])
def word_notin_vocab():
    # read vocab
    print ('start getting /word ')
    input_size = 16
    window_size = 2
    embedding_dim = 50
    batch_size_word2vec = 8
    file_to_save_word2vec_data = 'word2vec_ver6/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size-' + str(batch_size_word2vec) + '.pkl'
    vectors, word2int, int2word = read_trained_data(file_to_save_word2vec_data)
    # read all sentences in unknown file
    texts = []
    print ("Current day: %d" % now.day)
    print ("Current year: %d" % now.year)
    print ("Current month: %d" % now.month)
    
    with open(unknown_file_name, encoding="utf8") as file:
        for line in file :
            temp = line.split(",",1)
            temp[1] = temp[1].lower()
            texts.append(temp[1])  #list of train_word
    words_notin_vocab = []
    for text in texts:
        data_cleaner = DataCleaner(text)
        all_words = data_cleaner.separate_sentence()   
        for word in all_words:
            if word not in word2int:
                words_notin_vocab.append(word)
    return jsonify(results = words_notin_vocab) 
@app.route('/falseIntent',methods = ['GET'])
def getFalseIntent():
    texts = read_error()
    return jsonify(results = texts)
"""
Intent classify
"""
@app.route('/genIntent', methods=['POST'])
def nlu():    
    print ('call API submit OK')
    # print (request.json)
    content = request.form['text']
    print (content)
    content = content.lower()
    print (content)
    intent,all_words = text_classify(content)
    if(intent == "unknown"):
        save_to_database(1,content,intent)
    print ('intent: ',intent)
    print ('all_words: ',all_words)
    outputs,data = named_entity_reconignition(content,intent)
    # return jsonify(outputs=outputs)
    return render_template('home.html', content = content,intent = intent, outputs = data, all_words = all_words)
"""
submit if the classifier make false intent
"""
@app.route('/submitError', methods=['POST'])
def submitError():    
    print ('call API check OK')
    # print (request.json)
    content = request.form['text']
    intent = request.form['intent']
    if(intent == "unknown"):
        save_to_database(1,content,intent)
    else:
        save_to_database(2,content,intent)
    # texts = read_error()
    reply = "Cảm ơn bạn đã góp ý!"
    return render_template('home.html',reply = reply)

@app.route('/checkError', methods=['POST'])
def checkError():    
    print ('call API check OK')
    # print (request.json)
    content = request.form['content']
    intent = request.form['intent']
    save_to_database(content,intent)
    texts = read_error()
    return render_template('check.html',texts = texts)

@app.route('/getintents/api', methods=['POST'])
def understand_language():    
    print ('call API OK')
    print (request.json)
    content = request.json['text']
    print (content)
    content = content.lower()
    print (content)
    intent,all_words = text_classify(content)
    if(intent == "unknown"):
        save_to_database(1,content,intent)
    outputs,data = named_entity_reconignition(content,intent)
    print ('data')
    print (data)
    return jsonify(outputs=outputs)
@app.route('/submitNLUError', methods=['POST'])
def check_understand_language():    
    print ('call API OK')
    print (request.json)
    content = request.json['content']
    check = request.json['check']
    print (content)
    content = content.lower()
    print (content)
    intent,all_words = text_classify(content)
    if (check == "Sai"):
        print ('False 172')
        save_to_database(2,content,intent)
        # save_to_database(content,intent)
    # outputs,data = named_entity_reconignition(content,intent)
    reply = "Cảm ơn bạn đã góp ý!"
    return jsonify(reply = reply)

def save_to_database(index,sentence,intent):
    print ("start storing")
    if (index == 1):
        file = open(unknown_file_name,'a+', encoding="utf8")
    else:
        file = open(fail_file_name,'a+', encoding="utf8")
    file.write(intent + ',' + sentence+'\n')
def read_error():
    texts = []
    with open(fail_file_name, encoding="utf8") as file:
        for line in file :
            texti = []
            temp = line.split(",",1)
            temp[1] = temp[1].lower()
            texti.append(temp[1])  #list of train_word
            texti.append(temp[0])
            texts.append(texti)
    return texts
def pretrain():
    input_size = 16
    window_size = 2
    embedding_dim = 50
    batch_size_word2vec = 8
    file_to_save_word2vec_data = 'word2vec_ver6/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size-' + str(batch_size_word2vec) + '.pkl'
    vectors, word2int, int2word = read_trained_data(file_to_save_word2vec_data)
    int2intent = {0: 'end', 1: 'trade', 2: 'cash_balance', 3: 'advice', 4: 'order_status', 5: 'stock_balance', 6: 'market',7: 'cancel'}
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        #First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph('ANN_ver6/ws-2-embed-50batch_size_w2c-8batch_size_cl8.meta')
        saver.restore(sess,tf.train.latest_checkpoint('ANN_ver6/'))
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
def read_ner_model():
    ner = pk.load(open('./ner/crf_model.pkl','rb'))
    return ner
   


if __name__ == '__main__':
    app.run(debug=True, host= '0.0.0.0',port=5000)