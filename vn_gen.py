import random
from pyvi import ViTokenizer,ViPosTagger
import re
from data_cleaner import DataCleaner
def tokenize_tunning(tokens):
        new_tokens = []
        new_pos = []
        for i in range(len(tokens[0])):
            print ('==================tokens[0]=======================')
            print (tokens[0])
            if re.search("_dư",tokens[0][i]):
                sym,word = tokens[0][i].split("_",1) 
                new_tokens.append(sym)
                new_tokens.append(word)
                new_pos.append("Np")
                new_pos.append("V")
            elif re.search("mã_",tokens[0][i]):
                word,sym = tokens[0][i].split("_",1)
                if sym == "cổ_phiếu" or sym == "chứng_khoán":
                    new_tokens.append(word) #them chu 'ma' vao 
                    new_tokens.append(sym) #them chu 'co_phieu' vao
                    new_pos.append("N")
                    new_pos.append("N")
                else:#sau ma la ma co phieu : ma_ssi
                    new_tokens.append(word) #them chu 'ma' vao 
                    new_tokens.append(sym) #them chu 'ssi' vao
                    new_pos.append("N")
                    new_pos.append("Np")
            else:
                new_tokens.append(tokens[0][i])
                new_pos.append(tokens[1][i])
        return (new_tokens,new_pos)
class VnGen:
    """
    lop tao du lieu cho crf sk learn
    """
    #vn_sample = open("/home/hoangbao/Desktop/Untitled Folder/vietnam.txt","r")
    #eng_sample = open("/home/hoangbao/Desktop/Untitled Folder/english.txt","r")
    #file_name = /home/hoangbao/Desktop/Untitled Folder/stockslist.txt
    
   # stock_code = []
   # stock_name = []
  #  intents = []
  #  text = []
    #print(vn_sample.read())
    


    
    #file = open("./data/vn_trade.txt","a+")
  #  train_data = []
    def __init__(self):
        self.stock_code = []
        self.stock_name = []
        file_name = './data/stockslist.txt'
        self.read_stock_data(file_name)
        self.help_subject = ["xem","xem cho tôi","cho tôi xem"]
        self.chu_ngu = ["tôi có nhu cầu ","tao muốn","","mình cần","tôi cần","mình muốn","đặt lênh"]
        self.actions = ["mua","bán","chuyển nhượng","sang tên","đầu tư thêm","gom","thêm","mua thêm"]
        self.amounts = ["","khối lượng ","số lượng"]
        self.sub_amounts = ["","cái","cổ phiếu","cổ"]
        self.words = ["tôi muốn","bán","mã","khối lương","giá"]
        self.price_prefix = ["giá","","với giá","tại"]
        self.currency_unit = ["","nghìn đồng","vnđ","nghìn"] 
        self.prefix = ["nhận định","tình hình","thông tin",""]
        self.suffix = ["biến động","lên xuống"]
        self.quesword = ["thế nào","ra sao",""]
        self.infix = ["mã chứng khoán","mã","cổ phiếu","mã cổ phiếu"]
        self.balance_word = ["","còn dư","dư"]
        self.stock_prefix = ["","mã","số"]
        self.conjunction = ["","và"]
        self.advice_prefix = ["có","nên","có nên"]

        self.cash_prefix = ["tài khoản"]
        self.cash_infix = ["đuôi"]
    
        self.check_stopword = DataCleaner()
    def pos_tagging(self,string):
        return ViPosTagger.postagging(ViTokenizer.tokenize(string))
    def make_train_data(self,raw,raw_file=None):
        
        data = []
        sent = []
    #  print(raw[1],"-------\n",raw)
    #  word1 = re.sub('_'," ",word)
        for i in range(len(raw[1])):
            word = raw[0][i]
            pos = raw[1][i]
            if 1:
                
                entity_name = ""
                if word == 'bán':
                    entity_name = 'side-S'
                elif word == 'mua'or word == 'đầu_tư'or word =='gom' :
                    entity_name = 'side-B'
                elif word in self.stock_code :
                    entity_name = "symbol"
                    pos = "Np"
                elif word == "cổ_phiếu" or word == "chứng_khoán" or word == "mã" or word == "mã_chứng_khoán" or word == "mã_cổ_phiếu" or word =="cổ" :
                    #entity_name = "O"
                    
                    entity_name = 'symbol-prefix'
                elif pos == 'M' and word[0].isdigit() :
                    if i > 0 and raw[0][i-1] == "đuôi":
                        entity_name = "0"
                    else:    
                        entity_name = 'quantity'
                        for i in range(len(word)) :
                            if word[i] == "." :
                                entity_name = 'price'
                                r = random.randint(0,1)
                                if r :
                                    word = word[:i]
                                
                                # print("word",word)
                                break                  
                else :
                    entity_name = 'O'
                    
                #word1 = re.sub("_"," ",word)
                try:
                    raw_file.write(word+" "+pos+" "+entity_name+"\n")
                except AttributeError:
                    a = 0
                data.append((word,pos,entity_name))
                sent.append(word)
       # print(data)

        return data,sent
    def read_stock_data(self,file_name):
        stock_file = open(file_name,"r",encoding='utf8')
        for line in stock_file:
            temp = line.split(",")
            self.stock_code.append(temp[0].lower())
            self.stock_name.append(temp[1])
    def write_raw_data(self,raw):
        for line in raw[0]:
            a= 0
    def read_raw_data(self,file_name):
        f = open(file_name,encoding='utf8')
        data  = []
        sent = []
        for line in f :
            
            if line != '\n'  :
                line = line.rstrip()
                temp = line.split(" ")
               # print("temp:",temp)
                sent.append((temp[0],temp[1],temp[2]))
            else:
                data.append(sent)
                sent = []
        
        return data

    def gen_data(self,num_ex):
        raw_file = open("./data/ner_data.csv",'w',encoding='utf8')
        
        
        train_data = [] 
        for i in range(num_ex):
            subject = self.chu_ngu[random.randint(0,len(self.chu_ngu)-1)]
            action = self.actions[random.randint(0,len(self.actions)-1)]
            price = str(round(random.random()*26,2))
            quantity = str(int(random.random()*1000))
            amount = self.amounts[random.randint(0,len(self.amounts)-1)]
            sub_amount = self.sub_amounts[random.randint(0,len(self.sub_amounts)-1)]
            help_subject = self.help_subject[random.randint(0,len(self.help_subject)-1)]
            stock_code_index = int(random.random()*len(self.stock_code))
            account_id = str(random.randint(0,100))
            strings = []
            #trade 
            string1 = subject+" "+action+" "+self.stock_prefix[random.randint(0,len(self.stock_prefix)-1)]+" "+self.stock_code[stock_code_index]+" "+amount+" "+quantity+" "+sub_amount+" "+self.conjunction[random.randint(0,len(self.conjunction)-1)]+" "+self.price_prefix[random.randint(0,len(self.price_prefix)-1)]+" "+price+" "+self.currency_unit[int(random.random()*3)]  
            string2 = subject+" "+action+" "+amount+" "+quantity+" "+sub_amount+" "+self.words[2]+" "+self.stock_code[stock_code_index]+" "+self.price_prefix[random.randint(0,len(self.price_prefix)-1)]+" "+price+" "+self.currency_unit[int(random.random()*3)]
            string3 = subject+" "+action+" "+amount+" "+quantity+" "+sub_amount+" "+self.stock_code[stock_code_index]+" "+self.price_prefix[random.randint(0,len(self.price_prefix)-1)]+" "+price+" "+self.currency_unit[int(random.random()*3)]
            # pattern example : 
            string10 = action+" "+self.stock_prefix[random.randint(0,len(self.stock_prefix)-1)]+" "+self.stock_code[stock_code_index]+" "+self.stock_prefix[random.randint(0,len(self.stock_prefix)-1)]+" "+self.price_prefix[random.randint(0,len(self.price_prefix)-1)]+" "+quantity +" "+ sub_amount
            #market 
            string4 = self.prefix[random.randint(0,len(self.prefix)-1)] +" "+ self.infix[random.randint(0,len(self.infix)-1)]+" "+self.stock_code[stock_code_index] #+" "+self.quesword[1]#self.suffix[random.randint(0,len(self.suffix)-1)]
            #string7 = subject+" "+action+" "+self.stock_code[stock_code_index]+" "
            #market
            string5 = self.stock_code[stock_code_index]+" " +self.suffix[random.randint(0,len(self.suffix)-1)]
            #stock balance and cash balance:
            #vd cho toi xem thong tin( nhan dinh ) ma co phieu ssi con du 
            string6  = help_subject + " "+ self.prefix[random.randint(0,len(self.prefix)-1)]+" " + self.stock_prefix[random.randint(0,len(self.stock_prefix)-1)]+" "+self.stock_code[stock_code_index]+" "+self.balance_word[random.randint(0,len(self.balance_word)-1)]
            #string6 = "môt con vịt"
            #advice
            string7 = self.advice_prefix[random.randint(0,len(self.advice_prefix)-1)] +" "+ action+" " + self.stock_prefix[random.randint(0,len(self.stock_prefix)-1)] +" "+self.stock_code[stock_code_index] + " không?"
            #cash_balance 
            string8 = self.cash_prefix[random.randint(0,len(self.cash_prefix)-1)] + " " + self.cash_infix[random.randint(0,len(self.cash_infix)-1)] + " " + account_id + " còn bao tiền ?"

            string9 = "còn bao nhiêu tiền trong " + self.cash_prefix[random.randint(0,len(self.cash_prefix)-1)] + " " + self.cash_infix[random.randint(0,len(self.cash_infix)-1)] + " " + account_id 
                       
            
            strings.append(string1)
            strings.append(string2)
            strings.append(string3)
            strings.append(string4)
            strings.append(string5)
            strings.append(string6)
            strings.append(string7)
            strings.append(string8)
            strings.append(string9)
            s = random.randint(0,len(strings)-1) 
            
            
            string = strings[s]
            #print("string 1:",string)
           # raw = ViPosTagger.postagging(ViTokenizer.tokenize(string))
            tokens = self.check_stopword.remove_stopword_sent(string)
            new_raw = tokenize_tunning(tokens)
            data = self.make_train_data(new_raw,raw_file)
            raw_file.write("\n")
            train_data.append(data)
            
            ##train_data co dang =[ [(word1,pos1,entity_name),(word2,pos2,entity_name2)],
            #                       [(w1,p1,e1),(w1,p1,e2)]
            #                      ]
       # print(train_data)
        raw_file.close()
        return train_data
    def tokenize_tunning(self,tokens):
        new_tokens = []
        new_pos = []
        for i in range(len(tokens[0])):
            if re.search("_dư",tokens[0][i]):
                sym,word = tokens[0][i].split("_") 
                new_tokens.append(sym)
                new_tokens.append(word)
                new_pos.append("Np")
                new_pos.append("V")
            elif re.search("mã_",token[0][i]):
                word,sym = tokens[0][i].split("_",1)
                if sym == "cổ_phiếu" or sym == "chứng_khoán":
                    new_tokens.append(word) #them chu 'ma' vao 
                    new_tokens.append(word) #them chu 'co_phieu' vao
                    new_pos.append("N")
                    new_pos.append("N")
                else:#sau ma la ma co phieu : ma_ssi
                    new_tokens.append(word) #them chu 'ma' vao 
                    new_tokens.append(word) #them chu 'ssi' vao
                    new_pos.append("N")
                    new_pos.append("Np")
            else:
                new_tokens.append(tokens[0][i])
                new_pos.append(tokens[1][i])
        return (new_tokens,new_pos)
    def make_train_data_from_file(self,file_name):
        text = []
        raw_file = open("./data/ner_test_data.txt","w",encoding='utf8')
        with open(file_name,encoding='utf8') as input:  
            for line in input:
                if line != "\n":
                    temp = line.split(",")[1].lower()
                    token  = self.check_stopword.remove_stopword_sent(temp)
                    new_raw = tokenize_tunning(token)
                    data = self.make_train_data(new_raw,raw_file)
                    raw_file.write("\n")
        raw_file.close()
# k 0= VnGen()
# print(k.gen_data(5))u
if __name__ == "__main__":
    gen = VnGen()
    gen.gen_data(10000)
    gen.make_train_data_from_file("./data/filetester.txt")
    
