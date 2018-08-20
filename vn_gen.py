import random
from pyvi import ViTokenizer,ViPosTagger
import re
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
        self.chu_ngu = ["tôi có nhu cầu ","tao muốn","","mình cần","tôi cần","mình muốn"]
        self.actions = ["mua","bán","chuyển nhượng","sang tên"]
        self.amounts = ["","khối lượng ","số lượng"]
        self.sub_amounts = ["","cái","cổ phiếu"]
        self.words = ["tôi muốn","bán","mã","khối lương","giá"]
        self.currency_unit = ["","nghìn đồng","vnđ","nghìn"] 
        self.prefix = ["nhận định","tình hình","thông tin"]
        self.suffix = ["biến động"]
        self.quesword = ["thế nào","ra sao",""]
        self.infix = ["mã chứng khoán","mã","cổ phiếu","mã cổ phiếu"]
        self.raw = open("./data/ner_data.csv",'w')
    def pos_tagging(self,string):
        return ViPosTagger.postagging(ViTokenizer.tokenize(string))
    def make_train_data(self,raw):
        
        data = []
    #  print(raw[1],"-------\n",raw)
    #  word1 = re.sub('_'," ",word)
        for i in range(len(raw[1])):

            word = raw[0][i]
            pos = raw[1][i]
            entity_name = ""
            if word == 'bán':
                entity_name = 'side-S'
            elif word == 'mua':
                entity_name = 'side-B'
            elif word in self.stock_code :
                entity_name = "symbol"
            elif word == "cổ_phiếu" or word == "chứng_khoán" or word == "mã" or word == "mã_chứng_khoán" or word == "mã_cổ_phiếu":
                entity_name = 'symbol-prefix'
            elif pos == 'M' and word[0].isdigit() :
                
                entity_name = 'quantity'
                for i in range(len(word)) :
                    if word[i] == "." :
                        entity_name = 'price'
                        r = random.randint(0,1)
                        if r :
                            word = word[:i]
                        
                            #print("word",word)
                        break                  
            else :
                entity_name = 'O'
            #word1 = re.sub("_"," ",word)
            self.raw.write(word+" "+pos+" "+entity_name+"\n")
            data.append((word,pos,entity_name))
       # print(data)

        return data
    def read_stock_data(self,file_name):
        stock_file = open(file_name,"r")
        for line in stock_file:
            temp = line.split(",")
            self.stock_code.append(temp[0].lower())
            self.stock_name.append(temp[1])
    def write_raw_data(self,raw):
        for line in raw[0]:
            a= 0
    def read_raw_data(self,file_name):
        f = open(file_name)
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
        file_name = './data/stockslist.txt'
        self.read_stock_data(file_name)
        train_data = [] 
        for i in range(num_ex):
            subject = self.chu_ngu[random.randint(0,len(self.chu_ngu)-1)]
            action = self.actions[int(random.random()*4)]
            price = str(round(random.random()*26,2))
            quantity = str(int(random.random()*1000))
            amount = self.amounts[int(random.random()*3)]
            sub_amount = self.sub_amounts[int(random.random()*3)]
            
            stock_code_index = int(random.random()*len(self.stock_code))
            strings = []
            
            string1 = subject+" "+action+" "+self.words[2]+" "+self.stock_code[stock_code_index]+" "+amount+" "+quantity+" "+sub_amount+" "+self.words[4]+" "+price+" "+self.currency_unit[int(random.random()*3)]  
            string2 = subject+" "+action+" "+amount+" "+quantity+" "+sub_amount+" "+self.words[2]+" "+self.stock_code[stock_code_index]+" "+self.words[4]+" "+price+" "+self.currency_unit[int(random.random()*3)]
            string3 = subject+" "+action+" "+amount+" "+quantity+" "+sub_amount+" "+self.stock_code[stock_code_index]+" "+self.words[4]+" "+price+" "+self.currency_unit[int(random.random()*3)]
            string4 = self.prefix[random.randint(0,len(self.prefix)-1)] +" "+ self.infix[random.randint(0,len(self.infix)-1)]+" "+self.stock_code[stock_code_index] #+" "+self.quesword[1]#self.suffix[random.randint(0,len(self.suffix)-1)]
            string5 = self.stock_code[stock_code_index]+" " +self.suffix[random.randint(0,len(self.suffix)-1)]
            #print('string 4:',string4)
            s = random.randint(0,4)
            
            
            strings.append(string1)
            strings.append(string2)
            strings.append(string3)
            strings.append(string4)
            strings.append(string5)
            string = strings[s]
            #print("string 1:",string)
            raw = ViPosTagger.postagging(ViTokenizer.tokenize(string))
            data = self.make_train_data(raw)
            self.raw.write("\n")
            train_data.append(data)
            
            ##train_data co dang =[ [(word1,pos1,entity_name),(word2,pos2,entity_name2)],
            #                       [(w1,p1,e1),(w1,p1,e2)]
            #                      ]
       # print(train_data)
        return train_data
# k 0= VnGen()
# print(k.gen_data(5))u
if __name__ == "__main__":
    gen = VnGen()
    gen.gen_data(5000)
    gen.raw.close()

