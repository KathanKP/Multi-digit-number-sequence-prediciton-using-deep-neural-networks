import numpy as np 
import pandas as pd 
import tensorflow as tf 
from scipy import ndimage,misc
from random import shuffle
import pickle
import _pickle as cPickle
import gzip
from skimage.color import rgb2grey

images=[]
no_digits=[]
first=[]
second=[]
third=[]
fourth=[]
fifth=[]

def calculate_data():
    
    data=pd.read_csv('final_dataset.csv',header=None)
    data=data.values

    for i in range(1,33403):
        if i!=29930:
            s="/home/f2013112/cyclops/data/train_images/"+str(i)+".png"
            arr=ndimage.imread(s,mode="RGB")
            arr=rgb2grey(arr)
            temp=arr[int(data[i-1,1]):int(data[i-1,1]+data[i-1,3]),int(data[i-1,0]):int(data[i-1,0]+data[i-1,2])]
            temp_resized=misc.imresize(temp,(80,80)) 
            flattened=np.resize(temp_resized,(1,80*80))
            digits=np.zeros([1,5])
            digits[0,int(data[i-1,4])-1]=1
            first1=np.zeros([1,11])
            first1[0,int(data[i-1,5])]=1
            second1=np.zeros([1,11])
            second1[0,int(data[i-1,6])]=1
            third1=np.zeros([1,11])
            third1[0,int(data[i-1,7])]=1
            fourth1=np.zeros([1,11])
            fourth1[0,int(data[i-1,8])]=1
            fifth1=np.zeros([1,11])
            fifth1[0,int(data[i-1,9])]=1
            if i==1:
                images.append(temp_resized)
                # images=temp_resized
                no_digits=digits
                first=first1
                second=second1
                third=third1
                fourth=fourth1
                fifth=fifth1
            else:
                # images=np.vstack((images,flattened))
                images.append(temp_resized)
                no_digits=np.vstack((no_digits,digits))
                first=np.vstack((first,first1))
                second=np.vstack((second,second1))
                third=np.vstack((third,third1))
                fourth=np.vstack((fourth,fourth1))
                fifth=np.vstack((fifth,fifth1))
            #print images[i-1].shape
            # print no_digits.shape
            # print first.shape
            # print first
            # print second
            # print third 
            # print fourth
            if i%1000==0:
                print (i)

    tp=list(zip(images,no_digits,first,second,third,fourth,fifth))
    cPickle.dump(tp,open('saved_greyscale_100x100.p','wb'))





def weights_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")



#Layer1 Weights
W_conv1=weights_variable([3,3,1,32])
b_conv1=bias_variable([32])
#Matrix size after this is 64*64*32

#Layer2 Weights
W_conv2=weights_variable([3,3,32,64])
b_conv2=bias_variable([64])
#Matrix size after pooling done here is 32*32*64

#Layer3 Weights
W_conv3=weights_variable([3,3,64,128])
b_conv3=bias_variable([128])
#Matrix size after this is 32*32*128

#Layer4 Weights
W_conv4=weights_variable([3,3,128,256])
b_conv4=bias_variable([256])

#Layer4 Weights
W_conv5=weights_variable([3,3,256,512])
b_conv5=bias_variable([512])
#Matrix size after pooling here is 8*8*512
#CNN ends here,we get image features that go into 7 different softmax classifiers

W_single_layer=weights_variable([8*8*512,1024])
b_single_layer=bias_variable([1024])

#Classifier1 to predict how many digits are there in the image
W_c1_fc1=weights_variable([1024,1024])
b_c1_fc1=bias_variable([1024])
W_c1_fc2=weights_variable([1024,1024])
b_c1_fc2=bias_variable([1024])
W_c1_fc3=weights_variable([1024,5])
b_c1_fc3=bias_variable([5])

#Classifier2 to predict first digit
W_c2_fc1=weights_variable([1024,1024])
b_c2_fc1=bias_variable([1024])
W_c2_fc2=weights_variable([1024,1024])
b_c2_fc2=bias_variable([1024])
W_c2_fc3=weights_variable([1024,11])
b_c2_fc3=bias_variable([11])

#Classifier3 to predict second digit
W_c3_fc1=weights_variable([1024,1024])
b_c3_fc1=bias_variable([1024])
W_c3_fc2=weights_variable([1024,1024])
b_c3_fc2=bias_variable([1024])
W_c3_fc3=weights_variable([1024,11])
b_c3_fc3=bias_variable([11])

#Classifier4 to predict third digit
W_c4_fc1=weights_variable([1024,1024])
b_c4_fc1=bias_variable([1024])
W_c4_fc2=weights_variable([1024,1024])
b_c4_fc2=bias_variable([1024])
W_c4_fc3=weights_variable([1024,11])
b_c4_fc3=bias_variable([11])

#Classifier5 to predict fourth digit
W_c5_fc1=weights_variable([1024,1024])
b_c5_fc1=bias_variable([1024])
W_c5_fc2=weights_variable([1024,1024])
b_c5_fc2=bias_variable([1024])
W_c5_fc3=weights_variable([1024,11])
b_c5_fc3=bias_variable([11])

#Classifier6 to predict fifth digit
W_c6_fc1=weights_variable([1024,1024])
b_c6_fc1=bias_variable([1024])
W_c6_fc2=weights_variable([1024,1024])
b_c6_fc2=bias_variable([1024])
W_c6_fc3=weights_variable([1024,11])
b_c6_fc3=bias_variable([11])
#Inputs and Outputs
x=tf.placeholder(tf.float32,shape=[None,64,64,1],name="input")
y_1=tf.placeholder(tf.float32,shape=[None,5])
y_2=tf.placeholder(tf.float32,shape=[None,11])
y_3=tf.placeholder(tf.float32,shape=[None,11])
y_4=tf.placeholder(tf.float32,shape=[None,11])
y_5=tf.placeholder(tf.float32,shape=[None,11])
y_6=tf.placeholder(tf.float32,shape=[None,11]) 
keep_prob=tf.placeholder(tf.float32)
lr=tf.placeholder(tf.float32)

h_conv1=conv2d(x,W_conv1)+b_conv1  #Output of first conv + relu layer
batch_mean1, batch_var1 = tf.nn.moments(h_conv1,[0,1,2],keep_dims=False)
scale1 = tf.Variable(tf.ones([32]))
beta1 = tf.Variable(tf.zeros([32]))
BN1 = tf.nn.batch_normalization(h_conv1,batch_mean1,batch_var1,beta1,scale1,1e-3)
h_conv1=tf.nn.relu(BN1)

h_conv2=tf.nn.relu(conv2d(h_conv1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)                   #Output after second layer of conv+relu+pool

h_conv3=(conv2d(h_pool2,W_conv3)+b_conv3) #Output of third layer conv+relu
batch_mean2, batch_var2 = tf.nn.moments(h_conv3,[0,1,2],keep_dims=False)
scale2 = tf.Variable(tf.ones([128]))
beta2= tf.Variable(tf.zeros([128]))
BN2 = tf.nn.batch_normalization(h_conv3,batch_mean2,batch_var2,beta2,scale2,1e-3)
h_conv3=tf.nn.relu(BN2)

h_conv4=tf.nn.relu(conv2d(h_conv3,W_conv4)+b_conv4)
h_pool4=max_pool_2x2(h_conv4)                   #Output of fourth layer conv+relu+pool

h_conv5=max_pool_2x2(tf.nn.relu(conv2d(h_pool4,W_conv5)+b_conv5))

flattened_inputs=tf.reshape(h_conv5,[-1,8*8*512])

flattened_inputs=tf.nn.relu(tf.matmul(flattened_inputs,W_single_layer)+b_single_layer)
#First Classifier
h_c1_fc1=tf.nn.relu(tf.matmul(flattened_inputs,W_c1_fc1)+b_c1_fc1)
h_c1_fc2=tf.nn.relu(tf.matmul(h_c1_fc1,W_c1_fc2)+b_c1_fc2)
h_c1_drop = tf.nn.dropout(h_c1_fc2, keep_prob)
y_conv1=tf.add(tf.matmul(h_c1_drop,W_c1_fc3),b_c1_fc3,name="output1");

#Second Classifier
h_c2_fc1=tf.nn.relu(tf.matmul(flattened_inputs,W_c2_fc1)+b_c2_fc1)
h_c2_fc2=tf.nn.relu(tf.matmul(h_c2_fc1,W_c2_fc2)+b_c2_fc2)
h_c2_drop = tf.nn.dropout(h_c2_fc2, keep_prob)
y_conv2=tf.add(tf.matmul(h_c2_drop,W_c2_fc3),b_c2_fc3,name="output2");

#Third Classifier
h_c3_fc1=tf.nn.relu(tf.matmul(flattened_inputs,W_c3_fc1)+b_c3_fc1)
h_c3_fc2=tf.nn.relu(tf.matmul(h_c3_fc1,W_c3_fc2)+b_c3_fc2)
h_c3_drop = tf.nn.dropout(h_c3_fc2, keep_prob)
y_conv3=tf.add(tf.matmul(h_c3_drop,W_c3_fc3),b_c3_fc3,name="output3");

#Fourth Classifier
h_c4_fc1=tf.nn.relu(tf.matmul(flattened_inputs,W_c4_fc1)+b_c4_fc1)
h_c4_fc2=tf.nn.relu(tf.matmul(h_c4_fc1,W_c4_fc2)+b_c4_fc2)
h_c4_drop = tf.nn.dropout(h_c4_fc2, keep_prob)
y_conv4=tf.add(tf.matmul(h_c4_drop,W_c4_fc3),b_c4_fc3,name="output4");

#Fifth Classifier
h_c5_fc1=tf.nn.relu(tf.matmul(flattened_inputs,W_c5_fc1)+b_c5_fc1)
h_c5_fc2=tf.nn.relu(tf.matmul(h_c5_fc1,W_c5_fc2)+b_c5_fc2)
h_c5_drop = tf.nn.dropout(h_c5_fc2, keep_prob)
y_conv5=tf.add(tf.matmul(h_c5_drop,W_c5_fc3),b_c5_fc3,name="output5");

#Sixth Classifier
h_c6_fc1=tf.nn.relu(tf.matmul(flattened_inputs,W_c6_fc1)+b_c6_fc1)
h_c6_fc2=tf.nn.relu(tf.matmul(h_c6_fc1,W_c6_fc2)+b_c6_fc2)
h_c6_drop = tf.nn.dropout(h_c6_fc2, keep_prob)
y_conv6=tf.add(tf.matmul(h_c6_drop,W_c6_fc3),b_c6_fc3,name="output6");

sess=tf.InteractiveSession()

data_tuple=cPickle.load(open('saved_greyscale.p','rb')) 
shuffle(data_tuple)
train_break=int(len(data_tuple)*0.8)
#val_break=int(len(self.data_tuple)*0.8)
train_tuple=data_tuple[:train_break]
val_tuple=data_tuple[train_break:]

#First softmax
cross_entropy1=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_1,logits=y_conv1))

#Second softmax
cross_entropy2=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_2,logits=y_conv2))

#Third softmax
cross_entropy3=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_3,logits=y_conv3))

#Fourth softmax
cross_entropy4=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_4,logits=y_conv4))

#Fifth softmax
cross_entropy5=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_5,logits=y_conv5))

#Sixth softmax
cross_entropy6=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_6,logits=y_conv6))

total_entropy=cross_entropy1+cross_entropy2+cross_entropy3+cross_entropy4+cross_entropy5+cross_entropy6

train_step=tf.train.AdamOptimizer(lr).minimize(total_entropy)

g1=tf.argmax(y_conv1,1)
g2=tf.argmax(y_conv2,1)
g3=tf.argmax(y_conv3,1)
g4=tf.argmax(y_conv4,1)
g5=tf.argmax(y_conv5,1)
g6=tf.argmax(y_conv6,1)



sess.run(tf.global_variables_initializer())


def trainn():
    br=int(len(data_tuple)*0.8)
    for i in range(1,13):
        shuffle(train_tuple)
        for j in range(0,br,50):
            if(j+50>len(train_tuple)):
                endd=len(train_tuple)
            else:
                endd=j+50
            current_batch=train_tuple[j:endd]
            # batch_X=[np.expand_dims(k[0],2) for k in current_batch]
            batch_X=[np.expand_dims(k[0],2) for k in current_batch]
            batch_y1=[k[1] for k in current_batch]
            batch_y2=[k[2] for k in current_batch]
            batch_y3=[k[3] for k in current_batch]
            batch_y4=[k[4] for k in current_batch]
            batch_y5=[k[5] for k in current_batch]
            batch_y6=[k[6] for k in current_batch]

            #print (batch_X[0].shape)
            if(i<7):
                _,loss_val=sess.run([train_step,total_entropy],feed_dict={x:batch_X,y_1:batch_y1,y_2:batch_y2,y_3:batch_y3,y_4:batch_y4,y_5:batch_y5,y_6:batch_y6,keep_prob:0.5,lr:1e-4})

            elif (i<10):
                _,loss_val=sess.run([train_step,total_entropy],feed_dict={x:batch_X,y_1:batch_y1,y_2:batch_y2,y_3:batch_y3,y_4:batch_y4,y_5:batch_y5,y_6:batch_y6,keep_prob:0.5,lr:1e-5})
            
            else:
                _,loss_val=sess.run([train_step,total_entropy],feed_dict={x:batch_X,y_1:batch_y1,y_2:batch_y2,y_3:batch_y3,y_4:batch_y4,y_5:batch_y5,y_6:batch_y6,keep_prob:0.5,lr:1e-6})
            #if j%2000==0:
            print (str(i) + ' Step: ',j,' Loss_val:',loss_val)

   

    prediction=[]
    for i in range(0,len(val_tuple),100):
        if(i+100>len(val_tuple)):
            endd=len(val_tuple)
        else:
            endd=i+100
        # eval_X=[np.expand_dims(k[0],2) for k in self.val_tuple[i:endd]]
        eval_X=[np.expand_dims(k[0],2) for k in val_tuple[i:endd]]
        eval_y1=[k[1] for k in val_tuple[i:endd]]
        eval_y2=[k[2] for k in val_tuple[i:endd]]
        eval_y3=[k[3] for k in val_tuple[i:endd]]
        eval_y4=[k[4] for k in val_tuple[i:endd]]
        eval_y5=[k[5] for k in val_tuple[i:endd]]
        eval_y6=[k[6] for k in val_tuple[i:endd]]

        o1,o2,o3,o4,o5,o6=sess.run([g1,g2,g3,g4,g5,g6],feed_dict={x:eval_X,keep_prob:1.0})
        #o1=sess.run([tf.argmax(y_conv1,1)],feed_dict={self.x:eval_X})
        # print len(o1)
        # print o1
        # print 
        digit_pred=o1
        first_pred=o2
        second_pred=o3
        third_pred=o4
        fourth_pred=o5
        fifth_pred=o6

        correct_prediction1=np.equal(np.argmax(eval_y1,1),digit_pred)
        correct_prediction2=np.equal(np.argmax(eval_y2,1),first_pred)
        correct_prediction3=np.equal(np.argmax(eval_y3,1),second_pred)
        correct_prediction4=np.equal(np.argmax(eval_y4,1),third_pred)
        correct_prediction5=np.equal(np.argmax(eval_y5,1),fourth_pred)
        correct_prediction6=np.equal(np.argmax(eval_y5,1),fifth_pred)

        #print 'softmanx 1:',np.mean(correct_prediction1.astype(np.float32))

        eval_len=correct_prediction1.shape[0]
        for i in range(eval_len):
            if correct_prediction1[i]:
                if digit_pred[i]+1==5:
                    temp=np.logical_and(np.logical_and(np.logical_and(np.logical_and(correct_prediction2[i],correct_prediction3[i]),correct_prediction4[i]),
                            correct_prediction5[i]),correct_prediction6[i])
                    prediction.append(temp)
                elif digit_pred[i]+1==4:
                        temp=np.logical_and(np.logical_and(np.logical_and(correct_prediction2[i],correct_prediction3[i]),correct_prediction4[i]),
                            correct_prediction5[i])
                        prediction.append(temp)
                elif digit_pred[i]+1==3:
                        temp=np.logical_and(np.logical_and(correct_prediction2[i],correct_prediction3[i]),correct_prediction4[i])
                        prediction.append(temp)
                elif digit_pred[i]+1==2:
                        temp=np.logical_and(correct_prediction2[i],correct_prediction3[i])
                        prediction.append(temp)
                elif digit_pred[i]+1==1:
                        temp=correct_prediction2[i]
                        prediction.append(temp)

    # print prediction.astype(np.float32)
    pred1=np.array(prediction)
    pred1=pred1.astype(np.float32)
    correct_pred1=np.sum(pred1)
    accuracy=correct_pred1/len(val_tuple)
    print ("Validation accuracy is:",accuracy,'Shape',pred1.shape)


def dumpp():
    saver=tf.train.Saver()
    save_path=saver.save(sess,"letsago.ckpt")

def loadd():
    new_saver=tf.train.Saver()
    new_saver.restore(sess,'./letsago.ckpt')

def predict(test_x):
    o1,o2,o3,o4,o5,o6=sess.run([g1,g2,g3,g4,g5,g6],feed_dict={x:test_x,keep_prob:1.0})
    return o1,o2,o3,o4,o5,o6

class SVHN():

    path = ""
    
    def __init__(self, data_dir):
        self.path = data_dir
        
    def train(self):
        trainn()
        

        # saver=tf.train.Saver()
        # save_path=saver.save(sess,"skeleton1.ckpt")
        # print("Model saved to %s" % save_path)
    def get_sequence(self, image):
        """
            image : a variable resolution RGB image in the form of a numpy array

            return: list of integers with the sequence of digits. Example: [5,0,3] for an image having 503 as the sequence.

        """
        test_x=[]
        image=rgb2grey(image)
        resized=misc.imresize(image,(64,64))
        resized=np.expand_dims(resized,2)
        test_x.append(resized)
        o1,o2,o3,o4,o5,o6= predict(test_x)

        full_prediction=[]
        # print (o1)
        if(o2[0]==10):
            full_prediction.append(0)
        else:
            full_prediction.append(o2[0])
        if(o3[0]==10):
            full_prediction.append(0)
        else:
            full_prediction.append(o3[0])
        if(o4[0]==10):
            full_prediction.append(0)
        else:
            full_prediction.append(o4[0])
        if(o5[0]==10):
            full_prediction.append(0)
        else:
            full_prediction.append(o5[0])
        if(o6[0]==10):
            full_prediction.append(0)
        else:
            full_prediction.append(o6[0])
        print (full_prediction[0:o1[0]+1]) 
        return (full_prediction[0:o1[0]+1])  



    def save_model(self, **params):
        dumpp()


    @staticmethod
    def load_model(**params):
        loadd()
        return SVHN('dataset')
        
        

        """
            returns a pre-trained instance of SVHN class
        """

if __name__ == "__main__":
    # x=1
    # calculate_data()
    obj=SVHN('dataset')
    # obj.train()
    # obj.save_model()
    a=obj.load_model()
    test_image=ndimage.imread("/home/f2013112/chirag/edited_images_greyscale/gs15715.png",mode="RGB")
    a.get_sequence(test_image)