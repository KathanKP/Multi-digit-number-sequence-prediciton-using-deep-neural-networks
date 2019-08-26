from scipy import ndimage,misc
import pandas as pd 
import numpy as np 

data=pd.read_csv('train.csv')
data=data.values


# arr=ndimage.imread('1.png',mode="RGB")
# temp=arr[77:77+223,246:246+173,:]
# temp_resized=misc.imresize(temp,(64,64))
# misc.imsave('resize.png',temp_resized)

#print(data[1,:])

main_list=[]
final_dataset=[]
lenn=data.shape[0]
counter=0
for i in range(1,33403):
	left=data[counter,2]
	top=data[counter,3]
	right=data[counter,2]+data[counter,4]
	bottom=data[counter,3]+data[counter,5]
	# print 'right is ',right,'bottom is ',bottom
	no_digits=1
	labels=[]

	labels.append(data[counter,1])
	while(counter+1<lenn  and data[counter+1,0]==data[counter,0]):
		#print 'counter is ',counter
		counter+=1
		left=min(left,data[counter,2])
		top=min(top,data[counter,3])
		right=max(right,data[counter,2]+data[counter,4])
		bottom=max(bottom,data[counter,3]+data[counter,5])
		labels.append(data[counter,1])
		#print 'Label',labels[no_digits-1]
		no_digits+=1
	#print 'counter is ',counter
	#print labels
	counter+=1
	len_labels=len(labels)
	while(len_labels<5):
		labels.append(10)
		len_labels+=1
	if(len_labels>5):
		labels=labels[0:5]
	width=right-left
	height=bottom-top
	if(top<0):
		top=0
	if(left<0):
		left=0
	if(width<0):
		width=0
	if(height<0):
		height=0
	# if(no_digits>5):
	# 	continue
	# if(no_digits>5):
	# 	print i
	main_list.append((left,top,width,height,no_digits,labels))
	#print labels
	temp=np.asarray(labels)
	#print temp
	answer=np.hstack((np.array([left,top,width,height,no_digits]),temp))
	#print answer.shape
	if(i==1):
	  	final_dataset=answer
	else:
		#print 'Final:',final_dataset.shape,' Answer:',answer.shape
	  	final_dataset=np.vstack((final_dataset,answer))
print final_dataset.shape
print len(main_list)
np.savetxt("final_dataset.csv",final_dataset,delimiter=",")