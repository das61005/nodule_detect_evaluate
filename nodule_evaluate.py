from matplotlib import image
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import matplotlib.pylab as plt
from sklearn.utils.multiclass import type_of_target
import os
import random
import numpy as np

ground_truth_path="ground_turth/1.2.222.222.222.416.1310554326.npz"
predict_path="predict/1.2.222.222.222.416.1310554326.npz"
ground_truth=np.load(ground_truth_path)
predict=np.load(predict_path)

image_g=ground_truth["mask"]
image_p=predict["pbm"]


output_folder="result/"

y_scores = image_p.reshape(image_p.shape[0],image_p.shape[1]*image_p.shape[2], 1)
y_true = image_g.reshape(image_g.shape[0],image_g.shape[1]*image_g.shape[2], 1)
y_true = np.where(y_true>0.5, 1, 0)
#Area under the ROC curve
count = 0

confusion_list = []
AUC_ROC_list = []
AUC_PrecRec_list = []
accuracy_list = []
specificity_list = []
sensitivity_list = []
precision_list = []
Dice_list = []
for true,score,lung,mask in zip(y_true, y_scores, image_p, image_g):
    count = count+1
    if count % 10 == 0:
        print(count)

    #Confusion matrix
    threshold_confusion = 0.5
    y_pred = np.empty((score.shape[0]))
    for i in range(score.shape[0]):
        if score[i]>=threshold_confusion:
            y_pred[i]=1
        else:
            y_pred[i]=0
    confusion = confusion_matrix(true, y_pred)
    confusion_list.append(confusion)
    #c(0,0)=TN,c(0,1)=FP,c(1,1)=TP,c(1,0)=FN
    if confusion.shape == (2,2):
        print("TN=",confusion[0,0],",FP=",confusion[0,1],",TP=",confusion[1,1],",FN=",confusion[1,0])
        accuracy = 0
        if float(np.sum(confusion))!=0:
            accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
            accuracy_list.append(accuracy)
        specificity = 0
        if float(confusion[0,0]+confusion[0,1])!=0:
            specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
            specificity_list.append(specificity)
        sensitivity = 0
        if float(confusion[1,1]+confusion[1,0])!=0:
            sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
            sensitivity_list.append(sensitivity)
        precision = 0
        if float(confusion[1,1]+confusion[0,1])!=0:
            precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
            precision_list.append(precision)
        Dice = 0
        if float(confusion[1,1]+confusion[0,1])!=0:
            Dice = float(2*confusion[1,1])/float(confusion[0,1]+confusion[1,0]+confusion[1,1])
            Dice_list.append(Dice)
        #用dice判斷圖形相似度(若有一方是全黑的另一方有圖片會自動跑到<0.5)
        if Dice < 0.5:
            plt.figure()
            plt.imshow(true.reshape(512,512))
            plt.title('true'+str(count)+" Dice="+str(Dice))
            plt.savefig(output_folder+"IMG/"+str(count)+"_true.png")
            plt.close()
            plt.figure()
            plt.imshow(score.reshape(512,512))
            plt.title('score'+str(count)+" Dice="+str(Dice))
            plt.savefig(output_folder+"IMG/"+str(count)+"_score.png")
            plt.close()
        elif Dice > 0.8:
            plt.figure()
            plt.imshow(true.reshape(512,512))
            plt.title('true'+str(count)+" Dice="+str(Dice))
            plt.savefig(output_folder+"IMG_heigh_score/"+str(count)+"_true.png")
            plt.close()
            plt.figure()
            plt.imshow(score.reshape(512,512))
            plt.title('score'+str(count)+" Dice="+str(Dice))
            plt.savefig(output_folder+"IMG_heigh_score/"+str(count)+"_score.png")
            plt.close()
        else:
            plt.figure()
            plt.imshow(true.reshape(512,512))
            plt.title('true'+str(count)+" Dice="+str(Dice))
            plt.savefig(output_folder+"IMG_mid_score/"+str(count)+"_true.png")
            plt.close()
            plt.figure()
            plt.imshow(score.reshape(512,512))
            plt.title('score'+str(count)+" Dice="+str(Dice))
            plt.savefig(output_folder+"IMG_mid_score/"+str(count)+"_score.png")
            plt.close()
    
    if 0 in true:
        if 1 in true:
            if 1 in score:
                if 0 in score:
                    fpr, tpr, thresholds = roc_curve((true), score)
                    AUC_ROC = roc_auc_score(true, score)
                    AUC_ROC_list.append(AUC_ROC)
                    #print ("\nArea under the ROC curve: " +str(AUC_ROC))
                    '''
                    plt.figure()
                    plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
                    plt.title('ROC curve')
                    plt.xlabel("FPR (False Positive Rate)")
                    plt.ylabel("TPR (True Positive Rate)")
                    plt.legend(loc="lower right")
                    plt.savefig(output_folder+"ROC_"+str(count)+".png")
                    plt.close()
                    '''
                    precision, recall, thresholds = precision_recall_curve(true, score)
                    precision = np.fliplr([precision])[0] 
                    recall = np.fliplr([recall])[0]
                    AUC_prec_rec = np.trapz(precision,recall)
                    AUC_PrecRec_list.append(AUC_prec_rec)
    #            else: AUC_ROC_list.append(0)
    #        else: AUC_ROC_list.append(0)
    #    else: AUC_ROC_list.append(0)
    #else: AUC_ROC_list.append(0) #; print("預測值都是 0")
#要把roc curve ,PR curve(重點)全部畫出來
plt.figure()
plt.hist(AUC_ROC_list, bins=50, color='c')
plt.title("AUC")
plt.xlabel("AUC_ROC")
plt.ylabel("Frequency")
plt.savefig(output_folder+"ROC_Hist.png")
plt.close()

plt.figure()
plt.hist(AUC_PrecRec_list, bins=50, color='c')
plt.title("ROC")
plt.title("ROC")
plt.xlabel("AUC_PrecRec")
plt.ylabel("Frequency")
plt.savefig(output_folder+"AUC_PrecRec_Hist.png")
plt.close()

plt.figure()
plt.hist(accuracy_list, bins=50, color='c')
plt.title("accuracy")
plt.xlabel("accuracy")
plt.ylabel("Frequency")
plt.savefig(output_folder+"accuracy_Hist.png")
plt.close()

plt.figure()
plt.hist(specificity_list, bins=50, color='c')
plt.title("specifivity")
plt.xlabel("specificity")
plt.ylabel("Frequency")
plt.savefig(output_folder+"specificity_Hist.png")
plt.close()

plt.figure()
plt.hist(sensitivity_list, bins=50, color='c')
plt.title("sensitivity")
plt.xlabel("sensitivity")
plt.ylabel("Frequency")
plt.savefig(output_folder+"sensitivity_Hist.png")
plt.close()

plt.figure()
plt.hist(precision_list, bins=50, color='c')
plt.title("precision")
plt.xlabel("precision")
plt.ylabel("Frequency")
plt.savefig(output_folder+"precision.png")
plt.close()

plt.figure()
plt.hist(Dice_list, bins=50, color='c')
plt.title("dice")
plt.xlabel("Dice")
plt.ylabel("Frequency")
plt.savefig(output_folder+"Dice_Hist.png")
plt.close()


AUC_ROC_mean = np.mean(np.array(AUC_ROC_list))
AUC_PrecRec_mean =  np.mean(np.array(AUC_PrecRec_list))
accuracy_mean =  np.mean(np.array(accuracy_list))
specificity_mean =  np.mean(np.array(specificity_list))
sensitivity_mean =  np.mean(np.array(sensitivity_list))
precision_mean =  np.mean(np.array(precision_list))
Dice_mean = np.mean(np.array(Dice_list))
#Save the results
file_perf = open(output_folder+'performances_maen_noZero.txt', 'w')
file_perf.write("Area under the ROC curve: "+str(AUC_ROC_mean)
                + "\nArea under Precision-Recall curve: " +str(AUC_PrecRec_mean)
                +"\nACCURACY: " +str(accuracy_mean)
                +"\nSENSITIVITY: " +str(sensitivity_mean)
                +"\nSPECIFICITY: " +str(specificity_mean)
                +"\nPRECISION: " +str(precision_mean)
                +"\nDICE: " +str(Dice_mean)
                )
file_perf.close()




print("start caculate all ROC")
y_scores = image_p.reshape(image_p.shape[0]*image_p.shape[1]*image_p.shape[2], 1)#series拉成一個陣列
y_true = y_true.reshape(y_true.shape[0]*y_true.shape[1],1)#series拉成一個陣列
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
print ("\nArea under the ROC curve: " +str(AUC_ROC))
roc_curve =plt.figure()
plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(output_folder+"ROC.png")

print("start caculate all FROC")
fps = fpr * (y_scores.shape[0] - np.sum(y_true) )/ image_g.shape[0]
roc_curve =plt.figure()
plt.plot(fps,tpr,color='b',lw=2)
plt.title('ROC curve')
plt.xlabel("FPS")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(output_folder+"FROC.png")



