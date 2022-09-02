from matplotlib import image
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import matplotlib.pylab as plt
from sklearn.utils.multiclass import type_of_target
import os
import cv2
import numpy as np
import glob
def simple_evaluate(ground_truth_path,predict_path,output_folder):
    # ground_truth_path="ground_turth/1.2.222.222.222.416.1310554326.npz"
    # predict_path="predict/1.2.222.222.222.416.1310554326.npz"
    ground_truth=np.load(ground_truth_path)
    predict=np.load(predict_path)
    

    image_g=ground_truth["mask"]
    image_p=predict["pbm"]

    image_g=[cv2.resize(i,(image_p.shape[1],image_p.shape[2])) for i in image_g]
    image_g=np.array(image_g)

    # output_folder="result/"
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

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
        if count % 50 == 0:
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
            if float(confusion[1,1]+confusion[0,1])!=0 :
                precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
                precision_list.append(precision)
            Dice = 0
            if float(confusion[0,1]+confusion[1,0]+2*confusion[1,1])!=0:
                Dice = float(2*confusion[1,1])/float(confusion[0,1]+confusion[1,0]+2*confusion[1,1])
                Dice_list.append(Dice)
            #用dice判斷圖形相似度(若有一方是全黑的另一方有圖片會自動跑到<0.5)
            if Dice < 0.5:
                if not os.path.isdir(output_folder+"IMG/"):
                    os.mkdir(output_folder+"IMG/")
                plt.figure()
                plt.imshow(true.reshape(image_p.shape[1],image_p.shape[2]))
                plt.title('true'+str(count)+" Dice="+str(Dice))
                
                plt.savefig(output_folder+"IMG/"+str(count)+"_true.png")
                plt.close()
                plt.figure()
                plt.imshow(score.reshape(image_p.shape[1],image_p.shape[2]))
                plt.title('score'+str(count)+" Dice="+str(Dice))
                plt.savefig(output_folder+"IMG/"+str(count)+"_score.png")
                plt.close()
            elif Dice > 0.8:
                if not os.path.isdir(output_folder+"IMG_heigh_score/"):
                    os.mkdir(output_folder+"IMG_heigh_score/")
                plt.figure()
                plt.imshow(true.reshape(image_p.shape[1],image_p.shape[2]))
                plt.title('true'+str(count)+" Dice="+str(Dice))
                plt.savefig(output_folder+"IMG_heigh_score/"+str(count)+"_true.png")
                plt.close()
                plt.figure()
                plt.imshow(score.reshape(image_p.shape[1],image_p.shape[2]))
                plt.title('score'+str(count)+" Dice="+str(Dice))
                plt.savefig(output_folder+"IMG_heigh_score/"+str(count)+"_score.png")
                plt.close()
            else:
                if not os.path.isdir(output_folder+"IMG_mid_score/"):
                    os.mkdir(output_folder+"IMG_mid_score/")
                plt.figure()
                plt.imshow(true.reshape(image_p.shape[1],image_p.shape[2]))
                plt.title('true'+str(count)+" Dice="+str(Dice))
                plt.savefig(output_folder+"IMG_mid_score/"+str(count)+"_true.png")
                plt.close()
                plt.figure()
                plt.imshow(score.reshape(image_p.shape[1],image_p.shape[2]))
                plt.title('score'+str(count)+" Dice="+str(Dice))
                plt.savefig(output_folder+"IMG_mid_score/"+str(count)+"_score.png")
                plt.close()
        
            if 1-specificity>0 and sensitivity>0:
                fpr, tpr, thresholds = metrics.roc_curve((true), score)
                AUC_ROC = roc_auc_score(true, score)
                AUC_ROC_list.append(AUC_ROC)
            if precision>0 and sensitivity>0:    
                precision, recall, thresholds = metrics.precision_recall_curve(true, score)
                precision = np.fliplr([precision])[0] 
                recall = np.fliplr([recall])[0]
                AUC_prec_rec = np.trapz(precision,recall)
                AUC_PrecRec_list.append(AUC_prec_rec)

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
    

    if AUC_ROC_list==[]:AUC_ROC_list=[0.5]
    if AUC_PrecRec_list==[]:AUC_PrecRec_list=[0.5]
    if accuracy_list==[]:accuracy_list=[0]
    if specificity_list==[]:specificity_list=[0]
    if sensitivity_list==[]:sensitivity_list=[0]
    if precision_list==[]:precision_list=[0]
    if Dice_list==[]:Dice_list=[0]
    #平均數做成txt輸出
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
    fpr, tpr, thresholds = metrics.roc_curve((y_true), y_scores)
    AUC_ROC = metrics.roc_auc_score(y_true, y_scores)
    print ("\nArea under the ROC curve: " +str(AUC_ROC))
    # roc_curve =plt.figure()
    plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(output_folder+"ROC.png")
    plt.close()

    print("start caculate all FROC")
    fps = fpr * (y_scores.shape[0] - np.sum(y_true) )/ image_g.shape[0]
    # roc_curve =plt.figure()
    plt.plot(fps,tpr,color='b',lw=2)
    plt.title('ROC curve')
    plt.xlabel("FPS")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(output_folder+"FROC.png")
    plt.close()

    print("start caculate all P-R")
    precision, recall, thresholds = precision_recall_curve((y_true), y_scores)
    AUC_pr = metrics.auc(recall,precision)
    plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_pr)
    plt.plot(recall,precision,color='c')
    plt.title('Precision-Recall curve')
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.legend(loc="lower right")
    plt.savefig(output_folder+"PR.png")
    plt.close()

    return AUC_ROC_mean,AUC_PrecRec_mean,accuracy_mean,specificity_mean,sensitivity_mean,precision_mean,Dice_mean



def average_to_txt(AUC_ROC_sum ,
    AUC_PrecRec_sum ,
    accuracy_sum ,
    specificity_sum ,
    sensitivity_sum ,
    precision_sum ,
    Dice_sum ,output_folder):

    AUC_ROC_mean = np.mean(np.array(AUC_ROC_sum))
    AUC_PrecRec_mean =  np.mean(np.array(AUC_PrecRec_sum))
    accuracy_mean =  np.mean(np.array(accuracy_sum))
    specificity_mean =  np.mean(np.array(specificity_sum))
    sensitivity_mean =  np.mean(np.array(sensitivity_sum))
    precision_mean =  np.mean(np.array(precision_sum))
    Dice_mean = np.mean(np.array(Dice_sum))

    file_perf = open(output_folder, 'w')
    file_perf.write("**Total average**"
                    +"\nArea under the ROC curve: "+str(AUC_ROC_mean)
                    + "\nArea under Precision-Recall curve: " +str(AUC_PrecRec_mean)
                    +"\nACCURACY: " +str(accuracy_mean)
                    +"\nSENSITIVITY: " +str(sensitivity_mean)
                    +"\nSPECIFICITY: " +str(specificity_mean)
                    +"\nPRECISION: " +str(precision_mean)
                    +"\nDICE: " +str(Dice_mean)
                    )
    file_perf.close()

def resample_vs_no_resample(no_resample_sum,resample_sum,output_folder):
    
    file_perf = open(output_folder, 'w')

    for i in range(len(no_resample_sum[0])):
        #gap
        roc_gap=resample_sum[0][i]-no_resample_sum[0][i]
        pr_gap=resample_sum[1][i]-no_resample_sum[1][i]
        accuracy_gap=resample_sum[2][i]-no_resample_sum[2][i]
        specificity_gap=resample_sum[3][i]-no_resample_sum[3][i]
        sensitivity_gap=resample_sum[4][i]-no_resample_sum[4][i]
        precision_gap=resample_sum[5][i]-no_resample_sum[5][i]
        Dice_gap=resample_sum[6][i]-no_resample_sum[6][i]
        #groth rate
        roc_gr=resample_sum[0][i]/no_resample_sum[0][i]*100-100
        pr_gr=resample_sum[1][i]/no_resample_sum[1][i]*100-100
        accuracy_gr=resample_sum[2][i]/no_resample_sum[2][i]*100-100
        specificity_gr=resample_sum[3][i]/no_resample_sum[3][i]*100-100
        sensitivity_gr=resample_sum[4][i]/no_resample_sum[4][i]*100-100
        precision_gr=resample_sum[5][i]/no_resample_sum[5][i]*100-100
        Dice_gr=resample_sum[6][i]/no_resample_sum[6][i]*100-100

        file_perf.write("**difference with resample data and no resample data form"
                    +no_resample_sum[7][i]+"**"
                    +"\nArea under the ROC curve:\nno resample:"+str(no_resample_sum[0][i])+",resample:"+str(resample_sum[0][i])
                    +"\ngap:"+str(roc_gap)+",Growth Rates:"+str(roc_gr)+"%"
                    +"\n\nArea under Precision-Recall curve:\nno resample:"+str(no_resample_sum[1][i])+",resample:"+str(resample_sum[1][i])
                    +"\ngap:"+str(pr_gap)+",Growth Rates:"+str(pr_gr)+"%"
                    +"\n\nACCURACY:\nno resample:"+str(no_resample_sum[2][i])+",resample:"+str(resample_sum[2][i])
                    +"\ngap:"+str(accuracy_gap)+",Growth Rates:"+str(accuracy_gr)+"%"
                    +"\n\nSENSITIVITY:\nno resample:"+str(no_resample_sum[3][i])+",resample:"+str(resample_sum[3][i])
                    +"\ngap:"+str(specificity_gap)+",Growth Rates:"+str(specificity_gr)+"%"
                    +"\n\nSPECIFICITY:\nno resample:"+str(no_resample_sum[4][i])+",resample:"+str(resample_sum[4][i])
                    +"\ngap:"+str(sensitivity_gap)+",Growth Rates:"+str(sensitivity_gr)+"%"
                    +"\n\nPRECISION:\nno resample:"+str(no_resample_sum[5][i])+",resample:"+str(resample_sum[5][i])
                    +"\ngap:"+str(precision_gap)+",Growth Rates:"+str(precision_gr)+"%"
                    +"\n\nDICE:\nno resample:"+str(no_resample_sum[6][i])+",resample:"+str(resample_sum[6][i])
                    +"\ngap:"+str(Dice_gap)+",Growth Rates:"+str(Dice_gr)+"%"
                    +"\n******************************************************\n\n\n"
                    )
    file_perf.close()

#main
path_resample_g="nodule_evaluate/batch_9_training_set/"
path_resample_p="nodule_evaluate/unet3d_81_final_1_batch_9_TTA_True/"

path_no_resample_g="nodule_evaluate/batch_9_no_resample/"
path_no_resample_p="nodule_evaluate/unet3d_81_final_1_batch_9_no_resample_TTA_True/"

resample_g_npz=glob.glob(path_resample_g+"*.npz")
resample_p_npz=glob.glob(path_resample_p+"*.npz")
no_resample_g_npz=glob.glob(path_no_resample_g+"*.npz")
no_resample_p_npz=glob.glob(path_no_resample_p+"*.npz")

indexes=[i.replace(path_resample_g,'').replace(".npz",'') for i in resample_g_npz]
print(indexes[0])
if not os.path.isdir("result"):
    os.mkdir("result")

no_resample_sum=[[],[],[],[],[],[],[],[]]

resample_sum=[[],[],[],[],[],[],[],[]]

#no resample
for i,index in enumerate(indexes):
    print("no_resample:no.",i)
    # simple_evaluate(no_resample_g_npz[i],no_resample_p_npz[i],"result/"+index + "_  no_resample/")
    AUC_ROC_mean,AUC_PrecRec_mean,accuracy_mean,specificity_mean,sensitivity_mean,precision_mean,Dice_mean=simple_evaluate(no_resample_g_npz[i],no_resample_p_npz[i],"result/" + index + "_no_resample/")
    no_resample_sum[0].append(AUC_ROC_mean)
    no_resample_sum[1].append(AUC_PrecRec_mean)
    no_resample_sum[2].append(accuracy_mean)
    no_resample_sum[3].append(specificity_mean)
    no_resample_sum[4].append(sensitivity_mean)
    no_resample_sum[5].append(precision_mean)
    no_resample_sum[6].append(Dice_mean)
    no_resample_sum[7].append(index)

average_to_txt(no_resample_sum[0],no_resample_sum[1],no_resample_sum[2],no_resample_sum[3],no_resample_sum[4],no_resample_sum[5],no_resample_sum[6]
,"result/no_reample_average_performance.txt")


#resample
for i,index in enumerate(indexes):
    print("resample:no.",i)
    # simple_evaluate(resample_g_npz[i],resample_p_npz[i],"result/"+index + "resample")
    AUC_ROC_mean,AUC_PrecRec_mean,accuracy_mean,specificity_mean,sensitivity_mean,precision_mean,Dice_mean=simple_evaluate(resample_g_npz[i],resample_p_npz[i],"result/" + index + "_resample/")
    resample_sum[0].append(AUC_ROC_mean)
    resample_sum[1].append(AUC_PrecRec_mean)
    resample_sum[2].append(accuracy_mean)
    resample_sum[3].append(specificity_mean)
    resample_sum[4].append(sensitivity_mean)
    resample_sum[5].append(precision_mean)
    resample_sum[6].append(Dice_mean)
    resample_sum[7].append(index)

average_to_txt(resample_sum[0],resample_sum[1],resample_sum[2],resample_sum[3],resample_sum[4],resample_sum[5],resample_sum[6]
,"result/reample_average_performance.txt")

resample_vs_no_resample(no_resample_sum,resample_sum,"result/difference_with_resample_and_no_resample_data.txt")

    



    



    
