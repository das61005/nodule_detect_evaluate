
from matplotlib import image
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from scipy.ndimage import label
import scipy.ndimage as ndi
import matplotlib.pylab as plt
import os
import numpy as np
import glob
import cv2

def simple_evaluate(ground_truth_path,predict_path,output_folder):
    # ground_truth_path="ground_turth/1.2.222.222.222.416.1310554326.npz"
    # predict_path="predict/1.2.222.222.222.416.1310554326.npz"
    TP_num=0
    ground_truth=np.load(ground_truth_path)
    predict=np.load(predict_path)

    image_g=ground_truth["mask"]
    image_p=predict["pbm"]
    image_g=[cv2.resize(i,(image_p.shape[1],image_p.shape[2])) for i in image_g]
    image_g=np.array(image_g)
    #對每個腫瘤做整理
    nodule=np.unique(image_g)
    #nodule_size=腫瘤的總切片面積總和
    nodule_size=[]
    
    for n in nodule:
        if n==0:
            nodule_size.append(0)
            continue
        else:
            nodule_size.append(np.count_nonzero(image_g==n))
    print("ground_turth_nodule_amount:",len(nodule_size)-1)
    #合格的標準
    match_size=[]
    threshold=1/5
    for i in nodule_size:
        match_size.append(round(i*threshold))

    y_scores=np.where(image_p>0.5,1,0)
    y_scores=np.array(y_scores)
    y_scores,num_features =label(y_scores)
    print("predict_nodule_amount:",num_features-1)
    y_scores=np.array(y_scores)
    FP_num=len(np.unique(y_scores))-1
    # y_scores = y_scores.reshape(image_p.shape[0],image_p.shape[1]*image_p.shape[2])
    # y_true = image_g.reshape(image_g.shape[0],image_g.shape[1]*image_g.shape[2])

    nodule_finded={}
    for n in nodule:
        nodule_finded[n]=0
    #腫瘤中心標記
    nodule_center_2d=[]
    for n in nodule:
        nodule_center_2d.append([])
    predict_center_2d=[]
    for n in np.unique(y_scores):
        predict_center_2d.append([])

    for z,label_gt in enumerate(image_g):
        if len(np.unique(label_gt))>1:
            for n in np.unique(label_gt):
                if n==0:
                    continue
                y, x = ndi.center_of_mass(label_gt==n)
                nodule_center_2d[n].append([x,y,z])

    for z,label_pd in enumerate(y_scores):
        if len(np.unique(label_pd))>1:
            for n in np.unique(label_pd):
                if n==0:
                    continue
                y, x = ndi.center_of_mass(label_pd==n)
                predict_center_2d[n].append([x,y,z])
    

    nodule_center_3d=[]
    for n in nodule:
        if n==0:
            nodule_center_3d.append(0)
            continue
        x_sum=np.asarray(nodule_center_2d[n])[:,0]
        x_mean=round(np.mean(x_sum))
        y_sum=np.asarray(nodule_center_2d[n])[:,1]
        y_mean=round(np.mean(y_sum))
        z_sum=np.asarray(nodule_center_2d[n])[:,2]
        z_mean=round(np.mean(z_sum))
        nodule_center_3d.append([x_mean,y_mean,z_mean])
    predict_center_3d=[]
    for n in np.unique(y_scores):
        if n==0:
            predict_center_3d.append(0)
            continue
        x_sum=np.asarray(predict_center_2d[n])[:,0]
        x_mean=round(np.mean(x_sum))
        y_sum=np.asarray(predict_center_2d[n])[:,1]
        y_mean=round(np.mean(y_sum))
        z_sum=np.asarray(predict_center_2d[n])[:,2]
        z_mean=round(np.mean(z_sum))
        predict_center_3d.append([x_mean,y_mean,z_mean])
    print(nodule_center_3d)
    print(predict_center_3d)
    #計算歐機里德距離
    # print(np.unique(y_scores))
    for i_g,ground_truth in enumerate(nodule_center_3d):
        if i_g==0:continue
        great_distance=10000
        index=-1
        for i_p,predict in enumerate(predict_center_3d):
            if i_p==0:continue
            if great_distance>((predict[0]-ground_truth[0])**2+(predict[1]-ground_truth[1])**2+(predict[2]-ground_truth[2])**2)*0.5:
                index=i_p
                great_distance=((predict[0]-ground_truth[0])**2+(predict[1]-ground_truth[1])**2+(predict[2]-ground_truth[2])**2)*0.5
            
        if not index==-1:
            y_scores=np.where(y_scores==index,0-i_g,y_scores)
    y_scores=np.where(y_scores>0,-999,y_scores)
    # print(np.unique(y_scores))
        

    y_scores = y_scores.reshape(image_p.shape[0],image_p.shape[1]*image_p.shape[2])
    y_true = image_g.reshape(image_g.shape[0],image_g.shape[1]*image_g.shape[2])
    #比對pixel
    for per_groundturth,per_predict in zip(y_true,y_scores):
        if len(np.unique(per_groundturth)) > 1 or len(np.unique(per_predict)) > 1:
            
            for per_pixel_g,per_pixel_p in zip(per_groundturth,per_predict):
                if per_pixel_g+per_pixel_p==0 and per_pixel_g>0:
                    nodule_finded[per_pixel_g]+=1
                
            
    for i in range(len(nodule_finded)):
        if nodule_finded[i]>match_size[i]:   
            TP_num+=1   
    FP_num=FP_num-TP_num
    sensitivity=TP_num/(len(nodule_size)-1)
    print("nodule size:",nodule_size)
    print("match size:",match_size)
    print("nodule_finded_pixel:",nodule_finded.values())
    
    print("TP_num:",TP_num)
    print("FP_num:",FP_num)
    print("=================================================")
    # output_folder="result/"
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    
    #Save the results
    file_perf = open(output_folder+'performances.txt', 'w')
    file_perf.write("sensitivity: "+str(sensitivity)
                    +"\nFP amount: " +str(FP_num)
                    +"\nthreshold: " +str(threshold)
                    +"\nmatch size: " +str(match_size)
                    +"\nnodule find: " +str(nodule_finded)
                    )
    file_perf.close()
    
   

    
    return sensitivity,FP_num,threshold

def average_to_txt(sensitivity,FP_num,threshold,output_folder):

    sensitivity_mean = np.mean(np.array(sensitivity))
    FP_sum=np.sum(np.array(FP_num))

    file_perf = open(output_folder, 'w')
    file_perf.write("**Total average**"
                    "\nmean sensitivity: "+str(sensitivity_mean)
                    +"\nFP amount: " +str(FP_sum)
                    +"\nthreshold: " +str(threshold[0])
                    )
    file_perf.close()
    

def resample_vs_no_resample(no_resample_sum,resample_sum,output_folder):

    file_perf = open(output_folder, 'w')
    threshold=resample_sum[2][0]
    for i in range(len(no_resample_sum[0])):
         #gap
        sensitivity_gap=resample_sum[0][i]-no_resample_sum[0][i]
        FP_num_gap=resample_sum[1][i]-no_resample_sum[1][i]

        #groth rate
        if no_resample_sum[0][i]==0:
            sensitivity_gr=None
        else:
            sensitivity_gr=resample_sum[0][i]/no_resample_sum[0][i]*100-100
        if no_resample_sum[1][i]==0:
            FP_num_gr=None
        else:
            FP_num_gr=resample_sum[1][i]/no_resample_sum[1][i]*100-100
        

        file_perf.write("**difference with resample data and no resample data form"
                    +no_resample_sum[3][i]+"**"
                     +"\nthreshold: "+str(threshold)
                    +"\nsensitivity:\nno resample:"+str(no_resample_sum[0][i])+",resample:"+str(resample_sum[0][i])
                    +"\ngap:"+str(sensitivity_gap)+",Growth Rates:"+str(sensitivity_gr)+"%"
                    +"\n\nFalse Positive amount:\nno resample:"+str(no_resample_sum[1][i])+",resample:"+str(resample_sum[1][i])
                    +"\ngap:"+str(FP_num_gap)+",Growth Rates:"+str(FP_num_gr)+"%"
                    +"\n******************************************************\n\n\n"
                    )
    file_perf.close()
    



#####################main###########################
path_resample_g="batch_9_training_set/"
path_resample_p="unet3d_81_final_1_batch_9_TTA_True/"

path_no_resample_g="batch_9_no_resample/"
path_no_resample_p="unet3d_81_final_1_batch_9_no_resample_TTA_True/"

resample_g_npz=sorted(glob.glob(path_resample_g+"*.npz"))
resample_p_npz=sorted(glob.glob(path_resample_p+"*.npz"))
no_resample_g_npz=sorted(glob.glob(path_no_resample_g+"*.npz"))
no_resample_p_npz=sorted(glob.glob(path_no_resample_p+"*.npz"))

indexes=[i.replace(path_resample_g,'').replace(".npz",'') for i in resample_g_npz]

if not os.path.isdir("result"):
    os.mkdir("result")

no_resample_sum=[[],[],[],[]]

resample_sum=[[],[],[],[]]


#no_resample
for i,index in enumerate(indexes):
    print("no_resample:no.",i)

    # simple_evaluate(resample_g_npz[i],resample_p_npz[i],"result/"+index + "resample")
    sensitivity,FP_num,threshold=simple_evaluate(no_resample_g_npz[i],no_resample_p_npz[i],"result/" + index + "_no_resample/")
    no_resample_sum[0].append(sensitivity)
    no_resample_sum[1].append(FP_num)
    no_resample_sum[2].append(threshold)
    no_resample_sum[3].append(index)

print("make performance analysis for no resample data")
average_to_txt(no_resample_sum[0],no_resample_sum[1],no_resample_sum[2]
,"result/no_reample_average_performance.txt")

#resample
for i,index in enumerate(indexes):
    print("resample:no.",i)
    # simple_evaluate(resample_g_npz[i],resample_p_npz[i],"result/"+index + "resample")
    sensitivity,FP_num,threshold=simple_evaluate(resample_g_npz[i],resample_p_npz[i],"result/" + index + "_resample/")
    resample_sum[0].append(sensitivity)
    resample_sum[1].append(FP_num)
    resample_sum[2].append(threshold)
    resample_sum[3].append(index)

print("make performance analysis for resample data")
average_to_txt(resample_sum[0],resample_sum[1],resample_sum[2]
,"result/reample_average_performance.txt")

#resample_vs_no_resample(no_resample_sum,resample_sum,"result/difference_with_resample_and_no_resample_data.txt")







    



    
