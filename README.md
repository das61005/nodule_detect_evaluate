# nodule_detect_evaluate

### 比較有無resample過的nodule detection 結果差異，對其效能進行評估

將ground truth和預測結果對比，若預測的nodule和ground的nodule有重疊超過一個閥值則算入true positive，反之則是false positive

### input:

gound turth 和預測的nodule資料(nii.gz)

resample過的gound turth 和預測的nodule資料(nii.gz)

### output:

每筆資料的評估結果(txt)內容包含sensitivity、false positive nodule數量、評估資料的閥值(預設為1/3)、找到nodule的大小、需要符合找到標準的大小。以及有resample和梅resmaple兩種的每筆資料平均結果，兩種結果每筆的資料的比較
