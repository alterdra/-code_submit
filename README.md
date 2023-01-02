玉山人工智慧公開挑戰賽 - 老子不給你penta  

preprocess.py主包含取用編號、時間以外的25筆資料後，feature selections, imputation, encoding的方式。  
train.py則包含使用的模型、training及inferencing test set的過程。  

執行方式如下：
1. 前往此資料夾根目錄 
2. 在colab上執行時安裝以下套件：  
    !pip install xgboost==1.7.2    
    !pip install category_encoders   
2. 將training data, public testing data放在'./ITF_data'之下  
3. 將private testing data放在'./ITF_data/private-data'之下    
4. train.py, preprocess.py原先屬於同一個ipynb檔，請將train.py的程式碼接續在preprocess.py之下，並在根目錄下以此同一個py檔執行  

  
