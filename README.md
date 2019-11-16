# Author profiling - Report

# what can you expect from this model
 - Goal: dectect author's gender given a text
 - you may use my pre-trained model to predict your data
 - or apply your own data and train the model. After training, predict an unseen text.
 - I'll also explain my approach below
# requirments (System)
Mac or Linux should be both fine.

# Installation
 ```sh
 pip install scikit-learn
 pip install pickle
 git clone https://github.com/snlp2019/a7-a7-kelly.git
```
# Data 
- Format: 1) should be .tsv file 2) using \<tab> to separate gender and text
- I used the data from the course 2019snlp in Tübingen, and the data will be in the repository after you clone this git.
    - 'essays-train.tsv' for both training and tuning your model, and 'essays-test.tsv' for testing. 
- There is also external data from PAN: https://pan.webis.de/data.html 
    - PAN provide dataset in xml file, so I provide here an xml2tsv.py in a favor of the model.


# Tutorials
Follow the steps, you will be able to run a demo and get the results as I do. Running a demo can make sure the model works fine. And then you are ready to detect on your own data, or even train your own model!
 ### 1.1 Reproduce my results
 Note: No need to change the argument.
 ```sh
python author_profiling.py train --trainset=essays-train.tsv --testset=essays-test.tsv
```

 ### 1.2 Expected outcome
My results:

|Model and Prediction|F1-score|
|---|:---:|
|best model's prediction  on test data:        |  0.6419437340153453 |
|best model's prediction on training data :   |  1.0           |
|Ensemble on test data:                  |  0.4949494949494949|
 
| BaseLine | p_train | r_train | f_train | p_test | r_test | f_test |
|---|:---:|:---:|---|:---:|:---:|:---:|
| Random   |  55.63 |  55.65  | 55.53  | 29.29 | 29.29  |  29.29   |
| majority |  26.67 |  50.00  | 34.78  | 27.50 | 50.00  |  35.48   |

('p' for prediction, 'r' for recall and 'f' for f1_macro scores)


Now you are ready to train your own model! Or use my model to predict your text!
 
 - Use my model for detection : 2.2
    
      or
 - Train on your own data: 2.1, 2.2
 
 ### 1.3 : (Optional) using external data
 Another option to use an external data with my model: 
simply add the argument '--externalData.'
- The external data was .xml file from PAN databank(https://pan.webis.de/data.html), I used 'xml2tsv.py' to convert the data to '.tsv' file and saved them to the file 'data.tsv'
- You will find 'data.tsv' in the repository as well.
 ```sh
python author_profiling.py train --trainset=essays-train.tsv --testset=essays-test.tsv --externalData=data.tsv 
 ```
 
 ### 2.1 Train the model on your own data
- Prepare data in .tsv file, be sure that you use a <tab> to separate label and text, and the label should be 'f' or 'm'. The data should look like this: 

```
Label <tab> Text
f <tab> Hello World! This is an example text.
m <tab> Hello Python! This is an example text.
```
- Train the model on your data with the following command.  Note: please CHANGE the PATH for the argument 'trainset' and 'testset'.
```
python author_profiling.py train --trainset=essays-train.tsv --testset=essays-test.tsv 
```
- After training, you should expect to see an '.sav' file in the directory. That is the model you just trained, and we will use it for detection (prediction).
### 2.2 Detect
- Now we can predict on an unseen text with a model (either the model I provided or the model you just trained.)
- run the following command: 
 Note: you should give an unseen text to the argument 'text.' You might want to change the argument for 'model,' if you want to use your own trained model. 
- To encode the text as the same scheme as the model, please add the argument 'trainset' of your training set.
```
python author_profiling.py detect --model=finalized_model.sav --text=essays-test.tsv --trainset=essays-train.tsv 
```
 # More details and short Q&As:
 ##### What models did I use?
 In author_profiling.py, the method ’train()’ has 3 different kinds of methods from sklearn, which are SGD, SVM and LogisticRegression.
 ##### How did I tune my models?
 For each method(SGD,SVM and LogisticRegression), I prepared different parameters to tune by using 'GridSearch,' which gives me a best model among the combination of parameters. The parameters differ from each method.
 ##### How do I choose the best model among all methods and parameters?
 After all GridSearch is done, I've got the best model for each method. Then I compared among those 3 candidates to see which one has the best score. And that's the final model I use to predict on test data, and then I save to the PC. Also, that's the model we will use for the detection.
 
 ##### Did I employ any preprocessing or feature selection steps?
Yes, in my tf-idf approach, I set min_df=3, which discard words appearing in less than 3 documents. 
And I set max_df=0.6, which discard words appering in more than 60% of the documents.
 ##### Did I use external data?
 Yes, I tried to use the data from PAN(https://pan.webis.de/data.html)
 
 But the good results I report here are without external data.
 
 # A Report of my results:
 ##### Report the appropriate scores for your model(s)
|Model and Prediction|F1-score|
|----------|:------:|
|best model's prediction  on test data:        |  0.6419437340153453 |
|best model's prediction on training data :   |  1.0           |
 ##### If you used multiple models, or alternative data sources and/or other settings with the same model, please report all.

- Ensemble F1-score:  0.4949494949494949

 ##### A comparison with an appropriate trivial baseline.

| BaseLine | p_train | r_train | f_train | p_test | r_test | f_test |
|-----|:---:|---:|---|:---:|:---:|:---:|
| Random   |  55.63 |  55.65  | 55.53  | 29.29 | 29.29  |  29.29   |
| majority |  26.67 |  50.00  | 34.78  | 27.50 | 50.00  |  35.48   |

- In comparison, my model has 64% F1-score on the test set and I also get 49% F1-score from the Ensemble model.
- However, BaseLine has only 29% and 35% for random and majority respectively.

License
----

Chen, Pin-Zhen

Computational Linguistics, 
Tübingen University
