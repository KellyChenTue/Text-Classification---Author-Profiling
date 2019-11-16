"""
Author Profiling
Goal: Given a text, predict the gender of the author.
Author of the code: Chen, Pin-Zhen

"""
import csv
import argparse
import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import precision_recall_fscore_support

def read_data(filename, external = None):
    label_lst = []
    text_lst = []
    with open(filename, 'r') as csv_readerfile:
        reader = csv.reader(csv_readerfile, delimiter='\t')
        #skip header
        next(reader,None)
        for r in reader:
            label_lst.append(r[0])
            text_lst.append(r[1])

    if external:
        print("add external data")
        with open(external, 'r') as csv_readerfile:
            reader = csv.reader(csv_readerfile, delimiter='\t')
            # skip header
            next(reader, None)
            for r in reader:
                r[1]=r[1].replace("<br />", " ")
                if len(r[1]) < 150:
                    label_lst.append(r[0])
                    text_lst.append(r[1])
    return label_lst, text_lst

def encode(label=None, text=None, test_label=None, test_text=None):

    # Label
    # gender to int
    # when we predict on an unseen data, there is no label.
    # So we return y_test_int = None
    y_test_int = None
    if label:
        y_train_int = [0 if x == "m" else 1 for x in label]
    if test_label:
        y_test_int = [0 if x == "m" else 1 for x in test_label]

    # Tokenizing & Filtering the text
    # min_df=3, discard words appearing in less than 3 documents
    # max_df=0.6, discard words appering in more than 60% of the documents
    # sublinear_tf=True, use sublinear weighting
    # use_idf=True, enable IDF
    vectorizer = TfidfVectorizer(min_df=3,
                                 max_df=0.6,
                                 sublinear_tf=True,
                                 use_idf=True
                                 )

    # Create the vocabulary and the feature weights from the training data
    X_train_vectors = vectorizer.fit_transform(text)
    X_test_vectors = vectorizer.transform(test_text)

    return y_train_int, X_train_vectors, y_test_int,X_test_vectors

def train(y_train, X_train , y_test, X_test):
    """
    y_train :  label (gender) in trainset
    X_train : features (text) in trainset
    y_test : label (gender) in testset
    X_test : features (text) in testset

    train the model by SGD, SVM and logistic regression on the training set,
    choose the best model among SGD, SVM and logistic regression.
    Also use Ensemble to vote for the best decision.
    Print out the scores from the best model and from Ensemble.
    save the best model to disk as pre-trained model.

    """
    random_state = 128

    # params to tune
    # create params
    params_sgd = {
        'loss': ['log', 'modified_huber', 'hinge'],
        'learning_rate': ['constant', 'adaptive', 'invscaling', 'optimal'],
        'eta0': [0.1, 1],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'class_weight': ['balanced', None],
    }
    params_log = [{'C': [0.01,0.1, 1, 2, 5, 10],'penalty': ['l2', 'l1'],'class_weight': ['balanced', None]},
                  {'multi_class': ['multinomial'], 'solver':['lbfgs'], 'penalty': ['l2'] }]

    params_svm = [ {'C': [0.01, 0.1, 1, 10, 50, 100, 500, 1000], 'kernel': ['linear'], 'class_weight': ['balanced', None]},
        {'C': [0.01, 0.1, 1, 10, 50, 100, 500, 1000], 'gamma': [0.001, 0.0001, 0.1],
         'kernel': ['rbf'],'class_weight': ['balanced', None]}
                  ]

    # set up models
    sgd = SGDClassifier(random_state=random_state)
    log = LogisticRegression(random_state=random_state)
    svm = sk.svm.SVC(random_state=random_state)
    model = [sgd, log, svm]
    scores = {}
    # tune the models
    for m in model:
        if m == sgd:
            clf = GridSearchCV(sgd, params_sgd,  cv=2, scoring='f1_macro').fit(X_train, y_train)
            clf.best_params_["model"] = "sgd"
            scores[clf.best_score_] = clf.best_params_
            sgd_best = clf.best_estimator_

        if m == log:
            clf = GridSearchCV(log, params_log,  cv=2, scoring='f1_macro').fit(X_train, y_train)
            clf.best_params_["model"] = "log"
            scores[clf.best_score_] = clf.best_params_
            log_best = clf.best_estimator_

        if m == svm:
            clf = GridSearchCV(svm, params_svm,  cv=2, scoring='f1_macro').fit(X_train, y_train)
            clf.best_params_["model"] = "svm"
            scores[clf.best_score_] = clf.best_params_
            svm_best = clf.best_estimator_

    print('sgd: {}'.format(sgd_best.score(X_test, y_test)))
    print('log: {}'.format(log_best.score(X_test, y_test)))
    print('svm:: {}'.format(svm_best.score(X_test, y_test)))

    # get best model from tuning: choose the model which has the best score
    # pop the item 'model' and set_params
    best_model = scores[max(scores)]
    if best_model['model'] == "sgd":
        best_model.pop('model')
        final_model = sgd.set_params(**best_model)
    elif best_model['model'] == "log":
        best_model.pop('model')
        final_model = log.set_params(**best_model)
    elif best_model['model'] == "svm":
        best_model.pop('model')
        final_model = svm.set_params(**best_model)

    print("train model and predict on testset with the best params.......")
    clf = final_model.fit(X_train, y_train)
    pred = clf.predict(X_test)
    pred_train = clf.predict(X_train)
    print("pred:", pred)
    print("y_test", y_test)
    scores = precision_recall_fscore_support(y_test, pred, average='macro')
    scores_train = precision_recall_fscore_support(y_train, pred_train, average='macro')
    print("best model's prediction F1-score on test data: ", scores[2])
    print("best model's prediction F1-score on training data : ", scores_train[2])

    # save final_model
    save_model_name = 'finalized_model.sav'
    pickle.dump(final_model, open(save_model_name, 'wb'))


    # Dummy: random, majority and logistic comparison
    Dummy_strategies = ['uniform', 'most_frequent']
    train_and_test = [(X_train, y_train), (X_test, y_test)]

    print("\t\t\tp_train\tr_train\tf_train\tp_test\tr_test\tf_test\n")
    for st in Dummy_strategies:
        c = DummyClassifier(strategy=st, random_state=random_state)
        if st == 'uniform':
            print("Random\t\t", end="")
        elif st == 'most_frequent':
            print("majority\t", end="")
        c.fit(X_train, y_train)
        for item in train_and_test:
            pred = c.predict(item[0])
            p_score = precision_score(item[1], pred, average="macro") * 100
            r_score = recall_score(item[1], pred, average="macro") * 100
            f_score = f1_score(item[1], pred, average="macro") * 100
            print(str("%0.2f" % p_score) + "\t" + str("%0.2f" % r_score) + "\t" + str("%0.2f" % f_score) + "\t", end="")
            #print("\nDummy pred: ", pred, "\n")
        print("\n")


    # Ensemble
    from sklearn.ensemble import VotingClassifier
    # create a dictionary of our models
    estimators = [('sgd', sgd_best), ('log', log_best), ('svm', svm_best)]
    # create our voting classifier, inputting our models
    # voting= 'hard: for each text, decide the gender based on the votes from three models
    ensemble = VotingClassifier(estimators, voting='hard')
    # fit model to training data and test
    ensemble.fit(X_train, y_train)
    pred = ensemble.predict(X_test)
    s = precision_recall_fscore_support(y_test, pred, average='macro')
    print("Ensemble F1-score: ", s[2])

    #return the final model to use in detect
    return final_model

def detect(text_vectors, modelname):
    """
    use pre-trained model to detect on an unseen data.
    Because the model is trained, the speed of prediction(detection) can
    be faster.
    However, before we call this model, the argument text_vectors should
    be encoded as the same scheme as the training text.

    """
    loaded_model = pickle.load(open(modelname, 'rb'))
    result = loaded_model.predict(text_vectors)
    # transform back to 'f' or 'm' , it's clear to see what the result is.
    result = ["m" if x == 0 else "f" for x in result]
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Author-Text to detect gender.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--trainset', required=False,
                        metavar="/path/to/dataset/",
                        help='an tsv file of text to train')
    parser.add_argument('--testset', required=False,
                        metavar="/path/to/dataset/",
                        help='an tsv file of text to train')
    parser.add_argument('--text', required=False,
                        metavar="path",
                        help='an txt file of text to detect')
    parser.add_argument('--model', required=False,
                        metavar="path",
                        help='a pre-trained model')
    parser.add_argument('--externalData', required=False, help='an external data')
    args = parser.parse_args()
    # call train or detect method
    if args.command == "train":
        # if there is externalData, merge into args.trainset
        if args.externalData:
            label, text = read_data( args.trainset, external = args.externalData)
        else:
            label, text = read_data(args.trainset)

        test_label, test_text = read_data(args.testset)
        en_label, en_feature, en_tst_label, en_tst_feature  = encode(label,text, test_label, test_text)
        train(en_label, en_feature,  en_tst_label ,en_tst_feature)

    elif args.command == "detect":
        test_label, test_text = read_data(args.text)
        # to make sure the encode scheme to be consistent, we have to encode trainset
        train_l, train_text = read_data(args.trainset)
        _ , _ , _, en_text = encode(train_l, train_text, test_label=None, test_text=test_text)
        pred = detect(en_text,args.model)
        print("prediction: ", pred)
