def predict(xx,df):
    import  numpy   as  np
    import  pandas  as  pd
    import sklearn
    from    sklearn.tree    import  DecisionTreeRegressor
    from    sklearn.linear_model    import  LinearRegression
    from sklearn.naive_bayes import GaussianNB
    from    sklearn.model_selection import  train_test_split
    import  matplotlib.pyplot   as plt
    from sklearn.metrics import accuracy_score
    from sklearn.naive_bayes import BernoulliNB
    import warnings
    from datetime import date
    from datetime import datetime, timedelta
    import Technicals35v
    import prediction_Tech
    from prediction_Tech import predict

    pd.set_option("display.max_columns", 100)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 200)
    pd.set_option('display.max_columns', 0)
    pd.set_option('display.max_columns', None)

    
    
##    from Technicals35v import Techinicalsbb
    warnings.filterwarnings("ignore")
###############################    Trigger for moving today ot tommorows ##########
##    df=df[:df.shape[0]-1]
#########################################################################    
    print(df.tail(1))
    ##print('\n')

    print(df.shape,' original ',xx)
    forecast_out = 1
    df['Prediction'] = df[['Close']].shift(-forecast_out)
    ##df['Prediction'] = df[['Close']].shift(-forecast_out)
    ##df['Prediction'] = df[['Open']].shift(-forecast_out)
    print(df.shape,' forecasrt with prediction column added ',xx)
    print('\n')

    ###########################################################################
    X = np.array(df.drop(['Prediction'], 1))[:-forecast_out]
    y = np.array(df['Prediction'])[:-forecast_out]

    ##print(len(X),' -------------------------')
    ##print(len(y),' -------------------------')
    ##print(len(X)+len(y))
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=1234)
    ##x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

    tree = DecisionTreeRegressor().fit(x_train, y_train)
    lr = LinearRegression().fit(x_train, y_train)
    ##clf = BernoulliNB().fit(x_train,y_train)

    ##print(x_train,' ---- xtrain')
    ##print(y_train, ' ---- ytran')
    ##print('\n\n\n')
    ##gnb = GaussianNB().fit(x_train, y_train)

    ##clf = BernoulliNB()
    ##clf.fit(X_train,y_train)
    ##y_pred = clf.predict(X_test)
    ##print(accuracy_score(y_test, y_pred))



    X_future = df.drop(['Prediction'], 1)[:-forecast_out]
    X_future = X_future.tail(forecast_out)
    X_future = np.array(X_future)

    df['tree_p']=''
    df['tree_c']=''
    df['lr_p']=''
    df['lr_c']=''

    ###########################################################################
    ##if date.today()==df.index.tail(1):
    mm=str(df.index[-1]).split(' ')[0]
    print(mm,' mm')
    print(date.today(),' todays')
    if str(date.today())!=str(mm):
        print(' --------------------------- ',)
        print('Today Prediction ',date.today(),'   ',xx, ' stock will close at: ',' ***************************************** ')
        tree_prediction = tree.predict(X_future)
        tree_confidence = tree.score(x_test, y_test)
        print('1) Tree ',tree_prediction,'  ',"Prediction accuracy ", tree_confidence,'  ',xx)

        lr_prediction = lr.predict(X_future)
        lr_confidence = lr.score(x_test, y_test)
        print('2) LR ',lr_prediction,'  ',"Prediction accuracy ", lr_confidence,'  ',xx)
        print(' rambo ')
##        print(df)
        df['tree_p']=tree_prediction
        df['tree_c']=tree_confidence
        df['lr_p']=lr_prediction
        df['lr_c']=lc_confidence
        
    if str(date.today())==str(mm):
        print(' --------------------------- ',)
        print('Tommorows Prediction ',date.today()+timedelta(1),'   ',xx, 'stock will close at: ',' ***************************************** ')
        tree_prediction = tree.predict(X_future)
        tree_confidence = tree.score(x_test, y_test)
        print('1) Tree ',tree_prediction,'  ',"Prediction accuracy ", tree_confidence,'  ',xx)

        lr_prediction = lr.predict(X_future)
        lr_confidence = lr.score(x_test, y_test)
        print('2) LR ',lr_prediction,'  ',"Prediction accuracy ", lr_confidence,'  ',xx)
        print(' babu')
        df['tree_p']=tree_prediction
        df['tree_c']=tree_confidence
        df['lr_p']=lr_prediction
        df['lr_c']=lc_confidence
##        print(df,'333')
    return(xx,tree_prediction,tree_confidence,lr_prediction, lr_confidence)
      
