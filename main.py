from sklearn import preprocessing
import pandas as pd
from os import path
from sklearn.model_selection import train_test_split
import warnings



warnings.filterwarnings('ignore')
# declarations
global X_train_scaled
global X_test_scaled
global X_train
global X_test
global y_train
global y_test
dataset_encoded = pd.DataFrame()

###
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
numeric_col = ['int16','int32','int64','float16','float32','float64']


# functions
def print_task(task):
    print()
    print('select task you want to perform')
    print()
    for tasks in task:
        print(tasks)
    print()



def func_feature_scaling():
    print()
    global X_train_scaled
    global X_test_scaled

    task_scaling = [
        '1. perform standardization',
        '2. perform normalization',
        '0. Go to task menu'
    ]
    print_task(task_scaling)
    print()
    while 1:
        task_selected = int(
            input('select the task u want to perform(press 0 to exit):') or -1)
        if task_selected == 0:
            func_task()
            continue
        elif task_selected == -1:
            print('please enter a valid input')
            continue

        elif task_selected == 1:
            print()
            X_train_scaled , X_test_scaled = standardization(X_train , X_test)
            print(X_train_scaled.describe().round(2))
            print(X_test_scaled.describe().round(2))
            print_task(task_scaling)
            continue


        elif task_selected == 2:
            X_train_scaled , X_test_scaled = normalization(X_train , X_test)
            print(X_train_scaled.describe().round(2))
            print(X_test_scaled.describe().round(2))
            print_task(task_scaling)

            continue






def func_task():
    while 1:
        print()
        task = [
            '1. Data Description',
            '2. Handling NULL Values',
            '3. Encoding Categorical Data',
            '4. Feature Scaling of the Dataset',
            '5. Download the modified dataset',
            '6. split dataset into train-test dataset',
            '7. machine learning algorithm',
            '8. get full code'
        ]
        print_task(task)
        print()


        task_selected = int(
            input('select the task u want to perform(press 0 to exit):') or -1)
        if dataset_encoded.empty == True:
            if task_selected == 0:
                print('exit')
                func_task()

            elif task_selected == -1:
                print('please enter a task to perform')
                continue


            elif task_selected == 1:
                print()
                # function for Data Description
                func_task_Description()
                print_task(task)

            elif task_selected == 2:
                print('function for Handling NULL Values')
                func_task_missing()
                continue

            # function for Encoding Categorical Data
            elif task_selected == 3:
                print(' function for Encoding Categorical Data')
                categorical_encoding()
                continue


            elif task_selected == 4:
                print('function for Feature Scaling ')
                func_feature_scaling()
                # function for Feature Scaling
                continue


            elif task_selected == 5:
                download_new_dataset(dataset)
                # function for Download the modified dataset
                continue


            elif task_selected == 6:
                train_test_split_dataset(dataset)
                # function for Download the modified dataset
                continue


            elif task_selected == 7:
                algorithm_var()
                # function for Download the modified dataset
                continue


            elif task_selected == 8:
                print('get code')
                # function for Download the modified dataset
                continue

            else:
                print('please enter a valid input')

        else:
            if task_selected == 0:
                print('exit')
                break

            elif task_selected == -1:
                print('please enter a task to perform')
                continue


            elif task_selected == 1:
                print()
                # function for Data Description
                func_task_Description()
                print_task(task)

            elif task_selected == 2:
                print('function for Handling NULL Values')
                func_task_missing()
                continue

            # function for Encoding Categorical Data
            elif task_selected == 3:
                print(' function for Encoding Categorical Data')
                categorical_encoding()
                continue


            elif task_selected == 4:
                print('function for Feature Scaling ')
                func_feature_scaling()
                # function for Feature Scaling
                continue


            elif task_selected == 5:
                download_new_dataset(dataset_encoded)
                # function for Download the modified dataset
                continue

            elif task_selected == 6:
                train_test_split_dataset(dataset_encoded)
                # function for Download the modified dataset
                continue
            else:
                print('please enter a valid input')


def func_task_Description():
    print()
    task_Description = [
        '1. describe all column',
        '2. describe a specific column ',
        '3. show dataset',
        '0. go to tasks selection'
    ]
    print_task(task_Description)
    print()
    while True:

        task_selected_description = int(input('select the task u want to perform(press 0 to exit):') or -1)
        if dataset_encoded.empty == True:

            if task_selected_description == 0:
                func_task()

            elif task_selected_description == -1:
                print('please enter a task to perform')
                continue

            elif task_selected_description == 1:
                column_describe(dataset)
                print()
                print()
                column_info(dataset)
                print_task(task_Description)
                # function column_describe(dataset)
                continue

            elif task_selected_description == 2:
                spec_col_describe(dataset)
                print_task(task_Description)
                continue
            elif task_selected_description == 3:
                display_dataset(dataset)
                print_task(task_Description)
                continue
            else:
                print('please enter a valid input')
        
        
        else:
            if task_selected_description == 0:
                func_task()

            elif task_selected_description == -1:
                print('please enter a task to perform')

            elif task_selected_description == 1:
                column_describe(dataset_encoded)
                print()
                print()
                column_info(dataset_encoded)
                print_task(task_Description)
                # function column_describe(dataset)
                continue

            elif task_selected_description == 2:
                spec_col_describe(dataset_encoded)
                print_task(task_Description)
                continue
            elif task_selected_description == 3:
                display_dataset(dataset_encoded)
                print_task(task_Description)
                continue
            else:
                print('please enter a valid input')


def func_task_missing():
    print()
    task_missing = [
        '1. Number of missing values in each column',
        '2. remove unit from a particular column',
        '3. Remove a particular column',
        '4. Filling missing values using "MEAN"',
        '5. Filling missing values using "MEDIAN"',
        '6. Filling missing values using "MOST-FREQUENT VALUE"',
        '7. Display dataset',
        '0. go to tasks selection'
    ]
    print_task(task_missing)

    while True:

        task_selected_missing = int(input('select the task u want to perform(press 0 to exit):') or -1)

        if dataset_encoded.empty == True:
            if task_selected_missing == 0:
                func_task()
                continue

            elif task_selected_missing == -1:
                print('please enter a task to perform')
                continue

            elif task_selected_missing == 1:
                missing_sum(dataset)
                print_task(task_missing)
                continue



            elif task_selected_missing == 2:
                remove_unit_col(dataset)
                print_task(task_missing)
                continue



            elif task_selected_missing == 3:
                drop_column_missing(dataset)
                print_task(task_missing)
                continue



            elif task_selected_missing == 4:
                missing_mean(dataset)
                print_task(task_missing)
                continue



            elif task_selected_missing == 5:
                missing_median(dataset)
                print_task(task_missing)
                continue



            elif task_selected_missing == 6:
                missing_mode(dataset)
                print_task(task_missing)
                continue



            elif task_selected_missing == 7:
                display_dataset(dataset)
                print_task(task_missing)
                continue

            else:
                print()
                print('please enter a valid input')
                print()

        
        else:
            if task_selected_missing == 0:
                func_task()
                continue


            elif task_selected_missing == -1:
                print('please enter a task to perform')
                continue



            elif task_selected_missing == 1:
                missing_sum(dataset_encoded)
                print_task(task_missing)
                continue



            elif task_selected_missing == 2:
                remove_unit_col(dataset_encoded)
                print_task(task_missing)
                continue



            elif task_selected_missing == 3:
                drop_column_missing(dataset_encoded)
                print_task(task_missing)
                continue



            elif task_selected_missing == 4:
                missing_mean(dataset_encoded)
                print_task(task_missing)
                continue



            elif task_selected_missing == 5:
                missing_median(dataset_encoded)
                print_task(task_missing)
                continue



            elif task_selected_missing == 6:
                missing_mode(dataset_encoded)
                print_task(task_missing)
                continue



            elif task_selected_missing == 7:
                display_dataset(dataset_encoded)
                print_task(task_missing)
                continue

            else:
                print()
                print('please enter a valid input')
                print()


def categorical_encoding():
    print()
    task_encoding = [
        '1. show unique value in categorical column in dataset',
        '2. perform one hot encoding',
        '3. perform label encoding(prefered for target variable)',
        '0. Go to task menu'
    ]
    print_task(task_encoding)
    print()
    while 1:
        task_selected = int(
            input('select the task u want to perform(press 0 to exit):') or -1)
        if task_selected == 0:
            func_task()
            continue

        elif task_selected == -1:
            print('please enter a task to perform')
            continue

        elif task_selected == 1:
            print()
            unique_values(dataset)
            print_task(task_encoding)
            continue


        elif task_selected == 2:

            dataset_encoded=one_hot_encoder(dataset)
            print_task(task_encoding)
            continue

        elif task_selected == 3:

            dataset_encoded=label_encoder(dataset)
            print_task(task_encoding)
            continue


        else:
            print('please enter a valid input')




def data_Input():
    supported_extension = '.csv'
    paths = (input('please enter the full path of your dataset:') or '-1')
    if paths == "-1":
        print('please enter a dataset full path with extension')
        return data_Input()
    else:
        filename, file_extension = path.splitext(paths)
        print(file_extension)
        if file_extension != supported_extension:
            print('provide file name with path and extension')
            return data_Input()
        else:
            try:
                df = pd.read_csv(filename + file_extension)
                print('dataset successfully loaded')
                return df
            except:
                print("dataset not found please provide valid file")
                return data_Input()




def show_columns(dataset):
    column_name=dataset.columns
    for columns in column_name:
        print (columns+'\t')




def target_column(column_name):
    target_columns = (input('please select the target column:') or '-1')
    if target_columns == '-1':
        print('please enter a column name')
        return target_column(column_name)
    else:
        i = 0
        for column in column_name:
            if target_columns == column:
                return column

            else:
                i = 2
                continue
        if i == 2:
            i = 0
            print('please enter valid target column name from dataset')
            return target_column(column_name)


def column_info(dataset):
    return dataset.info()



def column_describe(dataset):
    return print(dataset.describe())



def spec_col_describe(dataset):
    column_name=input('please enter the specific column:')
    for colname in column_name_dict:
        if column_name == colname:
            return print(dataset[column_name].describe())
        else:
            continue
    print('please enter a valid column name')
    spec_col_describe(dataset)



def display_dataset(dataset):
    no_col=int(input('please enter number of row you want to print:') or 5)
    if no_col == None :
        print('enter a valid input')
        return display_dataset(dataset)
    else:
        return print(dataset.head(no_col))



def to_lower_case(col):

    for i in range(len(col)):
        col[i] = col[i].lower()
    return col


def remove_unit_col(dataframe):
    while True:
        print('press "0" to go to task menu')
        print()
        show_columns(dataframe)

        col_names = input('Enter the name of column where u want to remove the unit from (one at a time):' or '0')
        if col_names == '0':
            func_task_missing()
        i=0
        for column in column_name_dict:
            if col_names == column:
                dataframe['col'] = dataframe[col_names].str.replace(r'\D', '', regex=True)
                dataframe['col'] = pd.to_numeric(dataframe['col'])
                dataframe[col_names] = dataframe['col']
                print('unit changed')
                return remove_unit_col(dataframe)
            else :
                i=2
                continue
        if i == 2:
            print('please enter a valid column')
            return remove_unit_col(dataframe)


def missing_sum(dataframe):
    print(dataframe.isnull().sum())



def drop_column_missing(dataframe):
    col_name = input('please enter the column you want to drop(press 0 to go back):')
    if col_name =="0":
        func_task_missing()
    else:
        try:
            for column in column_name_dict:
                if column == col_name:
                    dataframe.drop(col_name, axis=1, inplace=True)
                else:
                    continue
        except:
            print('please enter a valid input')
            return drop_column_missing(dataframe)


def missing_mean(dataframe):
    int_col = dataframe.select_dtypes(include=numeric_col)
    print('Column where you can use "MEAN" to replace the missing values:')
    print()
    print(int_col.isnull().sum())
    print()
    task_mean = [
        '1. fill missing values in a specific column using MEAN',
        '2. fill missing values in whole dataset using MEAN',
        '0. go to task menu'
    ]
    while True:
        print_task(task_mean)
        task_missing_mean = int(input('select the task u want to perform:'))

        if task_missing_mean == 0:
            func_task_missing()

        elif task_missing_mean == 1:
            col_name = input('Enter a particular column name you want to fill missing value(one at a time):')
            dataframe[col_name].fillna(int_col[col_name].mean(), inplace=True)

        elif task_missing_mean == 2:
            for i in int_col.columns[int_col.isnull().any(axis=0)]:
                dataframe[i].fillna(int_col[i].mean(), inplace=True)
                int_col[i].fillna(int_col[i].mean(), inplace=True)
        print(int_col.isnull().sum())



def missing_median(dataframe):
    int_col = dataframe.select_dtypes(include=numeric_col)
    print('Column where you can use "MEDIAN" to replace the missing values:')
    print()
    print(int_col.isnull().sum())
    print()
    task_median = [
        '1. fill missing values in a specific column using MEDIAN',
        '2. fill missing values in whole dataset using MEDIAN',
        '0. go to task menu'
    ]
    while True:
        print_task(task_median)
        task_missing_median = int(input('select the task u want to perform:'))

        if task_missing_median == 0:
            func_task_missing()

        elif task_missing_median == 1:
            col_name = input('Enter a particular column name you want to fill missing value(one at a time):')
            dataframe[col_name].fillna(int_col[col_name].median(), inplace=True)

        elif task_missing_median == 2:
            for i in int_col.columns[int_col.isnull().any(axis=0)]:
                dataframe[i].fillna(int_col[i].median(), inplace=True)
                int_col[i].fillna(int_col[i].median(), inplace=True)
        print(int_col.isnull().sum())



def missing_mode(dataframe):
    int_col = dataframe.select_dtypes(include=numeric_col)
    print('Column where you can use "MODE" to replace the missing values:')
    print()
    print(int_col.isnull().sum())
    print()
    task_mode = [
        '1. fill missing values in a specific column using MODE',
        '2. fill missing values in whole dataset using MODE',
        '0. go to task menu'
    ]
    while True:
        print_task(task_mode)
        task_missing_mode = int(input('select the task u want to perform:'))

        if task_missing_mode == 0:
            func_task_missing()

        elif task_missing_mode == 1:
            col_name = input('Enter a particular column name you want to fill missing value(one at a time):')
            dataframe[col_name].fillna(int_col[col_name].mode()[0], inplace=True)

        elif task_missing_mode == 2:
            for i in int_col.columns[int_col.isnull().any(axis=0)]:
                dataframe[i].fillna(int_col[i].mode()[0], inplace=True)
                int_col[i].fillna(int_col[i].mode()[0], inplace=True)
        print(int_col.isnull().sum())



def unique_values(dataset):
    object_col = dataset.select_dtypes(include='object')
    i = 0

    for col in object_col:
        col_dict = {col: str(dataset[col].nunique())}
        print(col_dict)
        i = i + 1



def one_hot_encoder(dataframe):
    return pd.get_dummies(dataframe)



def normalization(X_train_dataframe,X_test_dataframe):
    scaler=preprocessing.MinMaxScaler()
    col_name=X_train_dataframe.columns.values
    train_scaled = scaler.fit_transform(X_train_dataframe)
    test_scaled = scaler.fit_transform(X_test_dataframe)

    dataframe_train_scaled = pd.DataFrame(train_scaled, columns=col_name)
    dataframe_test_scaled = pd.DataFrame(test_scaled, columns=col_name)
    return dataframe_train_scaled,dataframe_test_scaled


def standardization(X_train_dataframe,X_test_dataframe):
    scaler = preprocessing.StandardScaler()
    col_name = X_train_dataframe.columns.values
    train_scaled = scaler.fit_transform(X_train_dataframe)
    test_scaled = scaler.fit_transform(X_test_dataframe)

    dataframe_train_scaled = pd.DataFrame(train_scaled, columns=col_name)
    dataframe_test_scaled = pd.DataFrame(test_scaled, columns=col_name)
    return dataframe_train_scaled, dataframe_test_scaled




def download_new_dataset(dataframe):
    data_dict = {}
    for column in dataframe.columns.values:
        data_dict[column] = dataframe[column]
    newFileName = (input(
        "\nEnter the  FILENAME you want? (Press 0 to go back):  ") or '-1')
    if newFileName == "0":
        return
    elif newFileName == '-1':
        print('please enter a name')
        return download_new_dataset(dataframe)
    else:
        newFileName = newFileName + ".csv"
        # index=False as this will not add an extra column of index.
        global new_dataset
        pd.DataFrame(dataframe).to_csv(newFileName, index=False)
        new_dataset = pd.read_csv(newFileName)
        print(new_dataset)



def train_test_split_dataset(dataframe):
    global X_train
    global X_test
    global y_train
    global y_test

    X = dataframe.drop(target_columns, axis=1)
    y = dataframe[target_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('shape of X_train = ', X_train.shape)
    print('shape of X_test = ', X_test.shape)
    print('shape of y_train = ', y_train.shape)
    print('shape of y_test = ', y_test.shape)



def label_encoder(dataframe):
    print()
    from sklearn.preprocessing import LabelEncoder
    print(dataframe[target_columns].value_counts())
    la = LabelEncoder()
    dataframe[target_columns] = la.fit_transform(dataframe[target_columns])
    return dataframe


def algorithm_var():
    import time
    global shouldfinish
    global y_train
    y_train_int = y_train.astype(int)

    try:
        from sklearn.linear_model import Ridge, Lasso
        for i in range(1, 10):
            regressor = Lasso(alpha=i)
            regressor.fit(X_train_scaled, y_train)
            print(f"lasso for alpha {i}")
            y_pred = regressor.predict(X_test_scaled)
            print(round(regressor.score(X_test_scaled, y_test) * 100, 4))
            print()
            regressor = Ridge(alpha=i)
            regressor.fit(X_train_scaled, y_train)
            print(f"ridge for alpha {i}")
            y_pred = regressor.predict(X_test_scaled)
            print(round(regressor.score(X_test_scaled, y_test) * 100, 4))
            print()
    except:
        pass

    print()
    print("---------------------------------------------------------------------------")
    elapsed = 70

    try:
        from sklearn.linear_model import LinearRegression

        clf = LinearRegression()
        clf.fit(X_train_scaled, y_train)
        print("LinearRegression")
        y_pred_LinearRegression = clf.predict(X_test_scaled)
        LinearRegression_score = (round(clf.score(X_test_scaled, y_test) * 100, 4))
        print(LinearRegression_score)
        print()
        print("---------------------------------------------------------------------------")
    except:
        pass

    try:
        from sklearn import metrics, svm

        clf = svm.SVR()
        clf.fit(X_train_scaled, y_train)

        print("SVR")
        y_pred_SVR = clf.predict(X_test_scaled)
        print(round(clf.score(X_test_scaled, y_test) * 100, 4))
        print()
        print("---------------------------------------------------------------------------")

    except:
        pass

    try:
        from sklearn.tree import DecisionTreeRegressor

        clf = DecisionTreeRegressor(random_state=42)
        clf.fit(X_train_scaled, y_train)
        print("DecisionTreeRegressor")
        y_pred_DecisionTreeClassifier = clf.predict(X_test_scaled)
        print(round(clf.score(X_test_scaled, y_test) * 100, 4))
        print()
        print("---------------------------------------------------------------------------")
    except:
        pass

    try:
        import xgboost

        clf = xgboost.XGBRegressor(random_state=42)
        clf.fit(X_train_scaled, y_train)
        print("xgboost")
        y_pred_xgboost = clf.predict(X_test_scaled)
        print(round(clf.score(X_test_scaled, y_test) * 100, 4))
        print()
        print("---------------------------------------------------------------------------")
    except:
        pass

    try:
        from sklearn.ensemble import BaggingRegressor
        clf = BaggingRegressor(random_state=42)
        clf.fit(X_train_scaled, y_train)
        print("BaggingRegressor")
        y_pred_BaggingRegressor = clf.predict(X_test_scaled)
        print(round(clf.score(X_test_scaled, y_test) * 100, 4))
        print()
        print("---------------------------------------------------------------------------")
    except:
        pass

    try:
        from sklearn.neighbors import KNeighborsRegressor

        clf = KNeighborsRegressor()
        clf.fit(X_train_scaled, y_train_int)
        print("KNeighborsRegressor")
        y_pred_KNeighborsClassifier = clf.predict(X_test_scaled)
        print(round(clf.score(X_test_scaled, y_test) * 100, 4))
        print("---------------------------------------------------------------------------")
    except:
        pass

    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        clf.fit(X_train_scaled, y_train)
        print("LogisticRegression")
        y_pred = clf.predict(X_test_scaled)
        print(round(clf.score(X_test_scaled, y_test) * 100, 4))
        print("---------------------------------------------------------------------------")
    except:
        pass
    try:
        from sklearn.tree import DecisionTreeClassifier

        clf = DecisionTreeClassifier()
        clf.fit(X_train_scaled, y_train)
        print("DecisionTreeClassifier")
        y_pred = clf.predict(X_test_scaled)
        print(round(clf.score(X_test_scaled, y_test) * 100, 4))
        print("---------------------------------------------------------------------------")
    except:
        pass
    try:

        from sklearn.neighbors import KNeighborsClassifier

        clf = KNeighborsClassifier()
        clf.fit(X_train_scaled, y_train)
        print("KNeighborsClassifier")
        y_pred = clf.predict(X_test_scaled)
        print(round(clf.score(X_test_scaled, y_test) * 100, 4))
        print("---------------------------------------------------------------------------")
    except:
        pass
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train_scaled, y_train)
        print("LinearDiscriminantAnalysis")
        y_pred = clf.predict(X_test_scaled)
        print(round(clf.score(X_test_scaled, y_test) * 100, 4))
        print("---------------------------------------------------------------------------")
    except:
        pass

    try:
        from sklearn.naive_bayes import GaussianNB

        clf = GaussianNB()
        clf.fit(X_train_scaled, y_train)
        print("GaussianNB")
        y_pred = clf.predict(X_test_scaled)
        print(round(clf.score(X_test_scaled, y_test) * 100, 4))
        print("---------------------------------------------------------------------------")

    except:
        pass

    try:
        from sklearn.svm import SVC
        clf = SVC()
        clf.fit(X_train_scaled, y_train)
        print("SVC")
        y_pred = clf.predict(X_test_scaled)
        print(round(clf.score(X_test_scaled, y_test) * 100, 4))
        print("---------------------------------------------------------------------------")
    except:
        pass
#function ends


### programs start
print('\t\t\t(^O^)\tHII Welcome To ' +
      '\033[1m'+'EFFORTLESS PREPROCESSER'+'\033[0m'+'\t(^O^)')
print()
print()

# path of dataset=C:\Users\Admin\Desktop\ml project\train-data.csv
dataset = data_Input()
print()



print()

# displaying all column
print('this are the following column in the given dataset:')
show_columns(dataset)
column_name_dict = dataset.columns.values


print()

# asking for the target column
target_columns = target_column(column_name_dict)
print('target column:'+target_columns)
print()

func_task()

