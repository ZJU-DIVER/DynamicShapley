# -*- coding: utf-8 -*-
# Copyright (c) Haocheng Xia.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Various data processing functions for demonstration."""

import os

import numpy as np
import pandas as pd
from pathlib import Path


def load_tabular_data(data_name, dict_no, train_file_name,
                      test_file_name, idx_sorted=True, t2classification=False):
    """loads iris, wine, and Blood Transfusion Service Center datasets.

    UCI iris dataset link: https://archive.ics.uci.edu/ml/machine-learning-databases/iris
    UCI adult dataset link: https://archive.ics.uci.edu/ml/machine-learning-databases/adult

    This module loads the three tabular datasets and saves train.csv and 
    test.csv (with labels) files under data_files directory

    Args:
      data_name:         'adult' or 'iris'
      dict_no:           train and test set numbers
      train_file_name:   the name of train file
      test_file_name:    the name of test file
      idx_sorted:        data tuples as the original order, default is `True`
      t2classification:  force to 2 class classification, default is `False`
    """

    # Loads datasets from links
    uci_base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'

    # iris dataset
    if data_name == 'iris':

        data_url = uci_base_url + 'iris/iris.data'

        df = pd.read_csv(data_url, header=None)

        # Column names
        df.columns = ['SepalLength', 'SepalWidth',
                      'PetalLength', 'PetalWidth', 'Class']

        # Changes string to float
        df.SepalLength = df.SepalLength.astype(float)
        df.SepalWidth = df.SepalWidth.astype(float)
        df.PetalLength = df.PetalLength.astype(float)
        df.PetalWidth = df.PetalWidth.astype(float)

        # Sets label name as Y
        df = df.rename(columns={'Class': 'Y'})

        # Reset index
        # df.reset_index()
        # df = df.drop(columns=['index'])

    elif data_name == 'adult':

        train_url = uci_base_url + 'adult/adult.data'
        test_url = uci_base_url + 'adult/adult.test'

        data_train = pd.read_csv(train_url, header=None)
        data_test = pd.read_csv(test_url, skiprows=1, header=None)

        df = pd.concat((data_train, data_test), axis=0)

        # Column names
        df.columns = ['Age', 'WorkClass', 'fnlwgt', 'Education', 'EducationNum',
                      'MaritalStatus', 'Occupation', 'Relationship', 'Race',
                      'Gender', 'CapitalGain', 'CapitalLoss', 'HoursPerWeek',
                      'NativeCountry', 'Income']

        # Creates binary labels
        df['Income'] = df['Income'].map({' <=50K': 0, ' >50K': 1,
                                         ' <=50K.': 0, ' >50K.': 1})

        # Changes string to float
        df.Age = df.Age.astype(float)
        df.fnlwgt = df.fnlwgt.astype(float)
        df.EducationNum = df.EducationNum.astype(float)
        df.EducationNum = df.EducationNum.astype(float)
        df.CapitalGain = df.CapitalGain.astype(float)
        df.CapitalLoss = df.CapitalLoss.astype(float)

        # One-hot encoding
        df = pd.get_dummies(df, columns=['WorkClass', 'Education', 'MaritalStatus',
                                         'Occupation', 'Relationship',
                                         'Race', 'Gender', 'NativeCountry'])

        # Sets label name as Y
        df = df.rename(columns={'Income': 'Y'})
        df['Y'] = df['Y'].astype(int)

        # Resets index
        df = df.reset_index()
        df = df.drop(columns=['index'])

        # Splits train, test and test sets
        train_idx = range(len(data_train))
        train = df.loc[train_idx]

        test_idx = range(len(data_train), len(df))
        test = df.loc[test_idx]

        train_idx_final = np.random.permutation(len(train))[:dict_no['train']]

        temp_idx = np.random.permutation(len(test))
        test_idx_final = temp_idx[:dict_no['test']] + len(data_train)
        # test_idx_final = temp_idx[dict_no['test']:] + len(data_train)

        train = train.loc[train_idx_final]
        test = test.loc[test_idx_final]
        # test = test.loc[test_idx_final]

        train.to_csv('./data_files/' + str(train_file_name), index=False)
        test.to_csv('./data_files/' + str(test_file_name), index=False)

        return
    else:
        raise ValueError('unsupported dataset')

    if t2classification:
        # Make 2 class last
        classes = list(set(df.Y))
        if len(classes) > 2:
            for index, row in df.iterrows():
                if row['Y'] != classes[0] and row['Y'] != classes[1]:
                    df.at[index, 'Y'] = classes[1]

    # Sample training data and the testing data
    train_idx_final = dict_no['train']
    test_idx_final = dict_no['test']

    if idx_sorted:
        train_idx_final.sort()
        test_idx_final.sort()

    train = df.iloc[train_idx_final]
    test = df.iloc[test_idx_final]

    # Saves data
    if not os.path.exists('data_files'):
        os.makedirs('data_files')

    train.to_csv('./data_files/' + str(train_file_name), index=False)
    test.to_csv('./data_files/' + str(test_file_name), index=False)


def load_augmented_data(data_size_dict, data_name, clean_file=True):
    augment_flag = False
    new_data_size_dict = None

    if data_name == 'adult':
        # Need to be augmented
        augment_flag = True
        new_data_size_dict = data_size_dict \
            if data_size_dict['train'] <= 32561 - data_size_dict['test'] \
            else {'train': 32561 - data_size_dict['test'], 'test': data_size_dict['test']}

        load_tabular_data(data_name, new_data_size_dict, 'train_data_augmented.csv',
                          'test_data_augmented.csv', idx_sorted=True)

        if augment_flag:
            train_data = pd.read_csv('./data_files/' + 'train_data_augmented.csv')
            # Augment the data to match the required size
            cols = ['Age', 'fnlwgt', 'EducationNum']
            append_num = data_size_dict['train'] - new_data_size_dict['train']

            append_df = pd.DataFrame()
            while True:
                for i, row in train_data.iterrows():
                    for col in cols:
                        row[col] += np.random.randint(1, 3) * 1.0
                    append_num = append_num - 1
                    append_df = append_df.append(row)
                    if append_num == 0:
                        break
                if append_num == 0:
                    break

            new_train_data = pd.concat((train_data, append_df))
            os.remove('./data_files/' + 'train_data_augmented.csv')
            new_train_data.to_csv('./data_files/' + 'train_data_augmented.csv', index=False)

    else:
        raise ValueError('dataset %s is not supported yet' % data_name)

    if clean_file:
        os.remove('./data_files/' + 'train_data_augmented.csv')
        os.remove('./data_files/' + 'test_data_augmented.csv')


def preprocess_data(train_file_name, test_file_name):
    """Loads datasets
    Args:
      train_file_name: file name of training set
      test_file_name: file name of testing set
    Returns:
      x_train: training features
      y_train: training labels
      x_test: testing features
      y_test: testing labels
      columns_name
    """

    train_df = pd.read_csv('./data_files/' + train_file_name)
    test_df = pd.read_csv('./data_files/' + test_file_name)

    columns_name = train_df.columns

    x_train = train_df.drop(columns=['Y']).values
    x_test = test_df.drop(columns=['Y']).values
    y_train = train_df.Y.values
    y_test = test_df.Y.values

    return x_train, y_train, x_test, y_test, columns_name


def variance(list1, list2):
    """
    compare the variance (list2 to list1)
    """
    try:
        if round(float(np.sum(list1)), 10) != round(float(np.sum(list2)), 10):
            raise Exception('[*] Variance is invalid with different means!\n'
                            '[*] Info: mean1 = %f mean2 = %f'
                            % (float(np.mean(list1)), float(np.mean(list2))))
    except Exception as err:
        print(err)
    else:
        return ((np.copy(list2) - np.copy(list1)) ** 2).sum()


def normalize(list1, list2):
    """
    normalize list2 to list1
    """
    coef = np.sum(list1) / np.sum(list2)
    return coef * list2


def save_df(df, file_name):
    if isinstance(df, pd.DataFrame):
        df.to_csv(file_name)


def save_npy(file_name, arr):
    check_folder('res')
    if isinstance(arr, np.ndarray):
        np.save(str(Path.cwd().joinpath('res').joinpath(file_name)), arr)


def load_npy(file_name):
    check_folder('res')
    arr = np.load(str(Path.cwd().joinpath('res').joinpath(file_name)))
    if isinstance(arr, np.ndarray):
        return arr


def check_folder(folder_name):
    if not Path(folder_name).exists():
        prefix = Path.cwd()
        Path.mkdir(prefix.joinpath(folder_name))


def comp(base_v, comp_sv, name):
    var = variance(base_v, normalize(base_v, comp_sv))
    print("variance of %s   \t : %.10f" % (name, var))
