import pandas as pd
import numpy as np
import os
import pendulum
import random
import yaml

class label_handler():
    def __init__(self, config_file):
        ''' Create a label handler to label pictures of given Path.
        Create Parameters last_update and last_filename for further 
        functions.
        Parameters:
            path_data: string 
                Path of the pictures.
            path_label: string
                Path of existing or to create pickle file with dataframe
        '''
        self.config_data = yaml.load(open(config_file))
        self.last_update = pendulum.datetime(1879, 3, 14)
        self.last_filename = dict()

    def start_process(self):
        self.classes = self.config_data['classes']
        self.data_path = self.config_data['data_path'][0]
        self.label_path = self.config_data['label_path'][0]
        self.threshold = self.config_data['threshold'][0]

        if os.path.isfile(self.label_path):
            print('A labelfile already exist. It will be loaded and appended.')
            self.df = pd.read_pickle(self.label_path)
        else:
            print('A new label file will be created.')
            self.df = pd.DataFrame(
                    columns=['filename', *self.classes, 'finallabel'])

    def update_filename(self):
        ''' Load filenames of pictures and check if they in the 
        DataFrame. If they doesn't exist, they will be added.
        '''
        dff = pd.DataFrame({'filename': os.listdir(self.data_path)})
        self.df = self.df.merge(dff, on='filename', how='outer',
                validate='one_to_one')
        self.df[self.classes] = self.df[self.classes].fillna(0)
        self.last_update = pendulum.now()

    def get_pic(self, user):
        ''' Load a random not labeld picture file.
        Returns:
            path: string 
                string of filename of not labeld picture
        '''
        if(pendulum.now().subtract(seconds=1) > self.last_update):
            self.update_filename()
        mask = self.df.sum(axis=1) < self.threshold
        if sum(mask) == 0:
            print('Everything labeled, be happy that work is'
                    'done or add more data')
        self.last_filename[user] = np.random.choice(self.df.filename[mask])
        return self.data_path + self.last_filename[user]

    def last_pic(self, user):
        ''' Returns last load picture to catch if a user want 
        further information to a cloud. After that he can return 
        to the pic he see before.
        Returns:
            path: string
                string of filename of last picture
        '''
        return self.data_path + self.last_filename[user]

    def set_label(self, label, user):
        filename = self.last_filename[user]
        mask = filename == self.df['filename']
        self.df.loc[mask,label] += 1
        self.df.to_pickle(self.label_path)


def main():
    handler = label_handler('./config.yml')
    handler.start_process()
    for x in range(1000):
        pic = handler.get_pic('maximilian')
        handler.set_label('altocumulus', 'maximilian')
    for x in range(1000):
        pic = handler.get_pic('maximilian')
        handler.set_label('cirrus', 'maximilian')

if __name__ == '__main__':
    main()
