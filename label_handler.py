import pandas as pd
import numpy as np
import os
import pendulum
import random

class label_handler():
    def __init__(self, path_data, path_label='./label.pkl'):
        ''' Create a label handler to label pictures of given Path.
        Create Parameters last_update and last_filename for further 
        functions.
        Parameters:
            path_data: string 
                Path of the pictures.
            path_label: string
                Path of existing or to create pickle file with dataframe
        '''
        self.path_label = path_label
        self.path_data = path_data
        self.last_update = pendulum.datetime(1879, 3, 14)
        self.last_filename = None

    def load_data(self):
        ''' Load DataFrame from existing pickelfile or create a new 
        DataFrame. 
        '''
        if os.path.exists(self.path_label):
            self.df = pd.read_pickle(self.path_label)
        else:
            self.df = pd.DataFrame({
                'filename': [],
                'label1': [],
                'label2': [],
                'label3': [],
                'final_label': []
                })

    def save_data(self):
        ''' Save actuell DataFrame.
        '''
        self.df.to_pickle(self.path_label)

    def update(self):
        ''' Load filenames of pictures and check if they in the 
        DataFrame. If they doesn't exist, they will be added.
        '''
        self.load_data()
        dff = pd.DataFrame({'filename': os.listdir(self.path_data)})
        self.df = self.df.merge(dff, on='filename', how='outer',
                validate='one_to_one')
        self.last_update = pendulum.now()

    def get_pic(self):
        ''' Load a random not labeld picture file.
        Returns:
            path: string 
                string of filename of not labeld picture
        '''
        if(pendulum.now().subtract(hours=1) > self.last_update):
            self.update()
        mask = pd.isna(self.df.final_label)
        self.last_filename = np.random.choice(self.df.filename[mask])
        return self.path_data + self.last_filename

    def last_pic(self):
        ''' Returns last load picture to catch if a user want 
        further information to a cloud. After that he can return 
        to the pic he see before.
        Returns:
            path: string
                string of filename of last picture
        '''
        return self.path_data + self.last_filename

    def set_label(self, label):
        ''' Label Picture and set final label if 3 times labeled.
        Saved the modified DataFrame afterwards.
        Parameters:
            label: string
                label of the cloud type
        '''
        liste = []
        if(pd.isna(self.df.loc[self.df['filename'] == self.last_filename,
            'label1']).all()):
            self.df.loc[self.df['filename'] == self.last_filename, 'label1'] = label
        elif(pd.isna(self.df.loc[self.df['filename'] == self.last_filename,
            'label2']).all()):
            self.df.loc[self.df['filename'] == self.last_filename, 'label2'] = label
        else:
            self.df.loc[self.df['filename'] == self.last_filename, 'label3'] = label
            liste = [*self.df.loc[self.df['filename'] == self.last_filename,
                'label1'].values,
                    *self.df.loc[self.df['filename'] == self.last_filename,
                'label2'].values,
                    label]
            self.df.loc[self.df['filename'] == self.last_filename, 'final_label'
                    ] = max(set(liste), key=liste.count)
        self.save_data()

if __name__ == '__main__':
    hand = label_handler('/home/maximilian/pictures/')
    try:
        path= hand.get_pic()
        print('Path: ', path)
        hand.set_label('BVB')
    except ValueError:
        print('Everything labeled')
