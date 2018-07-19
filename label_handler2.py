import pandas as pd
import numpy as np
import os
import pendulum
import random
import yaml

class label_handler():
    def __init__(self, config_file):
        self.config_data = yaml.load(open(config_file))
        self.last_update = pendulum.datetime(1879, 3, 14)
        self.last_filename = dict()

    def start_process(self):
        self.classes = np.array(self.config_data['classes'])
        self.data_path = self.config_data['data_path'][0]
        self.label_path = self.config_data['label_path'][0]
        self.threshold = self.config_data['threshold'][0]

        if os.path.isfile(self.label_path):
            print('A labelfile already exist. It will be loaded and appended.')
            self.df = pd.read_pickle(self.label_path)
        else:
            print('A new label file will be created.')
            self.df = pd.DataFrame(
                    columns=['filename', *self.classes, 'final_label'])

    def update_filename(self):
        ''' Load filenames of pictures and check if they in the 
        DataFrame. If they doesn't exist, they will be added.
        '''
        dff = pd.DataFrame({'filename': os.listdir(self.data_path)})
        self.df = self.df.merge(dff, on='filename', how='outer',
                validate='one_to_one')
        self.df[self.classes] = self.df[self.classes].fillna(0)
        self.last_update = pendulum.now()

    def get_pic_to_label(self, user):
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

    def set_label(self, label, user):
        filename = self.last_filename[user]
        mask = filename == self.df['filename']
        self.df.loc[mask,label] += 1
        self.df.to_pickle(self.label_path)

    def get_final_label(self):
        final_label = self.df[self.classes].values.argmax(axis=1)
        label =self.classes[final_label.transpose()]
        self.df['final_label'] = label
    
    def check_predictions(self):
        self.classified_label = self.config_data['classified_label'][0]
        self.df_classified = pd.read_pickle(self.classified_label)
        self.df_classified['status'] = 'not checked'
        print(self.df_classified)
    
    def get_pic_to_check(self, user):
        mask = self.df_classified['status'] == 'not checked'
        if sum(mask) == 0:
            print('Everything checked, be happy that work is'
                    'done or add more data')
        self.last_filename[user] = np.random.choice(self.df.filename[mask])
        return self.data_path + self.last_filename[user]

    def check_pic(self, user, label):
        pos = self.df.filename == self.last_filename[user]
        if self.df[pos].final_label.values != label:
            print('Label denied ', self.last_filename[user])
            self.df.loc[pos,self.classes] = 0
        else:
            print('Label accepted', self.last_filename[user])


def main():
    handler = label_handler('./config.yml')
    handler.start_process()
    pic = handler.get_pic_to_label('maximilian')
    handler.set_label('altocumulus', 'maximilian')
    handler.get_final_label()
    handler.check_predictions()
    handler.get_pic_to_check('maximilian')
    handler.check_pic('maximilian','altocumulus')


if __name__ == '__main__':
    main()
