import random,copy,math,time,os,csv,sys
import scipy.io as sio # for loading .mat files


def get_data(data_directory):
    data = []
    for filename in os.listdir(data_directory):
            if filename.endswith(".mat"):
                experiment = filename[:-4]
                print(experiment)
                data_dict = sio.loadmat(data_directory+filename)

                if 't' in data_dict:
                    data_dict["experiment"] = experiment
                    if experiment == 'NoGuidedRNA':
                        data_dict["experiment"] = 'Control'
                    if '-' in experiment:
                        defect = experiment.split('-')[0]
                        if defect in 'NonRepeatedSequence':
                            data_dict['defect'] = 'NRS'
                        else:
                            data_dict['defect'] = experiment.split('-')[0]
                        data_dict['nuclease'] = experiment.split('-')[1]


                    if 'AllCells' in data_dict:
                         data_dict['AllCellsBF'] = data_dict['AllCells']
                    if 'AllGFPCells' in data_dict:
                         data_dict['AllCellsGFP'] = data_dict['AllGFPCells']
                    data.append(data_dict)
                    print('  success!')
                else:
                    print('  missing data!')
    return data
