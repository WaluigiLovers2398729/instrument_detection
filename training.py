from digital_sampling import *
from get_spectrogram import *
import os, sys
import pathlib as Path
import io
import pickle


def load_dictionary(file_path):
    """
    loads a dictionary from a Pickle file
    Parameters
    ----------
    file_path: string
        path and name of file
    
    Returns
    -------
    dictionary 
        unpickled dictionary
    Notes
    -----
    
    """
    with open(file_path, mode = "rb") as opened_file:
        return pickle.load(opened_file)
    

def save_dictionary(dict, file_path):
    """
    saves a dictionary to a Pickle file
    Parameters
    ----------
    dict: dictionary
        dictionary to pickle
    file_path: string
        path and name of file to store dictionary to 
    Returns
    -------
    None
    
    Notes
    -----
    
    """
    with open(file_path, mode = "wb") as opened_file:
        pickle.dump(dict, opened_file)

def initialize_database():
    """
    Initalizes a dictionary database 
    Parameters
    ----------
    None
    
    Returns
    ------
    database : dict
        initialized database
    """
    database = {}
    dictionary_type = int(input("Enter 0 to input a pickled dictionary, Enter 1 to have it initialized: "))
    # Pickled Dictionary
    if dictionary_type == 0:
        file_path = input("Enter the file path and file name to the dictionary: ")
        database = load_dictionary(file_path)
    # We initialized 
    elif dictionary_type == 1:
        pass
    # Invalid Option
    else:
        print("Error: Invalid Option") 
    return database

def populate_database(dict, file_path):
    """
    Populate a dictionary database 
    Parameters
    ----------
    dict: dictionary
        dictionary containing database of either training or testing data
    
    file_path: string
        path to folder of subfolders with audio files
    
    Returns
    ------
    database : dict
        populated database
    """
    for subdir, dirs, files in os.walk(file_path):
        inst = os.path.basename(os.path.normpath(subdir))
        for file in files:
            p = os.path.join(subdir, file)
            digital_samples, times = filesample(p)
            spectrogram = make_spectogram(digital_samples, times)
            dict[inst] = spectrogram

    return dict     