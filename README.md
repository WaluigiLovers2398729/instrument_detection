# Simple Python Project Template

The basics of creating an installable Python package.

To install this package, in the same directory as `setup.py` run the command:

```shell
pip install -e .
```

This will install `example_project` in your Python environment. You can now use it as:

```python
from example_project import returns_one
from example_project.functions_a import hello_world
from example_project.functions_b import multiply_and_sum
```

To change then name of the project, do the following:
   - change the name of the directory `example_project/` to your project's name (it must be a valid python variable name, e.g. no spaces allowed)
   - change the `PROJECT_NAME` in `setup.py` to the same name
   - install this new package (`pip install -e .`)

If you changed the name to, say, `my_proj`, then the usage will be:

```python
from my_proj import returns_one
from my_proj.functions_a import hello_world
from my_proj.functions_b import multiply_and_sum
```

You can read more about the basics of creating a Python package [here](https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/Modules_and_Packages.html).


# Planning

### data.py
   1. [Coded, Not Tested] Function to use Mic to record data in form of NumPy array (Day 1, WorkingWithMic)
      * Takes in listenduration
      * Uses microphone record_audio with listenduration to get frames, samplingrate
      * Uses np.hstack on frames to join frames in a single array of digital samples
      * Calculates respective times
      * Return digital samples and times
   2. [Coded, Not Tested] Function to convert any audio file to NumPy-array of digital samples (Day 1, AnalogToDigital)
      * Takes in file name and clipduration
      * Pathlib uses file name to find file path, stored as string
      * Uses librosa.load with filepath/samplerate/mono/duration to get samples, samplingrate
      * Calculates respective times
      * Returns digital samples and times

### get_spectrogram.py
   1. [Coded, Not Tested] Function to make Spectrogram (Day 3 Notebook: Spectrogram + Day
    2 Notebook: DFT)
      * takes in digital samples
      * does fourier transforms
      * gets spectogram np array

### model.py
   1. [NOT STARTED] init() 
      * Set up all the layers
   2. [NOT STARTED] call()
      * Do the forward pass, so imitate the paper
   3. [NOT STARTED] parameters()
      * get parameters
   4. [NOT STARTED] loss_accuracy()
      * returns loss and accuracy
   5. [NOT STARTED] save_weights()
      * Saves the weights from the trained model
   6. [NOT STARTED] load_weights()
      * Loads the weights from the trained model

### training.py
   # train_data = {<spect_audio_file>:<classification>}
   # test_data = {<spect_audio_file>:<classification>}
   1. [NOT STARTED] train(train_data):
      trains model using train_data
      divides database into batches of random files
   2. [Coded, Not Tested] load_dictionary(file_path):
      loads a dictionary from a Pickle file
   3. [Coded, Not Tested] save_dictionary(dict, file_path):
      saves a dictionary to a Pickle file
   4. [NOT STARTED] populate_database(dict, file_path):
      use file_path as the path to the overall folder
      loop through each folder - get classification from folder names
      in each folder, loop through each audio file and turn audio into spectrogram np array
      add array, classification to dict
      return dictionary
   5. [Coded, Not Tested] initialize_database():
      take in a dictionary loaded from a file or create new dictionary
      