This is the official implementation of the fair bundle recommendation algorithms from _Producer-Fairness in Sequential Bundle Recommendation_, currently under review at KDD 2026.

# Repository description

- ```main.py```: implements the main algorithm for fair sequential bundle recommendation. The ```run(args)``` function
takes in as input a series of arguments corresponding to the problem and algorithm parameters. Problem parameters are
defined into a ```problem_config.json``` file.
- ```exps.py```: allows to reproduce the fairness-quality trade-off figures for each individual method for the problem 
of your choice (by default, _FairWG_ on _MovieLens_). Also allows to easily reproduce most figures of the paper by changing the method
and the parameters to vary.
- ```algos```: contains the three algorithms from the paper in their one-user version (as ```main.py```takes care of
the sequential aspect). These are _ILP_ (```ilp.py```), FairWG (```fair_wg.py```) and F3R (```f3r.py```).
- ```config```: contains several JSON config files corresponding to several problems studied in the paper's experiments.
- ```data```: contains the data for the three tasks used in the experimental part of the paper (```ml-100k```, 
```yelp``` and ```amazon```).
  - For _Amazon_, the relevance matrix is too big, hence we provide the raw ratings (```rating_proc.csv```) as well as
  the file ```proc_ratings.py``` to process them and create the matrix file (```.npy``` format). 

# Install requirements
Please first install requirements using:
````
pip install -r requirements.txt
````

# Run an experiment 
To run a particular experiment, run the following command line:
````
python main.py --problem_config ml-100k
````
Specify other arguments as wanted.

# Results
Results will be saved in the ```runs``` folder. In particular, for each run, a tensorboard file is created containing many
metrics, as well as ```bundle_history.csv``` file containing the most important results.
