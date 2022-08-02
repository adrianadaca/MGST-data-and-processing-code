
The raw data and processing code here were used to generate plots and analysis in the publication 'Expansion and experimental evaluation of scaling relations for the prediction of wheel performance in reduced gravity' (submitted to Microgravity Science and Technology). 

### Requirements
Create an environment with python 3.7
```
conda create -n myenv python=3.7
```
Install dependencies
```
pip install -r requirements.txt
```
```
conda install -c conda-forge firefox geckodriver
```

### Instructions
Run in the following order
1. process_flight_data.py
2. process_gsl_data.py
3. scale_data.py
4. compute_error.py
