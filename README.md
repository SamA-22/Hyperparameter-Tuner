# Hyperparameter-Tuner

Machine learning code in relation to stocks. Tunes hyperparameters and outputs optimum parameters.

## Latest version/ Added features

- Version 1.0.0
  - Retrieving and formatting historical data using pandas_datareader
  - Creating neural network models using Tensorflow and Keras
  - Simple brute force looping to test hyperparameters
  - Percentage error calculation to determine best hyperparameters

## Key Information

### Historical Data

- Historical data is retrived from [Yahoo Finance](https://uk.finance.yahoo.com/)

### Key Global Variables

- These variables will have to be manually set, default values are given. (future updates may change this)
  - ticker: Stock ticker symbol will need to be given.
  - predictionDays: default value 7. This is a hyperparameter that may be tweaked in future updates.
  - Train Data Dates (default values start: 2012, 1, 1, end: 2021, 4, 1) Test Data Dates (default values start: 2021, 4, 2,     end: 2022, 4, 1).
  - maxLayers, maxUnits, maxEpochs, maxBatchSize. Hyperparameters that will need to be manualy set recomended vaules are       given in code (recomended values liable to change if the train and test data dates are changed.
  - loopAmount: training a model even with the same data and hyperparams the weights could vairy sometimes leading to           anomilous results or incorrect results and so this allows for hyperparams to be tested multiple times to truley get the       best hyperparameters.

- These variables will be manually set and changes arnt recommended. (future updates may change this)
  - featureNo: The amount of features that are used from Yahoo Finance. If changes are made to the defult value, 4, the get     and format data function will need to be modified. 

### Whilst running

- Code may take awhile to run depending on the max hyperprameters that are given. Changing the verbose in the createModel function (line 121) to 1-2 will output a graphic that will show if the code is still running.

## Potential Updates

- [] Save and Load function. As the code can take awhile to run, the current hyperparameters being used can be automatically saved as well as other data which can then be picked back up when the code is once again loaded.
- [] When perfect hyperparams are found train the model till weights that predict most accurately are saved.
- [] List holding multiple tickers that will run one after the next.







