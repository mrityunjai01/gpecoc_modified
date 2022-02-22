# STMPyramid
## Running the code
Simply change constants in constants.py.


```python main.py``` is enough to run the code after that.

#### Note:
Please ensure that these libraries have been downloaded and installed before running the code.
- numpy
- pandas
- scipy
- sklearn
- cvxpy
- tensorly
- torchvision


### Request:
- In case you encounter any bugs, please bring it to my notice immediately.
- If you have any suggestions for features to be added, let me know as well.

### Known Bugs:
1. SHTM giving error.

### TODO:
1. More datasets in loader
2. Better cross validation / grid search pipeline with more information
3. In case any grid search hyperparameter gives error, please handle that with 'try except block'
4. Saving the images and data to another folder (better visualisation needs to be added especially for RGB images)
5. More solvers (and adding the hyperparams in the pipeline) (Logistic regressor needs to be added at the least)
6. Dimension reduction for all data happens earlier and not at every iteration. (kernel still needs to be made at every iteration though)
