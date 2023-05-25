# deep-learning-challenge

## Overview:
The nonprofit foundation Alphabet Soup distributes funds to help support work done by other charitable organizations. Each of these organizations submits an application to Alphabet Soup, who then determines which applicants for funding have the best chance of success in their ventures and grant funding to those organizations. The business team had provided data for over 34,000 organizations which have already received funding from Alphabet Soup and whether or not their venture was successful. We will use this data to create a neural network model to help predict which applicants are likely to be successful if funded by Alphabet Soup.

## Results:
* **Data Preprocessing**:
Much of the data included was in a categorical format, this required grouping rare categories into an ‘other’ category before encoding these variables with `pd.get_dummies()` and applying `StandardScaler()`.
* **Targeted Variables**:
This model aims to predict success, the binary success column served as the target variable
* **Features**:
The features for this model were the descriptive and informational columns including application type, industry affiliation, government classification, use case, organization type, active status, income amount, funding amount requested, success, and were there any special considerations for the application.
* **Excluded Variables**: All identifying information was excluded from the dataset such as organization names and EINs.

### Compiling, Training, and Evaluating the Model

My initial model included two hidden layers which each had nine neurons and utilized the ReLU activation function which I ran for 100 epochs to get a baseline for the neural network. This resulted in an accuracy score of `0.7245`. During this I found that the accuracy was no longer improving after 10-15 epochs.

<img width="702" alt="Screen Shot 2023-05-24 at 8 35 35 PM" src="https://github.com/leah-apking/deep-learning-challenge/assets/119013360/93d62f9d-f4e0-4015-9c86-c1462cdc308c">

To optimize this model, I ran a Teras Tuner optimization function to cycle through a variety of combinations of hidden layers, neuron counts, and activation functions each of which was tested at 20 epochs. This tuner selected a four-layer neural network utilizing the tanh activation function which produced an accuracy score of `0.7279`, nearly the same as my initial model.

<img width="748" alt="Screen Shot 2023-05-24 at 8 38 01 PM" src="https://github.com/leah-apking/deep-learning-challenge/assets/119013360/43153a12-763a-4b16-a64e-447a22fd35f3">

I then tried to combine these models by using both the ReLU and tanh activation functions with additional hidden layers. Unfortunately, these also resulted in accuracy score around 0.72. Finally, tried eliminating what appeared to be extraneous features such as application type, status, special considerations, and affiliations. This improved the accuracy score slightly to `0.7334`, but still fell short of the 0.75 goal.

## Summary: 
Using a neural network deep learning model produced a maximum accuracy score of `0.727` when utilizing all the data provided, shy of the 0.75 goal. This was acheived even when utilizing the Teras Tuner to test a variety of different combinations, although it was not significantly better than my initial guess. Due to the limited effect of changing activation functions, additional hidden layers, and additional neurons in each layer as well as the network reaching its maximum accuracy with relatively few epochs, I believe a different method of machine learning may be a better fit for this dataset.

A supervised learning model may be a good fit for this dataset such as a random forest or logistic regression. This would allow us to better understand where the model is struggling by producing a confusion maxtrix and classification report. When using a random forest we can use the `rf_model.feature_importances_` to better understand which features are having the biggest impact on success rate and use that insight make additional modifications. If the model is struggling due to a lack of unsuccessful cases to train on, a `RandomOverSampler` can be used. One of the big draw backs of using a deep learning neural network like we used here, is that they are difficult to understand. Without an understanding of where/why the model is stuggling it is hard to make adjustments to improve the accuracy of the model. No matter what model is used I think it would also be helpful to check for and eliminate outliers which can disrupt the training of machine learning models.
