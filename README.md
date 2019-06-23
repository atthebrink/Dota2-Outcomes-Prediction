# Dota2-Outcomes-Prediction

## What is this project?
This is the final assignment of Neural Network Theory and Applications.

But I also made this project out of my interests in Dota2.

## Introduction of the project
Detailed description is included in the [technical report](https://github.com/atthebrink/Dota2-Outcomes-Prediction/blob/master/Using%20LSTM%20and%20AAE%20to%20Predict%20Dota2%20Outcomes%20by%20analyzing%20the%20Draft.pdf).

This project is to predict the outcomes of a Dota2 game based on drafts. 

Unfortunaterly this task is very hard that most of the previous work reported accuracy less than 60%.

I used the dataset at version 7.06e and it is collected by Andrei Apostoae in 2017. The dataset is availabel at
https://github.com/andreiapostoae/dota2-predictor. And Andrei also reported to have less tan 60% accuracy on traditional methods.

I believe this is because the datasets are only ovservations of outcomes, thus they cannot truly reflect the winning chance of of a team.
Still we cannot use frequences to estimate probabilities because I actually found no same draft in the entrire datasets.

Althogh I tried to use more complex neural network models and tried to filter the dataset to get observations 
that reflect win probablities better, I only get a slightly better accuray of 61.3% and the AUC score is 0.651 which is almost the same
as reported in Andrei's work.

It seems that there is still a long way to reliably predict the outcomes of Dota2 games only based on drafts.

## How to use this project
This is a python project and the dependencies are as follows:
1. keras
2. numpy
3. matplotlib
4. sklearn
5. pandas 
6. tensorflow 
