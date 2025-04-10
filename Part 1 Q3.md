# Evaluation Metrics
1. Accuracy - Basically the number of correcct predictions. So I would find the total number of correct predictions. Then divide it by the total number of predictions made.
2. Precision - Out of all the predictions the model made for a specific class, how many were actually correct.(false positives are costly)
3. Recall - Out of all the actual instances of a class, how many did the model correctly identify. (false negatives are costly)
4. F1 Score - Used to find a balance between precision and recall. The harmonic mean of Precision and Recall.

Formulas used for calculating each:
Suppose no_of_true_positives = TP, no_of_false_positives = FP, no_of_false_negatives = FN
1. Accuracy = no_of_correct_predictions / total_no_of_predictions
2. Precision = TP/(TP+FP)
3. Recall = TP/(TP+FN)
4. F1 Score = Precision*Recall/(Precision + Recall)

So we get class specific Precision, Recall and F1 Score. To consolidate it into just one value each there are many methods that can be used including... 
1. Macro Average - Good when all classes are equally important. (Add all Precision / Total no of classes)
2. Weighted Average - Takes the average of each class's metric, weighted by the number of samples in that class. (Add all no_of_sample_per_class*precision_of_class / total no of sample)
3. Micro Average - Good when you care more about overall performance, especially in imbalanced datasets.
But for my sample I have given quite balanced out data, so I'll be using the Macro Averages.



# Strategies for Improvement
One thing i definitely did to improve my model was using more data. Now my model works quite well. In terms of different using different models, I do believe that my Tranformer based Neural Network was the best option. Next I did try hyper parameter tuning (which is basically just finding the best set of parameters to optimize my model performance). I did this manually though. And these are the parameters I tried to change:
1. Learning rate = [0.0001,0.001]
2. No_of_layers = [6,8]
3. Dropout = [0.1, 0.2]
4. No_of_heads = [8,16]

I found that a learning rate of 0.0001 worked best... why? Because now the model doesn't go accidentaly skip the right balance in its "eagerness" to learn. Anyway I have given a total of 100 iterations so even a slow learning curve doesn't matter. This is like the saying "Hard work over talent 😜"

As for the number of layers... 6 worked best. This baffled me a lot because ideally more the number of layer better the model will learn right? Turns out that assumption is wrong. Why tho? So I could not come up with a good enough explanation for this and so *cough* *cough* consulted chatGPT. "Overfitting - More layers → More parameters → Higher capacity to "memorize" the training data. If your dataset is not big enough, the model doesn't generalize well and ends up doing worse on unseen (validation/test) data." This is what it gave and it does make sense. It relies too much on my data. 

Now regarding dropouts... Surrisingly(or maybe not) dropout of 0.2 worked best. Now this might be because my data is not very inclusive. So when I gave a 0.1 my model relying more on my data gave higher losses. While a dropout of 0.2 helped it learn meaningful patterns from my data while at the same time allowing for randomness. 

Ohk the final thing I tried is changing the number of heads. And this thankfully worked how I wanted it to. So yes! 16 gave better results though not noticably much after 100 iterations in the initial stages it was way better. Now why might this be. So for this a head must be thought of intuitively as representing a specific part of sentence identification. For instance, One head might focus on “what” is being asked, Another might focus on “how technical” the question is, Another might notice words like "React" or "CNN" and link them to their domains. Of course this is just an intuitive way of looking at thing. Heads don't actually understand things like humans 🤣. So basically splitting heads lets it look into different aspects. But increasing it has its drawbacks too... for instance more computation. And the fact that each head gets lesser dimensions so it might not get any meaningful information even though it covers many aspects.
