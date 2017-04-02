[Ypred,scores] = classify(best_model,test_images);
score = sum((Ypred==categorical(test_labels)))/numel(test_labels);

display('Test set accuracy =')
display(100*score)