%% Code to import training data from MNIST dataset

clear all;

train_images = reshape((loadMNISTImages('train-images.idx3-ubyte')),[28,28,1,60000]);
train_labels = ((loadMNISTLabels('train-labels.idx1-ubyte')));


test_images = reshape((loadMNISTImages('t10k-images.idx3-ubyte')),[28,28,1,10000]);
test_labels = ((loadMNISTLabels('t10k-labels.idx1-ubyte')));


%% Code to generate 10 fold train data

train_images_fold=zeros(28,28,1,6000,10);
train_labels_fold=zeros(6000,1,10);

for i=(1:10)
train_images_fold(:,:,:,:,i)=train_images(:,:,:,(6000*(i-1))+1:6000*i);
train_labels_fold(:,1,i)=train_labels((6000*(i-1))+1:6000*i,1);
end

train_images_data=zeros(28,28,1,54000,10);
train_labels_data=zeros(54000,1,10);
cross_valid_images=zeros(28,28,1,6000,10);
cross_valid_labels=zeros(6000,1,10);

for (i=1:10)
    cross_valid_images(:,:,:,:,i)=train_images_fold(:,:,:,:,i);
    cross_valid_labels(:,:,i)=train_labels_fold(:,:,i);
    k=1;
    for j=(1:10)
        if (j~=i)
           train_images_data (:,:,:,(6000*(k-1))+1:6000*k,i)=train_images_fold(:,:,:,:,j);
           train_labels_data ((6000*(k-1))+1:6000*k,1,i)=train_labels_fold(:,:,j);
           k=k+1;
        end
    end
end


%% Code to display first 100 training images
figure(1)
for i=1:100 
        subplot(10,10,i);
        imshow(train_images(:,:,:,i));
end

%% Code to generate the ANN layers

inputlayer = imageInputLayer([28 28 1],'DataAugmentation','none',...
    'Normalization','none','Name','input');

%%volume 28*28*1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                Layer 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

convlayer1 = convolution2dLayer(4,32,'Stride',1,'Padding',0, ...
    'BiasLearnRateFactor',2,'NumChannels',1,...
    'WeightLearnRateFactor',2, 'WeightL2Factor',1,...
    'BiasL2Factor',1,'Name','conv1');

%%volume 25*25*32

convlayer1.Weights = randn([4 4 1 32])*0.1;
convlayer1.Bias = randn([1 1 32])*0.1;

relulayer1 = reluLayer('Name','relu1');

localnormlayer1 = crossChannelNormalizationLayer(3,'Name',...
    'localnorm1','Alpha',0.0001,'Beta',0.75,'K',2);

maxpoollayer1 = maxPooling2dLayer(3,'Stride',3,'Name','maxpool1','Padding',1);

%volume 9*9*32 

droplayer1 = dropoutLayer(0.35);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                  Layer 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

convlayer2 = convolution2dLayer(3,16,'Stride',1, 'Padding',0,...
    'BiasLearnRateFactor',1,'NumChannels',32,...
    'WeightLearnRateFactor',1, 'WeightL2Factor',1,...
    'BiasL2Factor',1,'Name','conv2');

%7*7*16

convlayer2.Weights = randn([3 3 32 16])*0.0001;
convlayer2.Bias = randn([1 1 16])*0.00001;

relulayer2 = reluLayer('Name','relu2');

localnormlayer2 = crossChannelNormalizationLayer(3,'Name',...
    'localnorm2','Alpha',0.0001,'Beta',0.75,'K',2);

%maxpoollayer2 = maxPooling2dLayer(2,'Stride',2,'Name','maxpool1','Padding',1);
    
droplayer2 = dropoutLayer(0.25);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                  Output Layers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fullconnectlayer = fullyConnectedLayer(10,'WeightLearnRateFactor',1,...
    'BiasLearnRateFactor',1,'WeightL2Factor',1,'BiasL2Factor',1,...
    'Name','fullconnect1');
fullconnectlayer.Weights = randn([10 784])*0.0001;
fullconnectlayer.Bias = randn([10 1])*0.0001+1;

smlayer = softmaxLayer('Name','sml1');

coutputlayer = classificationLayer('Name','coutput');


%% Code to define training parameters
    
options = trainingOptions('sgdm',...
      'LearnRateSchedule','piecewise',...
      'LearnRateDropFactor',0.75,... 
      'LearnRateDropPeriod',1,'L2Regularization',0.0001,... 
      'MaxEpochs',16,'Momentum',0.9,'Shuffle','once',... 
      'MiniBatchSize',15,'Verbose',1,...
      'CheckpointPath','E:\ccfuser3\checkpoints','InitialLearnRate',0.043);
  
 %% Code to make the network
    layers =[inputlayer, convlayer1, relulayer1,localnormlayer1, ...
       maxpoollayer1, droplayer1,...
       convlayer2, relulayer2, localnormlayer2,droplayer2,...
       fullconnectlayer, smlayer, coutputlayer]; 
   
   
   %% Train the ANN
   train_im=zeros(28,28,1,54000);
   train_lb=categorical(zeros(54000,1));
   cross_valid_im=zeros(28,28,1,6000);
   cross_valid_lb=zeros(6000,1);
   for (i=1:10)
       train_im=train_images_data(:,:,:,:,i);
       train_lb=categorical(train_labels_data(:,:,i));
       cross_valid_im=cross_valid_images(:,:,:,:,i);
       cross_valid_lb=categorical(cross_valid_labels(:,:,i));
trainedNet(i) = trainNetwork(train_im,train_lb,layers,options);

[Ypred,scores] = classify(trainedNet(i),cross_valid_im);
score(i) = sum((Ypred==cross_valid_lb))/numel(cross_valid_lb)
   end
   
   [max,index]=max(score);
   best_model=trainedNet(index); 