#Neural Network that sums the input and provides a single output - linearly seperable


inputs<-c();dataset<-c();output<-c() #initialise empty vectors

#create a randomly generated matrix of 10 units of 5000 inputs
n<-5000
for (j in 1:n){
  for (i in 1:10){
    input<-ceiling(runif(1,-1,1))
    if (input==0){
      inputs[i]<--1
    }else{
      inputs[i]<-1
    }
  }
  dataset<-rbind(dataset,inputs)
}
dataset<-unique(dataset) #select only the unique randomly generated values

#create output vector depending on sum of the input vector units
for (row in 1:nrow(dataset)){
  out<-sum(dataset[row,])
  if (out<0){
    output[row]<--1 #if the sum is -ve, output is -1
  }else{
    output[row]<-1 #if the sum is 0 or position, output is +1
  }
}

dataset<-cbind(dataset,output) #concatenate inputs and outputs into a dataset

#create training, validation and testing datasets
train<-dataset[1:(round((4*nrow(dataset))/5)),] #80%
valid<-dataset[(nrow(train)+1):(nrow(dataset)-round(nrow(dataset)/10)),] #10%
test<-dataset[((nrow(dataset)-round(nrow(dataset)/10))+1):nrow(dataset),]#10%

ninputs = nrow(train) #number of inputs in training set
targets = train[,ncol(train)] #the specified targets
inputs = cbind(as.matrix(train[,1:10]), rep(-1,ninputs)) #10-unit inputs & bias
epoch_numbers<-seq(1,20,1) #the number of epochs to train under
epsilon = 0.01 #learning rate
wts = runif(11,0,10) #randomise weights for inputs and bias

#store adjusted weights for every epoch in a matrix
epoch_weights<-matrix(0,nrow=length(epoch_numbers), ncol=length(wts))
error_vec<-c() #initialise error vector
train_accuracy<-c()#initialise accuracy vector
wrong<-c(0); mae_train<-c(); rmse_train<-c()

#The perceptron
for (iteration in epoch_numbers) {
  order = sample(ninputs);
  wrong<-c(0) #vector for misclassified outputs
  error = 0;
  for (i in order) {
    x = inputs[i,]
    t = targets[i]
    y = sum ( x * wts )
    y=tanh(30*(y))
    error = error + (0.5*(t - y)^2);
    dw = epsilon * ( t - y ) * x
    wts = wts + dw;
    mae<-abs(t-y) #mean absolute error(calculated later)
    mae_train<-append(mae_train,mae)
    rmse_train<-append(rmse_train,mae^2)
    if (t!=y){
      misclassified<-1
      wrong<-append(wrong,misclassified)
    }
  }
  train_accuracy[iteration]<-(1-(sum(wrong)/ninputs))*100
  error_vec[iteration]<-error
  epoch_weights[iteration,]<-wts
  if (error == 0)
    break;
  }

plot(seq(1,length(error_vec),1),error_vec, pch= 19,
     main= "Variation of Epoch Error",xlab="Number of Epochs",
     ylab=" Epoch Error", col = "red", cex.main=1.7,cex.lab=1.5, cex.axis=1.3)
lines(seq(1,length(error_vec),1),error_vec, type = "l")

#validate the perceptron on the validation dataset (10%)
validation_ninputs = nrow(valid)
validation_targets = valid[,ncol(valid)]
validation_inputs = cbind(as.matrix(valid[,1:10]), rep(-1,validation_ninputs))

test_error_vec<-c();valid_accuracy<-c();mae_valid<-c();rmse_valid<-c()
#Do not update the weights for every epoch, only validate weights at every epoch
for (j in 1:nrow(epoch_weights)){
  error = 0
  wrong<-c()
  for (i in 1:nrow(validation_inputs)) {
    x = validation_inputs[i,]
    t = validation_targets[i]
    y = sum ( x * epoch_weights[j,] )
    y=tanh(30*(y))
    error = error + (0.5*(t - y)^2);
    mae<-abs(t-y) #mean absolute error(calculated later)
    mae_valid<-append(mae_valid,mae)
    rmse_valid<-append(rmse_valid,mae^2)
    if (t!=y){
      misclassified<-1
      wrong<-append(wrong,misclassified)
    }
  }
  valid_accuracy[j]<-(1-(sum(wrong)/nrow(validation_inputs)))*100
  test_error_vec[j]<-error
}

#testing the perceptron on the remaining 10% of the data
test_ninputs = nrow(test)
test_targets = test[,ncol(test)]
test_inputs = cbind(as.matrix(test[,1:10]), rep(-1,test_ninputs))
#the perceptron is tested at the closest possible epoch with good error
for (i in 1:nrow(test_inputs)) {
  x = test_inputs[i,]
  t = test_targets[i]
  y = sum ( x * epoch_weights[10,] )
  y=tanh(30*(y))
  error = error + (0.5*(t - y)^2);
}
points(seq(1,length(test_error_vec),1),test_error_vec, pch= 19, col = "blue")
lines(seq(1,length(test_error_vec),1),test_error_vec, type = "l")
points(10,error, pch= 12, col = "black")
grid(12, 12, lwd = 1)
legend("topright",
       legend = c("Training Error", "Validation Error", "Testing Error"),
       col = c("red", "blue", "black"),
       pch = c(19, 19, 12),
       bty = "n",
       pt.cex = 2,
       cex = 1.2,
       text.col = "black",
       horiz = F ,
       inset = c(0.1, 0.1))

#Example of input and output
output<-paste("output:",tanh(30*(sum ( test_inputs[1,]*epoch_weights[10,]))),
              "target:",test_targets[1])
input<-paste(c("inputs: ",test_inputs[1,]), collapse=" ")
input
output

#Accuracy Plot
plot(seq(1,length(train_accuracy),1),train_accuracy,col="red", cex.main=1.7,
     cex.lab=1.5, cex.axis=1.3, main= "Variation of Accuracy", lwd=2,
     xlab="Number of Epochs", ylab="Accuracy %", type="l", ylim=c(80,100))
lines(seq(1,length(valid_accuracy),1),valid_accuracy, col="blue", type="l",
      lwd=2)
grid(10, 10, lwd = 2)
legend("bottomright",
       legend = c("Training Accuracy", "Validation Accuracy"),
       col = c("red", "blue"),
       bty = "n",
       lty=1,
       cex = 1.2,
       text.col = "black",
       horiz = F ,
       inset = c(0.1, 0.1))

#MAE
print(paste("MAE of Training:", sum(mae_train)/length(mae_train)))
print(paste("MAE of Validation:", sum(mae_valid)/length(mae_valid)))

#RMSE
print(paste("RMSE of Training:", sqrt(mean(rmse_train))))
print(paste("RMSE of Validation:",sqrt(mean(rmse_valid))))

#### PRODUCT #####

inputs<-c();dataset<-c();output<-c()
n<-2000
for (j in 1:n){
  for (i in 1:10){
    input<-ceiling(runif(1,-1,1))
    if (input==0){
      inputs[i]<--1
    }else{
      inputs[i]<-1
    }
  }
  dataset<-rbind(dataset,inputs)
}
dataset<-unique(dataset)
#output based on input parity
for (row in 1:nrow(dataset)){
  out<-prod(dataset[row,])
  if (out<0){
    output[row]<--1
  }else{
    output[row]<-1
  }
}

dataset<-cbind(dataset,output)
train<-dataset[1:(round((4*nrow(dataset))/5)),] #80%
test<-dataset[nrow(train):nrow(dataset),]#20%

ninputs = nrow(train)
targets = train[,ncol(train)]
inputs = cbind(as.matrix(train[,1:10]), rep(-1,ninputs))
epsilon = 0.03
wts = runif(11)
train_accuracy<-c()#initialise accuracy vector
wrong<-c(0); mae_train<-c(); rmse_train<-c()

error_vec<-c()
for (iteration in 1:80) {
  order = sample(ninputs);
  error = 0;
  wrong<-c()
  for (i in order) {
    x = inputs[i,]
    t = targets[i]
    y = sum ( x * wts )
    y=tanh(30*(y))
    error = error + (0.5*(t - y)^2);
    dw = epsilon * ( t - y ) * x #+ (0.1*dw)
    wts = wts + dw;
    mae<-abs(t-y) #mean absolute error(calculated later)
    mae_train<-append(mae_train,mae)
    rmse_train<-append(rmse_train,mae^2)
    if (t!=y){
      misclassified<-1
      wrong<-append(wrong,misclassified)
    }
  }
  train_accuracy[iteration]<-(1-(sum(wrong)/ninputs))*100
  error_vec[iteration]<-error
  if (error == 0)
    break;
}

training_wts<-wts

par(mfrow=c(1,2))
plot(seq(1,length(error_vec),1),error_vec, pch= 19,
     main= "Variation of Epoch Error in Training",xlab="Number of Epochs",
     ylab=" Epoch Error", col = "red", cex.main=1.7, cex.lab=1.5, cex.axis=1.3)
lines(seq(1,length(error_vec),1),error_vec, type = "l")
grid(10,10)

plot(seq(1,length(train_accuracy),1),train_accuracy,col="red", cex.main=1.7,
    cex.lab=1.5, cex.axis=1.3, main= "Variation of Training Accuracy", lwd=2,
    xlab="Number of Epochs", ylab="Accuracy %", type="l")
grid(10,10)
