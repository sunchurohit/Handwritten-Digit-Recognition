clear ; close all; clc;

train_data = loadMNISTImages('train-images-idx3-ubyte')' ; %% 60000 x 784 matrix 
train_label = loadMNISTLabels('train-labels-idx1-ubyte') ; %%60000 x 1 vector 

m = length(train_label) ; %%training data size 

test_data = loadMNISTImages('t10k-images-idx3-ubyte')';  %% 10000 x 784 matrix
test_label = loadMNISTLabels('t10k-labels-idx1-ubyte'); %%% 10000 x 1 vector 
m_test = length(test_label) ;

input_layer_size = 784 ;
hidden_layer_size = 250 ;
output_layer_size = 10 ;

init_epsilon = 0.03 ;

iterations = 1 ;
alpha = 0.01 ;
X = [ones(m ,1) train_data] ;

Theta1 = rand(hidden_layer_size,input_layer_size+1)*2*init_epsilon - init_epsilon ;
Theta2 = rand(output_layer_size,hidden_layer_size+1)*2*init_epsilon - init_epsilon ;

for j=1:iterations
  for i=1:m
    
    yy = zeros(output_layer_size,1) ;
    yy(train_label(i)+1) = 1;
    
    z2 = Theta1*X(i,:)' ;
    a2 = sigmoid(z2) ;
    a2 = [1;a2] ;
    z3 = Theta2*a2 ;
    a3 = sigmoid(z3) ;
    
    del3 = a3 - yy ;
    del2 = (Theta2'*del3).*(a2.*(1-a2)) ;
    
    Theta1 = Theta1 - alpha*del2(2:end)*X(i,:) ;
    Theta2 = Theta2 - alpha*del3*a2' ;
      
  endfor
endfor

X_test = [ones(m_test,1) test_data] ;
counter = 0;

for i=1:m_test
    
  z2 = Theta1*X_test(i,:)' ;
  a2 = sigmoid(z2) ;
  a2 = [1;a2] ;
  z3 = Theta2*a2 ;
  a3 = sigmoid(z3) ;
  
  [maxx ind] = max(a3) ;
  
  if(ind == test_label(i)+1)
    counter++ ;
  endif
  
endfor

acc = 100*counter/m_test ;

acc















