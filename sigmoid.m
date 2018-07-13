function ans = sigmoid(z)
  
  %% ans has the same dimensions of z
  
  ans = 1.0 ./ (1.0 + exp(-z)) ; 
  
end
