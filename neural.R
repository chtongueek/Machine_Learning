neural = function(IN, hide, out, data, outcome, LR, steps, BIAS = 1 , test, BS = 1, val = NULL){
  
  runtimes = round(dim(data)[1]/BS, 0)
  
  IN = as.matrix(IN)
  hide = as.matrix(hide)
  out = as.matrix(out)
  
  INb = IN + BIAS
  hideb = hide + BIAS
  
  neur = cbind(IN, hide, out)
  neurb = cbind(INb, hideb, out)
  
  data  = as.matrix(data)
  outcome = as.matrix(outcome)
  test = as.matrix(test)
  
  error = matrix(1, steps, 1)
  
  ah = sqrt(6/(INb + hideb))
  ao = sqrt(6/(out + hideb[length(hideb)]))
  
  wh = list()
  
  for (i in 1:(length(neur) - 2)){
    
    wTemp = seq(-ah[1,i], ah[1,i], length.out = neur[1, i+1]*neurb[1,i])
    whTemp = matrix(wTemp, neur[1, i+1], neurb[1, i], byrow = T)
    wh[[length(wh) + 1]] =  whTemp
    
  }
  
  woTemp = seq(-ao[1,i], ao[1,i], length.out = (neurb[1, length(neurb) - 1]*neur[1,length(neur)]))
  wo = matrix(woTemp, neurb[1, length(neur)], neurb[1, length(neurb)-1], byrow = T)
  
  bias = matrix(1, dim(data)[1])
  if (BIAS == 1){
    data = cbind(data, bias)
  }
  
  data = t(data)
  
  outcome = t(outcome)
  
  allYout = matrix(0, out, dim(data)[2])
  
  ### TP/FP Section ###
  ###
  
  for (i in 1:steps){
    
    for (j in 1:runtimes){

      start = BS*(j-1)
      stopt = BS*j
      
      if (stopt > dim(data)[2]){stopt = dim(data)[2]}
      
      batch = data[, start:stopt]
      batchOut = outcome[, start:stopt]
      
      act = list()
      prep = list()
      weight = list()

      act1 = batch[1:(dim(batch)[1]-1),]
      act[[length(act) + 1]] = act1
      prep[[length(prep) + 1]] = batch
      
      for (k in 1:dim(hide)[2]){
        weight[[length(weight) + 1]] = wh[[k]]%*%prep[[k]]
        act[[length(act) + 1]] = 1/(1 + exp(-weight[[k]]))
        
        bias2 = matrix(1, 1, dim(act[[k+1]])[2])
        if (BIAS == 1){
          prep[[length(prep) + 1]] = rbind(act[[k + 1]], bias2)
        }else{
          prep[[length(prep) + 1]] = act[[k + 1]]
        }
        
      }
      
      yOut = wo%*%prep[[length(prep)]]
      yOut = 1/(1+exp(-yOut))
      
      if (i == (steps -1)){

        allYout[, start:stopt] = yOut
        
      }

      error[i] = sum((yOut - batchOut)^2)
      
      #### Backpropigation ####
      
      if (i == (steps - 1) & (i != 0) & (j == runtimes - 1)){break}
      else{

        delta0 = (yOut - batchOut)*(yOut*(1-yOut))
        dE0 = delta0%*%t(prep[[length(prep)]])
        dW0 = -LR*dE0
        wo = wo + dW0
        
        delta = delta0
        wNext = wo
        
        for (k in 1:dim(hide)[2]){
          
          L = dim(hide)[2] - (k-1)
          wo2 = as.matrix(wNext[, 1:hide[1,L]])
          if(dim(wo2)[2] == dim(delta)[1]){
            Eh1 = wo2%*%delta
          }else{
            Eh1 = t(wo2)%*%delta
          }
          Eh2 = act[[L+1]]*(1-act[[L+1]])
          Eh = Eh2*Eh1
          dEh = Eh%*%t(prep[[L]])
          dWh = -LR*dEh
          wh[[L]] = wh[[L]] + dWh
          wNext = wh[[L]]
          delta = Eh
          
        }
        
      }
    
    }
    
  }

  testBias = matrix(1, 1, dim(test)[1])
  
  if (is.null(val)){
    val = matrix(0,1,1)
  }
  
  valBias = matrix(1,1, dim(val)[1])
  yhNew = test
  
  if (BIAS == 1){
    
    yhNew = rbind(t(yhNew), testBias)
    val = rbind(t(val), valBias)
    
    
  }else{
    yhNew = t(yhNew)
    val = t(val)
  
  }

  for (j in 1:length(hide)){
    
    yh1 = wh[[j]]%*%yhNew
    yh1 = 1/(1+exp(-yh1))
    yhNew = yh1
    
    val1 = wh[[j]]%*%val
    val1 = 1/(1+exp(-val1))
    val = val1
    
    if (BIAS == 1){
      
      yhNew = rbind(yhNew, testBias)
      val = rbind(val, valBias)
      
    } else {
      
      yhNew = yhNew
      val = val
      
    }
    
  }
  testOut = wo%*%yhNew
  testOut = 1/(1+exp(-testOut))
  
  valOut = wo%*%val
  valOut = 1/(1+exp(-valOut))
  
  ret = list(t(allYout), t(testOut), t(valOut), error)
  return(ret)
  
}
