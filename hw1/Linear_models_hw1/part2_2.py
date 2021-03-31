# input: y_true - matrix with shape [1, n], y_predict with shape [k, n] 
# for example: y_true = [1, 0, 3, 3], y_predict = [[0.1, 0.5, 0.3, 0.1],
#                                                  [0.6, 0.1, 0.2, 0.1],
#                                                  [0.1, 0.6, 0.2, 0.1],
#                                                  [0.2, 0.2, 0.2, 0.4]].
#              (y_predict with labels = [1, 0, 1, 3]
# output: vector of metrix for each classes except accuracy_score

def accuracy_score(y_true, y_predict, percent = None):
    if not percent:
        percent = 50
    
    bound = int(percent/100*y_true.shape[0])
    assert bound > 0, 'Too low percent, slice hasnt any objects'
    
    result = (y_true[:bound] == y_predict.argmax(axis = 1)[:bound]).sum()/y_true[:bound].shape[0]
    return result


def precision_score(y_true, y_predict, percent = None):
    if not percent:
        percent = 50

    bound = int(percent/100*y_true.shape[0])
    assert bound > 0, 'Too low percent, slice hasnt any objects'
    
    result = 0
    for x_i in np.unique(y_true):
        
        tp = np.where(((y_predict[:bound].argmax(axis = 1) == x_i) == (y_true[:bound] == x_i)) &\
                      (y_predict[:bound].argmax(axis = 1) == x_i), 1, 0).sum()
        if tp == 0:
            continue
        
        # all_p = tp + fp
        all_p = (y_predict.argmax(axis = 1) == x_i).sum()
                              
        result += tp/all_p
    return result/y_true[:bound].shape[0]


def recall_score(y_true, y_predict, percent = None):
    if not percent:
        percent = 50
    
    bound = int(percent/100*y_true.shape[0])
    assert bound > 0, 'Too low percent, slice hasnt any objects'
    
    result = 0
    for x_i in np.unique(y_true):
        
        tp = np.where(((y_predict[:bound].argmax(axis = 1) == x_i) == (y_true[:bound] == x_i)) &\
                      (y_predict[:bound].argmax(axis = 1) == x_i), 1, 0).sum()
        if tp == 0:
            continue
        # all_t_p = tp + fn
        all_t_p = (y_true[:bound] == x_i).sum()
                            
        result += tp/(all_t_p)
    return result/y_true[:bound].shape[0]


def f1_score(y_true, y_predict, percent = None):
    r = recall_score(y_true, y_predict, percent)
    p = precision_score(y_true, y_predict, percent)
    result = 2*r*p/(r+p)
    return result


def lift_score(y_true, y_predict, percent = None):

    if not percent:
        percent = 50
    
    bound = int(percent/100*y_true.shape[0])
    assert bound > 0, 'Too low percent, slice hasnt any objects'
    
    p = precision_score_(y_true, y_predict, percent = None)
    result = 0
    for x_i in np.unique(y_true):
        
        all_t_p = (y_true[:bound] == x_i).sum()
        result += 2/(all_t_p)        
                                
    return result/y_true[:bound].shape[0]*p
