def accuracy_score(y_true, y_predict, percent = None):
    if not percent:
        percent = 50
    
    bound = int(percent/100*y_true.shape[0])
    assert bound > 0, '������� ����� percent, ��� �� ������ ��������� �� �������� �� ������ ��������'
    
    result = (y_true[:bound] == y_predict.argmax(axis = 1)[:bound]).sum()/y_true[:bound].shape[0]
    return result


def precision_score(y_true, y_predict, percent = None):
    if not percent:
        percent = 50

    bound = int(percent/100*y_true.shape[0])
    assert bound > 0, '������� ����� percent, ��� �� ������ ��������� �� �������� �� ������ ��������'
    
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
    assert bound > 0, '������� ����� percent, ��� �� ������ ��������� �� �������� �� ������ ��������'
    
    result = 0
    for x_i in np.unique(y_true):
        
        tp = np.where(((y_predict[:bound].argmax(axis = 1) == x_i) == (y_true[:bound] == x_i)) &\
                      (y_predict[:bound].argmax(axis = 1) == x_i), 1, 0).sum()
        if tp == 0:
            continue
        
        all_t_p = (y_true[:bound] == x_i).sum()
                            
        result += tp/(all_t_p)
    return result/y_true[:bound].shape[0]


def f1_score(y_true, y_predict, percent = None):
    r = recall_score_(y_true, y_predict, percent)
    p = precision_score_(y_true, y_predict, percent)
    result = 2*r*p/(r+p)
    return result


def lift_score(y_true, y_predict, percent = None):

    if not percent:
        percent = 50
    
    bound = int(percent/100*y_true.shape[0])
    assert bound > 0, '������� ����� percent, ��� �� ������ ��������� �� �������� �� ������ ��������'
    
    p = precision_score_(y_true, y_predict, percent = None)
    result = 0
    for x_i in np.unique(y_true):
        
        all_t_p = (y_true[:bound] == x_i).sum()
        result += 2/(all_t_p)        
                                
    return result/y_true[:bound].shape[0]*p