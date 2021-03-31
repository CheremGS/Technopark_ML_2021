# input: y_true - matrix with shape [1, n], y_predict with shape [k, n] 
# for example: y_true = [1, 0, 3, 3], y_predict = [[0.1, 0.5, 0.3, 0.1],
#                                                  [0.6, 0.1, 0.2, 0.1],
#                                                  [0.1, 0.6, 0.2, 0.1],
#                                                  [0.2, 0.2, 0.2, 0.4]].
#              (y_predict with labels = [1, 0, 1, 3]
# output: vector of metrics for each classes except accuracy_score

# P.S.: По умолчанию выставляется percent = 50, а не порог вероятностей = 0.5, 
# для многоклассового случая это не имеет смысла, так как иногда порог никто не преодолевает, или же наоборот сразу несколько преодолевают
# например y_predict = [... , [0.2, 0.4, 0.4], ...], при пороге = 0.5, ни один класс не пройдет, при пороге = 0.4 пройдут сразу два класса
# функции работают для многоклассового случая(в случае бинарной классификации вектор метрик имеет размер (2, 1), все метрики также работают)

def accuracy_score(y_true, y_predict, percent = None):
    if not percent:
        percent = 50
    
    bound = int(percent/100*y_true.shape[0])
    assert bound > 0, 'Слишком малый percent, топ от такого параметра не включает ни одного элемента'
    
    result = (y_true[:bound] == y_predict.argmax(axis = 1)[:bound]).sum()/y_true[:bound].shape[0]
    return result


def precision_score(y_true, y_predict, percent = None):
    if not percent:
        percent = 50

    bound = int(percent/100*y_true.shape[0])
    assert bound > 0, 'Слишком малый percent, топ от такого параметра не включает ни одного элемента'
    
    result = []
    x_unique = set(np.unique(y_true)).union(set(np.unique(y_predict.argmax(axis = 1))))

    for x_i in x_unique:
        
        tp = np.where(((y_predict[:bound].argmax(axis = 1) == x_i) == (y_true[:bound] == x_i)) &\
                      (y_predict[:bound].argmax(axis = 1) == x_i), 1, 0).sum()
        
        # all_p = tp + fp
        all_p = (y_predict.argmax(axis = 1) == x_i).sum()
                              
        result.append(tp/all_p)

    return np.nan_to_num(np.array(result), 0)


def recall_score_(y_true, y_predict, percent = None):
    if not percent:
        percent = 50
    
    bound = int(percent/100*y_true.shape[0])
    assert bound > 0, 'Слишком малый percent, топ от такого параметра не включает ни одного элемента'
    
    result = []
    x_unique = set(np.unique(y_true)).union(set(np.unique(y_predict.argmax(axis = 1))))
    
    for x_i in x_unique:
        tp = np.where(((y_predict[:bound].argmax(axis = 1) == x_i) == (y_true[:bound] == x_i)) &\
                      (y_predict[:bound].argmax(axis = 1) == x_i), 1, 0).sum()
        
        #all_t_p = tp + fn
        all_t_p = (y_true[:bound] == x_i).sum()
                            
        result.append(tp/(all_t_p))
    
    return np.nan_to_num(np.array(result), 0)


def f1_score_(y_true, y_predict, percent = None):
    r = recall_score_(y_true, y_predict, percent)
    p = precision_score_(y_true, y_predict, percent)
    result = 2*r*p/(r+p)
    return np.nan_to_num(np.array(result), 0)


def lift_score(y_true, y_predict, percent = None):
    if not percent:
      percent = 50
       
    bound = int(percent/100*y_true.shape[0])
    assert bound > 0, 'Слишком малый percent, топ от такого параметра не включает ни одного элемента'
    
    p = precision_score_(y_true, y_predict, percent = percent)
    
    result = []
    x_unique = set(np.unique(y_true)).union(set(np.unique(y_predict.argmax(axis = 1))))
    for x_i in x_unique:
        
        all_t_p = (y_true[:bound] == x_i).sum()
        result.append(p[int(x_i)]/(all_t_p/y_true[:bound].shape[0]))     
                                
    return np.nan_to_num(np.array(result), 0)

