class funcion_automl:


    def __init__(self,
             X_train , y_train ,  y_test , X_test    
             ): 
        self.X_train = X_train
        self.y_train = y_train
        self.y_test = y_test
        self.X_test = X_test
        # self.models = models
        
    def modelo_logistico(self):
        # Ajustar el modelo de Regresión Logística en el Conjunto de Entrenamiento
        Modelo =  'Regresion Logistica'
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        Resultado= metricas (classifier_model = classifier,
                             X_test=  self.X_test,
                             y_test= self.y_test ,
                             Modelo= Modelo)
        return Resultado

    def modelo_naive_bayes(self):
        # Ajustar el modelo de probabilidad bayersiana ngenua en el Conjunto de Entrenamiento
        Modelo =  'Naive Bayes'
        classifier = GaussianNB().fit(self.X_train, self.y_train)
        Resultado= metricas (classifier_model = classifier,
                             X_test=  self.X_test,
                             y_test= self.y_test ,
                             Modelo= Modelo)
        return Resultado

    def modelo_SVM_lineal(self):
        classifier = SVC(kernel = "linear", random_state = 0).fit(self.X_train, self.y_train)
        Resultado= metricas (classifier_model = classifier,
                             X_test=  self.X_test,
                             y_test= self.y_test ,
                             Modelo= 'SVM Lineal')
        return Resultado

    def modelo_SVM_Kernel(self):
        classifier = SVC(kernel = "rbf", random_state = 0).fit(self.X_train, self.y_train)
        Resultado= metricas (classifier_model = classifier,
                             X_test=  self.X_test,
                             y_test= self.y_test ,
                             Modelo= 'SVM Kernel')
        return Resultado 

    def modelo_arboles_de_desicion(self):
        classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0).fit(X_train, y_train)
        Resultado= metricas (classifier_model = classifier,
                             X_test=  self.X_test,
                             y_test= self.y_test ,
                             Modelo= 'Árbol de Decisión')
        return Resultado 
 
    def modelo_Random_Forest(self):
        classifier = RandomForestClassifier(n_estimators = 20, criterion = "entropy", random_state = 0).fit(self.X_train, self.y_train)
        Resultado= metricas (classifier_model = classifier,
                             X_test=  self.X_test,
                             y_test= self.y_test ,
                             Modelo= 'Random Forest')
        return Resultado 
 
    def modelo_XGBClassifier(self):
        classifier = XGBClassifier().fit(X_train, y_train)
        Resultado= metricas (classifier_model = classifier,
                             X_test=  self.X_test,
                             y_test= self.y_test ,
                             Modelo= 'eXtreme Gradient Boost')
        return Resultado 
 
    def run_automl_models(self):
        Resultado = pd.DataFrame() 
        try:
            Resultado = Resultado.append(self.modelo_logistico(), ignore_index=True)
        except: 
            print("Modelo Logistico no pudo ser calculado")
        
        try:
            Resultado = Resultado.append(self.modelo_naive_bayes(), ignore_index=True)
        except: 
            print("Modelo naive_bayes no pudo ser calculado")
        try:
            Resultado = Resultado.append(self.modelo_arboles_de_desicion(), ignore_index=True)
        except: 
            print("Modelo arboles_de_desicion no pudo ser calculado")        
        try:
            Resultado = Resultado.append(self.modelo_Random_Forest(), ignore_index=True)
        except: 
            print("Modelo Random_Forest no pudo ser calculado")        
        try:
            Resultado = Resultado.append(self.modelo_XGBClassifier(), ignore_index=True)
        except: 
            print("Modelo XGBClassifier no pudo ser calculado")        
        #Resultado = Resultado.append(self.modelo_SVM_Kernel(), ignore_index=True)
        #Resultado = Resultado.append(self.modelo_SVM_lineal(), ignore_index=True)
        #Resultado = Resultado.style.apply(highlight_max)
        Resultado = Resultado.set_index('Modelo')
        #Rsultado = Resultado.style.highlight_max(color = 'yellow', axis = 0)
        return Resultado.style.highlight_max(color = 'yellow', axis = 0)

def metricas (classifier_model, X_test,y_test , Modelo):
    y_pred  = classifier_model.predict(X_test)
    y_pred = y_pred >=0.5 
    y_pred = y_pred.astype(int)
 
    dict_ = {'Modelo': [Modelo],
            'Accuracy': [accuracy_score(y_test, y_pred)],
            'Precision': [precision_score(y_test, y_pred)],
            'Recall': [recall_score(y_test, y_pred)],
            'F1': [f1_score(y_test, y_pred)],
           'ROC-AUC': [roc_auc_score(y_test, y_pred)]} 
    df = pd.DataFrame(dict_)
    return df  

def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    #remove % and cast to float
    data = data.replace('%','', regex=True).astype(float)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)
