import chardet
import pandas as pd
import time
import re
from google.cloud import bigquery
# Método del codo para averiguar el número óptimo de clusters
import warnings

def metricas (classifier_model, X_test,y_test , Modelo):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score
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

class funcion_automl:
    import joblib
    import os 
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score
    ####################################################################
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    ####################################################################
    def __init__(self,
             X_train , y_train ,  y_test , X_test    
             ): 
        self.X_train = X_train
        self.y_train = y_train
        self.y_test = y_test
        self.X_test = X_test
        # self.models = models
    def folder_(self):
        if os.path.exists('./model') == False:
            os.mkdir('./model')
            return print('the folder to save models has been created: ' + str(os.path.exists('./model') ) )
        else:
            pass
            
        
    
    def modelo_logistico(self):
        # Creamos el folder donde guardaremos los modelos realizados
        self.folder_()
            
        # Ajustar el modelo de Regresión Logística en el Conjunto de Entrenamiento
        Modelo =  'Regresion Logistica'
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        
        Resultado= metricas (classifier_model = classifier,
                             X_test=  self.X_test,
                             y_test= self.y_test ,
                             Modelo= Modelo)
        # guadamos el modelo
        Modelo_ = Modelo.replace(' ', '').lower().replace('á','a').replace('ó','o')
        joblib.dump(classifier,'./model/'+ Modelo_+'.model' )
        return Resultado

    def modelo_naive_bayes(self):
        # Creamos el folder donde guardaremos los modelos realizados
        self.folder_()
        # Ajustar el modelo de probabilidad bayersiana ngenua en el Conjunto de Entrenamiento
        
        Modelo =  'Naive Bayes'
        classifier = GaussianNB().fit(self.X_train, self.y_train)
        Resultado= metricas (classifier_model = classifier,
                             X_test=  self.X_test,
                             y_test= self.y_test ,
                             Modelo= Modelo)
        # guadamos el modelo
        Modelo_ = Modelo.replace(' ', '').lower().replace('á','a').replace('ó','o')
        joblib.dump(classifier,'./model/'+ Modelo_+'.model' )
        return Resultado

    def modelo_SVM_lineal(self):
        # Creamos el folder donde guardaremos los modelos realizados
        Modelo =  'SVM lineal'
        self.folder_()
        classifier = SVC(kernel = "linear", random_state = 0).fit(self.X_train, self.y_train)
        Resultado= metricas (classifier_model = classifier,
                             X_test=  self.X_test,
                             y_test= self.y_test ,
                             Modelo= Modelo)
        # guadamos el modelo
        Modelo_ = Modelo.replace(' ', '').lower().replace('á','a').replace('ó','o')
        joblib.dump(classifier,'./model/'+ Modelo_+'.model' )
        return Resultado

    def modelo_SVM_Kernel(self):
        # Creamos el folder donde guardaremos los modelos realizados
        Modelo =  'SVM Kernel'
        self.folder_()
        classifier = SVC(kernel = "rbf", random_state = 0).fit(self.X_train, self.y_train)
        Resultado= metricas (classifier_model = classifier,
                             X_test=  self.X_test,
                             y_test= self.y_test ,
                             Modelo= Modelo)
        # guadamos el modelo
        Modelo_ = Modelo.replace(' ', '').lower().replace('á','a').replace('ó','o')
        joblib.dump(classifier,'./model/'+ Modelo_+'.model' )
        return Resultado

    def modelo_arboles_de_desicion(self):
        Modelo =  'Árbol de decisión'
        # Creamos el folder donde guardaremos los modelos realizados
        self.folder_()
        classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0).fit(X_train, y_train)
      
        Resultado= metricas (classifier_model = classifier,
                             X_test=  self.X_test,
                             y_test= self.y_test ,
                             Modelo= Modelo)
        # guadamos el modelo
        Modelo_ = Modelo.replace(' ', '').lower().replace('á','a').replace('ó','o')
        joblib.dump(classifier,'./model/'+ Modelo_+'.model' )
        return Resultado
 
    def modelo_Random_Forest(self):
        # Creamos el folder donde guardaremos los modelos realizados
        Modelo =  'Random Forest'
        self.folder_()
        classifier = RandomForestClassifier(n_estimators = 20, criterion = "entropy", random_state = 0).fit(self.X_train, self.y_train)
        Resultado= metricas (classifier_model = classifier,
                             X_test=  self.X_test,
                             y_test= self.y_test ,
                             Modelo= Modelo)
        # guadamos el modelo
        Modelo_ = Modelo.replace(' ', '').lower().replace('á','a').replace('ó','o')
        joblib.dump(classifier,'./model/'+ Modelo_+'.model' )
        return Resultado
 
    def modelo_XGBClassifier(self):
        Modelo =  'XBoosted Classifer'
        # Creamos el folder donde guardaremos los modelos realizados
        self.folder_()
        
        classifier = XGBClassifier().fit(X_train, y_train)
        Resultado= metricas (classifier_model = classifier,
                             X_test=  self.X_test,
                             y_test= self.y_test ,
                             Modelo= Modelo)
        # guadamos el modelo
        Modelo_ = Modelo.replace(' ', '').lower().replace('á','a').replace('ó','o')
        joblib.dump(classifier,'./model/'+ Modelo_+'.model' )
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

        return Resultado.set_index('Modelo').style.highlight_max(color = 'yellow', axis = 0)


def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    #remove % and cast to float
    data = data.str.replace('%','', regex=True).astype(float)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)


def schema_bq(columnas):
    import chardet
    import pandas as pd
    import time
    import re
    from google.cloud import bigquery
    # Método del codo para averiguar el número óptimo de clusters
    import warnings
    schema_ = []
    for i in range(len(columnas)):
        if columnas[i] == '':
            columna = 'UNNAMED' + str(i)
            schema_line   =   bigquery.SchemaField(columna , 'STRING' , mode = 'NULLABLE')
            schema_.append(schema_line)
            
        else:
            schema_line   =   bigquery.SchemaField("""{}""".format(columnas[i]), 'STRING' , mode = 'NULLABLE')
            schema_.append(schema_line)
            
    return schema_

def load_simple_file_in_bq(dataset, data, squema, table_in_bq):      
        import chardet
        import pandas as pd
        import time
        import re
        from google.cloud import bigquery
        # Método del codo para averiguar el número óptimo de clusters
        import warnings

        bq_client = bigquery.Client()
        Data = data.astype(str)
        Schema_ = squema

        table_id = "{}.{}.{}".format( bq_client.project, dataset, table_in_bq )
        job_config = bigquery.LoadJobConfig( schema= Schema_, write_disposition='WRITE_TRUNCATE' )
 
        job = bq_client.load_table_from_dataframe(Data, table_id , job_config=job_config )
        RESULTADO = job.result()  # Wait for the job to complete. 
        print("La tabla cargo correctamente en {}.{}.{}".format( bq_client.project, dataset, table_in_bq ) )

def normalize_(string):
            replacements = (
                (" ","_")    ,("á", "a"),("à", "a"),
                ("é", "e"),("è", "e"),
                ("í", "i"),('ï»¿',''), ("ì","i"), ("ó","o"),  ("ò","o"),  ("ö","o"),
                
                ("ú", "u"),("ù", "u"), ("ü", "u"), ("û", "u"),  
                ("ñ", "n"), ("*",""),
                (" ","_")    ,("Á", "A"),("À", "A"),
                ("É", "E"),("È", "E"),
                ("Í", "I"),('Ï»¿',''), ("Ì","I"), ("Ó","O"),  ("Ò","O"),  ("Ö","O"),
                
                ("Ú", "U"),("Ù", "U"), ("Ü", "U"), ("Û", "U"),  
                ("Ñ", "N"), ("*",""),
                ######################################################                
                ('├í','a'),  ("├®","e"), ('├¡','i'), ('├│','o'),
                ('├║','u'), ('├▒','ni'),  
                ('├ü','A'),('├Ç','A'),('├ç','A'),(":","_"),
                ('├ë','E'),("├ë","E"),("├ê","E"),("╔","E"),
                ('├ì','I'),("├î","I"),('├û','I'),('├Å','I'),
                ('├ô','O'),('├ï','O'),('├è','O'),('@', 'O'),
                ('├Ö','U'),('%',''),
                ('├æ','Ñ'),('├É','ni'),('├æ','Ñ'),('Ð','ni'),
                ('├Ü','U'),("\xef\xbb\xbf",""), ("\ufeff","") ,
                ('├û','Í' )  ,  ("\n","_"),
                ###############################################################
                (".", "_"), ("/","_"),("\\", "_"), (")", "_"),("(", "_"),('>',''), ('<',''),
                ("\¿", ""),("\?", ""),(" ", "_"),(",", "_"),("__", "_"),("-", ""),
                ("/¿", ""),("/?", ""),
                ("0_", "X0_"), ("1_", "X1_"),("2_", "X2_"),("3_", "X3_"), ("4_", "X4_"),
                ("5_", "X5_"),("6_", "X6_"),("7_", "X7_"),("8_", "X8_"),("9_", "X9_"),
                ("__", "_"),  (" ", "_"),  (" ", "_")
             )
            for a, b in replacements:        
                string = string.replace(a, b).replace(a.upper(), b.upper()) 
                #R = re.sub(r"[^a-zA-Z0-9]","",string.upper()) 
            return string.upper()

        
def normalize_coma(string):
            replacements = (
                (" ","_")    ,("á", "a"),("à", "a"),
                ("é", "e"),("è", "e"),
                ("í", "i"),('ï»¿',''), ("ì","i"), ("ó","o"),  ("ò","o"),  ("ö","o"),
                
                ("ú", "u"),("ù", "u"), ("ü", "u"), ("û", "u"),  
                ("ñ", "n"), ("*",""),
                (" ","_")    ,("Á", "A"),("À", "A"),
                ("É", "E"),("È", "E"),
                ("Í", "I"),('Ï»¿',''), ("Ì","I"), ("Ó","O"),  ("Ò","O"),  ("Ö","O"),
                
                ("Ú", "U"),("Ù", "U"), ("Ü", "U"), ("Û", "U"),  
                ("Ñ", "N"), ("*",""),
                ######################################################                
                ('├í','a'),  ("├®","e"), ('├¡','i'), ('├│','o'),
                ('├║','u'), ('├▒','ni'),  
                ('├ü','A'),('├Ç','A'),('├ç','A'),(":","_"),
                ('├ë','E'),("├ë","E"),("├ê","E"),("╔","E"),
                ('├ì','I'),("├î","I"),('├û','I'),('├Å','I'),
                ('├ô','O'),('├ï','O'),('├è','O'),('@', 'O'),
                ('├Ö','U'),('%',''),
                ('├æ','Ñ'),('├É','ni'),('├æ','Ñ'),('Ð','ni'),
                ('├Ü','U'),("\xef\xbb\xbf",""), ("\ufeff","") ,
                ('├û','Í' )  ,  ("\n","_"),
                ###############################################################
                (".", "_"), ("/","_"),("\\", "_"), (")", "_"),("(", "_"),('>',''), ('<',''),
                ("\¿", ""),("\?", ""),(" ", "_"),("__", "_"),("-", ""),
                ("/¿", ""),("/?", ""),
                ("0_", "X0_"), ("1_", "X1_"),("2_", "X2_"),("3_", "X3_"), ("4_", "X4_"),
                ("5_", "X5_"),("6_", "X6_"),("7_", "X7_"),("8_", "X8_"),("9_", "X9_"),
                ("__", "_"),  (" ", "_"),  (" ", "_")
             )
            for a, b in replacements:        
                string = string.replace(a, b).replace(a.upper(), b.upper()) 
            return string.upper()

        
        
def encabezado(tabla_csv, delimitador ):
    import subprocess as sp
    if delimitador==';':
        output = str(sp.getoutput("head -1 data/{}.csv".format(tabla_csv)))
        output = str(normalize_(output))
        return output
    elif delimitador == ',':
        output = str(sp.getoutput("head -1 data/{}.csv".format(tabla_csv)))
        output = str(normalize_coma(output))
        return output 
    else:
        pass
        
        
def optimal_cluster (X):
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1,11), wcss)
    plt.title("Método del codo")
    plt.xlabel("Número de Clusters")
    plt.ylabel("WCSS(k)")
    plt.show()
def minmax_norm(df):
    return df #(df - df.min()) / ( df.max() - df.min())
def feature_eng(tabla, X, Y, X_num , X_Cat ):
    label =  tabla[Y]
    features = tabla[ X ]
    #####
    numeric_feature = X_num
    lista = X_Cat
    # HERE WE CAN NORMALI
    features = pd.get_dummies(features, columns=lista, drop_first=False)    
    # for i in numeric_feature:
    #     features[i] = minmax_norm(features[i] )
    # ##
    from sklearn.model_selection import train_test_split
    train_features, test_features, train_label, test_label = train_test_split(features, label, test_size = 0.25, random_state = 0)
    
    columnas = train_features.columns
     
    with open(r'./model/col_predict.txt', 'w') as fp:
        fp.write('\n'.join(columnas))
    return  train_features, test_features, train_label, test_label

def feature_eng_predict(tabla, X,  ID_ , X_Cat ):
    ##### 
    features = tabla[ X ]
    IDENTIFICADOR  = tabla[ID_]
    # try:
    #     IDENTIFICADOR = features[ID]    
    # except:
    #     IDENTIFICADOR = features.index
            
    ### Comparamos las variables de prediccion 
    variables = []     
    with open(r'./model/col_predict.txt', 'r') as fp:
        for line in fp:
            x = line[:-1]
            variables.append(x)
    lista = X_Cat
    # Creamos las dummies de acuerdo a las nuevas variables
    features = pd.get_dummies(features, columns=lista, drop_first=False)  
    for i in variables:
        try:
            features[i]  = features[i]  
        except KeyError: 
            features[i]  = 0
    features = features[variables]

    return features.set_index(IDENTIFICADOR)
 

class load_simple_file_class:
    import chardet
    import pandas as pd
    import time
    import re
    from google.cloud import bigquery
    # Método del codo para averiguar el número óptimo de clusters
    import warnings
    def __init__(self,
                 path = "" ,
                 delimit = "",
                 
                 ):  
        
        # Instance Variable 
        self.tabla =path
        self.formato_ = path.split('.')[-1].upper()
        self.table_name = path.split('/')[-1].upper().split('.')[0]  
        self.delimit = delimit
        
    def econde_tabla(self): 
        if (self.formato_.upper() == 'CSV') or (self.formato_.upper() == 'TXT'):        
            import chardet

            with open(self.tabla, 'rb') as rawdata:
                result = chardet.detect(rawdata.read(10000))

            return result['encoding']
        else:
            pass

    def delimitador(self):
        if self.delimit == "":
            x = 15
            if (self.formato_.upper() == 'CSV') or (self.formato_.upper() == 'TXT'):
                self.encode = self.econde_tabla( )
                if pd.read_csv(self.tabla, sep=',', encoding= self.encode , nrows = 1).shape[1]>x:
                    sep_ = ','
                elif pd.read_csv(self.tabla, sep=';', encoding= self.encode, nrows = 1).shape[1]>x:
                    sep_ = ';'
                elif pd.read_csv(self.tabla, sep='|', encoding= self.encode, nrows = 1).shape[1]>x:
                    sep_ = '|'
                elif pd.read_csv(self.tabla, sep=':', encoding= self.encode, nrows = 1).shape[1]>x:
                    sep_ = ':'  
                else:
                    pass  
                return sep_          
            else:
                pass
        else:
            sep_ = self.delimit
            return sep_  
         

    def types_dict(self):
        if self.formato_.upper() == 'CSV'  or self.formato_.upper() == 'TXT' :
            pass
        #     col_names = pd.read_csv(self.tabla,sep= self.delimitador(), nrows=0, encoding =  self.econde_tabla() ).columns
        #     types_dict_ = { }
        #     types_dict_.update({col: str for col in col_names  })
        #     return types_dict_
        # elif self.formato_.upper() == 'XLS' or self.formato_.upper() == 'XLSX':
        #     col_names = pd.read_excel(self.tabla, nrows = 1 )
        #     types_dict_ = { }
        #     types_dict_.update({col: str for col in col_names  })
        #     return types_dict_
        # elif self.formato_.upper() == 'XLSB' :
        #     col_names = pd.read_excel(self.tabla, nrows = 1 )
        #     types_dict_ = { }
        #     types_dict_.update({col: str for col in col_names  })
        #    return types_dict_
        else:
            pass
        
       # print('reading parameters have been set');
 
    def read_tables(self):
        self.start = time.time()
        ### 
        if  (self.formato_.upper() == 'CSV'  or self.formato_.upper() == 'TXT' ) :
            
            try:
                Data  =  pd.read_csv(self.tabla, sep=self.delimitador(), 
                                     encoding= self.econde_tabla(),  
                                     warn_bad_lines=False, error_bad_lines=False )
            except:
                Data = pd.read_csv(self.tabla, sep=self.delimitador(), 
                                   encoding= self.econde_tabla(),
                                   engine = 'python',    warn_bad_lines=False,
                                   error_bad_lines=False )
            
            return Data
        
        elif (self.formato_.upper() == 'XLS' or self.formato_.upper() == 'XLSX' ):    
            
            Data  = pd.read_excel(self.tabla, converters = self.types_dict())
            
            return Data
        else:
            pass
        
    def read_table(self):
        start = time.time()
        def normalize_(string):
            replacements = (
                (" ","_")    ,("á", "a"),("à", "a"),
                ("é", "e"),("è", "e"),
                ("í", "i"),('ï»¿',''), ("ì","i"), ("ó","o"),  ("ò","o"),  ("ö","o"),
                
                ("ú", "u"),("ù", "u"), ("ü", "u"), ("û", "u"),  
                ("ñ", "n"), ("*",""),
                (" ","_")    ,("Á", "A"),("À", "A"),
                ("É", "E"),("È", "E"),
                ("Í", "I"),('Ï»¿',''), ("Ì","I"), ("Ó","O"),  ("Ò","O"),  ("Ö","O"),
                
                ("Ú", "U"),("Ù", "U"), ("Ü", "U"), ("Û", "U"),  
                ("Ñ", "N"), ("*",""),
                ######################################################                
                ('├í','a'),  ("├®","e"), ('├¡','i'), ('├│','o'),
                ('├║','u'), ('├▒','ni'),  
                ('├ü','A'),('├Ç','A'),('├ç','A'),(":","_"),
                ('├ë','E'),("├ë","E"),("├ê","E"),("╔","E"),
                ('├ì','I'),("├î","I"),('├û','I'),('├Å','I'),
                ('├ô','O'),('├ï','O'),('├è','O'),('@', 'O'),
                ('├Ö','U'),('%',''),
                ('├æ','Ñ'),('├É','ni'),('├æ','Ñ'),('Ð','ni'),
                ('├Ü','U'),("\xef\xbb\xbf",""), ("\ufeff","") ,
                ('├û','Í' )  ,  ("\n","_"),
                ###############################################################
                (".", "_"), ("/","_"),("\\", "_"), (")", "_"),("(", "_"),('>',''), ('<',''),
                ("\¿", ""),("\?", ""),(" ", "_"),(",", "_"),("__", "_"),("-", ""),
                ("/¿", ""),("/?", ""),
                ("0_", "X0_"), ("1_", "X1_"),("2_", "X2_"),("3_", "X3_"), ("4_", "X4_"),
                ("5_", "X5_"),("6_", "X6_"),("7_", "X7_"),("8_", "X8_"),("9_", "X9_"),
                ("__", "_"),  (" ", "_"),  (" ", "_")
             )
            for a, b in replacements:        
                string = string.replace(a, b).replace(a.upper(), b.upper()) 
                #R = re.sub(r"[^a-zA-Z0-9]","",string.upper()) 
            return string.upper()
        
        nombre_columnas = []
        Data = self.read_tables()
         
        for i in range(len(Data.columns)):
            lista  = normalize_(pd.DataFrame(list(Data.columns), columns =['columnas']).columnas[i].upper() )[0:20] + '_'+ str(1+i)
            nombre_columnas.append(lista)
 
 
        
        Data.columns = nombre_columnas
        schema_ = []
        for i in Data.columns:
            schema_line   =   bigquery.SchemaField("""{}""".format(i), 'STRING' , mode = 'NULLABLE')
            schema_.append(schema_line)
        print('esquema creado correctamente')
        
        return [Data, schema_, nombre_columnas]
        

