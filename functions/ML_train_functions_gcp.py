import chardet
import pandas as pd
import time
import re
from google.cloud import bigquery
# Método del codo para averiguar el número óptimo de clusters
import warnings
warnings.filterwarnings("ignore")
# execfile("functions/automl_class.py")
################## 
def schema_bq(columnas):
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
##################
def load_simple_file_in_bq(dataset, data, squema, table_in_bq):      
        bq_client = bigquery.Client()
        Data = data.astype(str)
        Schema_ = squema

        table_id = "{}.{}.{}".format( bq_client.project, dataset, table_in_bq )
        job_config = bigquery.LoadJobConfig( schema= Schema_, write_disposition='WRITE_TRUNCATE' )
 
        job = bq_client.load_table_from_dataframe(Data, table_id , job_config=job_config )
        RESULTADO = job.result()  # Wait for the job to complete. 
        print("La tabla cargo correctamente en {}.{}.{}".format( bq_client.project, dataset, table_in_bq ) )
##################
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
##################
##################        
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

        
##################        
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
##################        
        
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
    
##################    
def minmax_norm(df):
    return df #(df - df.min()) / ( df.max() - df.min())
################## 
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
    return  train_features, test_features, train_label, test_label

##################
class load_simple_file_class:
    def __init__(self,
                 path = "" ,
                 delimit = "",
                 
                 ):  
        
        # Instance Variable 
        self.tabla =path
        self.formato_ = path.split('.')[-1].upper()
        self.table_name = path.split('/')[-1].upper().split('.')[0]  
        self.delimit = delimit
    ##################   ##################  ##################
    def econde_tabla(self): 
        if (self.formato_.upper() == 'CSV') or (self.formato_.upper() == 'TXT'):        
            import chardet

            with open(self.tabla, 'rb') as rawdata:
                result = chardet.detect(rawdata.read(10000))

            return result['encoding']
        else:
            pass
    ##################   ##################  ##################
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
         
    ##################   ##################  ##################
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
     ##################   ##################  ##################
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
    ##################   ##################  ##################        
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
    ##################   ##################  ##################        
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
        

