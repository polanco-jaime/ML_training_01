from google.cloud import bigquery, bigquery_storage, storage
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error 
from typing import Union, List
import os, logging, json, pickle, argparse
import dask.dataframe as dd
import pandas as pd
import numpy as np
# import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor

import joblib

# # Save the model
# 

# feature selection.  The FEATURE list defines what features are needed from the training data.
# as well as the types of those features. We will perform different feature engineering depending on the type

# List all column names for binary features: 0,1 or True,False or Male,Female etc
BINARY_FEATURES = [ ]

# List all column names for numeric features
NUMERIC_FEATURES = ['weight', 'volume' ,'usd' , 'quantity' ]  #, 

# List all column names for categorical features 
CATEGORICAL_FEATURES = ['driverId', 'locationId' ,'warehouseId' ,'sku' , 'storeId','orderId' ] # ] 

# ALL_COLUMNS = BINARY_FEATURES+NUMERIC_FEATURES+CATEGORICAL_FEATURES
ALL_COLUMNS =  NUMERIC_FEATURES+CATEGORICAL_FEATURES

# define the column name for label
LABEL = 'ServiceTime'

# Define the index position of each feature. This is needed for processing a 
# numpy array (instead of pandas) which has no column names.
BINARY_FEATURES_IDX = list(range(0,len(BINARY_FEATURES)))
NUMERIC_FEATURES_IDX = list(range(len(BINARY_FEATURES), len(BINARY_FEATURES)+len(NUMERIC_FEATURES)))
CATEGORICAL_FEATURES_IDX = list(range(len(BINARY_FEATURES+NUMERIC_FEATURES), len(ALL_COLUMNS)))



def load_data_from_bq(bq_uri: str) -> pd.DataFrame:
    '''
    Loads data from BigQuery table (BQ) to a dataframe

            Parameters:
                    bq_uri (str): bq table uri. i.e: example_project.example_dataset.example_table
            Returns:
                    pandas.DataFrame: a dataframe with the data from GCP loaded
    '''
    if not bq_uri.startswith('bq://'):
        raise Exception("uri is not a BQ uri. It should be bq://project_id.dataset.table")
        
    logging.info("reading bq data: {}".format(bq_uri))
    project,dataset,table =  bq_uri.split(".")
    bqclient = bigquery.Client(project=project[5:])
    bqstorageclient = bigquery_storage.BigQueryReadClient()
    query_string = """
    SELECT * from {ds}.{tbl}
    """.format(ds=dataset, tbl=table)

    return (
        bqclient.query(query_string)
        .result()
        .to_dataframe(bqstorage_client=bqstorageclient)
    )



def load_data_from_gcs(data_gcs_path: str) -> pd.DataFrame:
    '''
    Loads data from Google Cloud Storage (GCS) to a dataframe

            Parameters:
                    data_gcs_path (str): gs path for the location of the data. Wildcards are also supported. i.e gs://example_bucket/data/training-*.csv

            Returns:
                    pandas.DataFrame: a dataframe with the data from GCP loaded
    '''
        
    # using dask that supports wildcards to read multiple files. Then with dd.read_csv().compute we create a pandas dataframe
    # Additionally I have noticed that some values for TotalCharges are missing and this creates confusion regarding TotalCharges the data types. 
    # to overcome this we manually define TotalCharges as object. 
    # We will later fix this upnormality
    logging.info("reading gs data: {}".format(data_gcs_path))
    return dd.read_csv(data_gcs_path, dtype={'TotalCharges': 'object'}).compute()

def clean_missing_numerics(df: pd.DataFrame, numeric_columns):
    '''
    removes invalid values in the numeric columns        

            Parameters:
                    df (pandas.DataFrame): The Pandas Dataframe to alter 
                    numeric_columns (List[str]): List of column names that are numberic from the DataFrame
            Returns:
                    pandas.DataFrame: a dataframe with the numeric columns fixed
    '''
    
    for n in numeric_columns:
        df[n] = pd.to_numeric(df[n], errors='coerce')
        
    df = df.fillna(df.mean())
         
    return df
def data_selection(df: pd.DataFrame, selected_columns: List[str], label_column: str) -> (pd.DataFrame, pd.Series):
    '''
    From a dataframe it creates a new dataframe with only selected columns and returns it.
    Additionally it splits the label column into a pandas Series.

            Parameters:
                    df (pandas.DataFrame): The Pandas Dataframe to drop columns and extract label
                    selected_columns (List[str]): List of strings with the selected columns. i,e ['col_1', 'col_2', ..., 'col_n' ]
                    label_column (str): The name of the label column

            Returns:
                    tuple(pandas.DataFrame, pandas.Series): Tuble with the new pandas DataFrame containing only selected columns and lablel pandas Series
    '''
    # We create a series with the prediciton label
    labels = df[label_column]
    
    data = df.loc[:, selected_columns]
    

    return data, labels
def pipeline_builder(params_lgb: dict, bin_ftr_idx: List[int], num_ftr_idx: List[int], cat_ftr_idx: List[int]) -> Pipeline:
    '''
    Construye un pipeline sklearn con preprocesamiento y configuracion del modelo.

    Los pasos de preprocesamiento son:
        * OrdinalEncoder - used for binary features
        * StandardScaler - used for numerical features
        * OneHotEncoder - used for categorical features
    El modelo "lgb.LGBMRegressor" es un modelo de regresion de gradient boosting que utiliza la biblioteca "LightGBM" en Python.

            Parameters:
                    "objective": Define la funcion objetivo que se utilizara durante el entrenamiento. En este caso, "regression" indica que se esta realizando una tarea de regresion.
                    "metric": Define la metrica utilizada para evaluar el rendimiento del modelo durante el entrenamiento. En este caso, "rmse" indica que se esta utilizando el error cuadratico medio (RMSE) como metrica.
                    "num_leaves": Define el número maximo de hojas que puede tener un arbol de decision. Un valor mas alto aumentara la complejidad del modelo y puede ayudar a mejorar el rendimiento, pero tambien puede aumentar el riesgo de sobreajuste.
                    "learning_rate": Es la tasa de aprendizaje del modelo, que controla el tamanio de los ajustes realizados en cada iteracion del algoritmo. Un valor mas alto significa ajustes mas grandes y un modelo que converge mas rapido, pero tambien puede aumentar el riesgo de sobreajuste.
                    "feature_fraction": Es la fraccion de caracteristicas que se seleccionan aleatoriamente en cada iteracion para construir un arbol. Un valor mas bajo reduce la complejidad del modelo y puede ayudar a evitar el sobreajuste.
                    "bagging_fraction": Es la fraccion de muestras de entrenamiento que se seleccionan aleatoriamente para entrenar cada arbol. Un valor mas bajo reduce la complejidad del modelo y puede ayudar a evitar el sobreajuste.
                    "bagging_freq": Es la frecuencia con la que se seleccionan nuevas muestras de entrenamiento para cada arbol. Un valor mas alto puede ayudar a mejorar el rendimiento del modelo.
                    "max_depth": Es la profundidad maxima permitida para cada arbol. Un valor mas bajo reduce la complejidad del modelo y puede ayudar a evitar el sobreajuste.
                    "min_child_samples": Es el número minimo de muestras requeridas para formar un nuevo nodo en el arbol. Un valor mas alto reduce la complejidad del modelo y puede ayudar a evitar el sobreajuste.
                    "min_child_weight": Es la suma minima de pesos necesarios en un nodo para continuar dividiendolo. Un valor mas alto reduce la complejidad del modelo y puede ayudar a evitar el sobreajuste.
                    "subsample_for_bin": Es el número de muestras utilizadas para la construccion de los bins. Un valor mas alto puede mejorar el rendimiento del modelo.
                    "min_split_gain": Es el umbral minimo necesario para dividir un nodo. Un valor mas alto reduce la complejidad del modelo y puede ayudar a evitar el sobreajuste.
                    "min_data_in_leaf": Es el número minimo de muestras requeridas en una hoja del arbol. Un valor mas alto reduce la complejidad del modelo y puede ayudar a evitar el sobreajuste.
                    "boosting_type": Define el tipo de boosting utilizado. En este caso, "gbdt" indica que se esta utilizando Gradient Boosting Decision Tree.
                    "verbosity": Controla la cantidad de informacion de registro que se mostrara durante el entrenamiento.
                    "n_jobs": Es el número de trabajos en paralelo que se ejecutan durante el entrenamiento. Un valor de -1 significa utilizar todos los procesadores disponibles.
                    "random_state": Es una semilla utilizada por el generador de números aleatorios del modelo para garantizar la reproducibilidad de los resultados.
            Returns:
                     Pipeline: sklearn.pipelines.Pipeline with preprocessing and model training
    '''
        
    # Definining a preprocessing step for our pipeline. 
    # it specifies how the features are going to be transformed
    preprocessor = ColumnTransformer(
        transformers=[
            ('bin', OrdinalEncoder(), bin_ftr_idx),
            ('num', StandardScaler(), num_ftr_idx),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_ftr_idx)], n_jobs=-1)


    # We now create a full pipeline, for preprocessing and training.
    # for training we selected a linear  regression
    
    # clf = lgb.LGBMRegressor(**params_lgb)
    clf = LGBMRegressor(**params_lgb)
    # clf.set_params(**params_lgb)
   
    
    return Pipeline(steps=[ ('preprocessor', preprocessor),
                          ('regression', clf)])

def train_pipeline(clf: Pipeline, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]) -> float:
    '''
    Trains a sklearn pipeline by fiting training data an labels and returns the accuracy f1 score
    
            Parameters:
                    clf (sklearn.pipelines.Pipeline): the Pipeline object to fit the data
                    X: (pd.DataFrame OR np.ndarray): Training vectors of shape n_samples x n_features, where n_samples is the number of samples and n_features is the number of features.
                    y: (pd.DataFrame OR np.ndarray): Labels of shape n_samples. Order should mathc Training Vectors X

            Returns:
                    score (float): Average F1 score from all cross validations
    '''
    # run cross validation to get training score. we can use this score to optimise training
    score = cross_val_score(clf, X, y, cv=10, n_jobs=-1).mean()
    
    # Now we fit all our data to the regression. 
    clf.fit(X, y)
    
    return score

def process_gcs_uri(uri: str) -> (str, str, str, str):
    '''
    Receives a Google Cloud Storage (GCS) uri and breaks it down to the scheme, bucket, path and file
    
            Parameters:
                    uri (str): GCS uri

            Returns:
                    scheme (str): uri scheme
                    bucket (str): uri bucket
                    path (str): uri path
                    file (str): uri file
    '''
    url_arr = uri.split("/")
    if "." not in url_arr[-1]:
        file = ""
    else:
        file = url_arr.pop()
    scheme = url_arr[0]
    bucket = url_arr[2]
    path = "/".join(url_arr[3:])
    path = path[:-1] if path.endswith("/") else path
    
    return scheme, bucket, path, file
def pipeline_export_gcs(fitted_pipeline: Pipeline, model_dir: str) -> str:
    '''
    Exports trained pipeline to GCS
    
            Parameters:
                    fitted_pipeline (sklearn.pipelines.Pipeline): the Pipeline object with data already fitted (trained pipeline object)
                    model_dir (str): GCS path to store the trained pipeline. i.e gs://example_bucket/training-job
            Returns:
                    export_path (str): Model GCS location
    '''
    scheme, bucket, path, file = process_gcs_uri(model_dir)
    if scheme != "gs:":
            raise ValueError("URI scheme must be gs")
    
    # Upload the model to GCS
    b = storage.Client().bucket(bucket)
    export_path = os.path.join(path, 'saved_model.pb')
    blob = b.blob(export_path)
    
    
    blob.upload_from_string(pickle.dumps(fitted_pipeline))
    return scheme + "//" + os.path.join(bucket, export_path)




def prepare_report(cv_score: float, model_params: dict, regression_report: str, columns: List[str], example_data: np.ndarray) -> str:
    '''
    Prepares a training report in Text
    
            Parameters:
                    cv_score (float): score of the training job during cross validation of training data
                    model_params (dict): dictonary containing the parameters the model was trained with
                    regression_report (str): Model regression report with test data type of r2_score
                    columns (List[str]): List of columns that where used in training.
                    example_data (np.array): Sample of data (2-3 rows are enough). This is used to include what the prediciton payload should look like for the model
            Returns:
                    report (str): Full report in text
    '''
    
    buffer_example_data = '['
    for r in example_data:
        buffer_example_data+='['
        for c in r:
            if(isinstance(c,str)):
                buffer_example_data+="'"+c+"', "
            else:
                buffer_example_data+=str(c)+", "
        buffer_example_data= buffer_example_data[:-2]+"], \n"
    buffer_example_data= buffer_example_data[:-3]+"]"
        
    report = """
                Training Job Report    
                  
                Cross Validation Score: {cv_score}

                Training Model Parameters: {model_params}
                    
                Test Data Regression Report:
                {regression_report} 

                Example of data array for prediciton:

                Order of columns:
                {columns}

                Example for clf.predict()
                {predict_example}


                Example of GCP API request body:
                {{
                    "instances": {json_example}
                }}

          """.format(
              cv_score=cv_score,
              model_params=json.dumps(model_params),
              regression_report=regression_report,
              columns = columns,
              predict_example = buffer_example_data,
              json_example = json.dumps(example_data.tolist()))
    
    return report


def report_export_gcs(report: str, report_dir: str) -> None:
    '''
    Exports training job report to GCS
    
            Parameters:
                    report (str): Full report in text to sent to GCS
                    report_dir (str): GCS path to store the report model. i.e gs://example_bucket/training-job
            Returns:
                    export_path (str): Report GCS location
    '''
    scheme, bucket, path, file = process_gcs_uri(report_dir)
    if scheme != "gs:":
            raise ValueError("URI scheme must be gs")
            
    # Upload the model to GCS
    b = storage.Client().bucket(bucket)
    
    export_path = os.path.join(path, 'report.txt')
    blob = b.blob(export_path)
    
    blob.upload_from_string(report)
    
    return scheme + "//" + os.path.join(bucket, export_path)

# Define all the command line arguments your model can accept for training
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
   
    ''' 
          Vertex AI rellena automaticamente un conjunto de variables de entorno en el contenedor que ejecuta el trabajo de entrenamiento. 
          Estas variables incluyen:
              * AIP_MODEL_DIR - Directorio seleccionado como dir del modelo
              * AIP_DATA_FORMAT - Tipo de datos seleccionados para el entrenamiento (puede ser csv o bigquery)
          
          Vertex AI dividira automaticamente el conjunto de datos seleccionado en entrenamiento, validacion y prueba.
          y otras 3 variables de entorno reflejaran la ubicacion de los datos:
              * AIP_TRAINING_DATA_URI - URI de los datos de entrenamiento
              * AIP_VALIDATION_DATA_URI - URI de los datos de validacion
              * AIP_TEST_DATA_URI - URI de los datos de prueba
              
           Observe que estas variables de entorno son por defecto. Si el usuario proporciona un valor utilizando el argumento CLI, 
           la variable de entorno sera ignorada. Si el usuario no proporciona nada como argumento CLI 
           el programa tratara de utilizar las variables de entorno si existen. de lo contrario dejara vacio.
 
    '''   
    parser.add_argument(
        '--model_dir',
        help = 'Directory to output model and artifacts',
        type = str,
        default = os.environ['AIP_MODEL_DIR'] if 'AIP_MODEL_DIR' in os.environ else ""
    )
    parser.add_argument(
        '--data_format',
        choices=['csv', 'bigquery'],
        help = 'format of data uri csv for gs:// paths and bigquery for project.dataset.table formats',
        type = str,
        default =  os.environ['AIP_DATA_FORMAT'] if 'AIP_DATA_FORMAT' in os.environ else "csv"
    )
    parser.add_argument(
        '--training_data_uri',
        help = 'location of training data in either gs:// uri or bigquery uri',
        type = str,
        default =  os.environ['AIP_TRAINING_DATA_URI'] if 'AIP_TRAINING_DATA_URI' in os.environ else ""
    )
    parser.add_argument(
        '--validation_data_uri',
        help = 'location of validation data in either gs:// uri or bigquery uri',
        type = str,
        default =  os.environ['AIP_VALIDATION_DATA_URI'] if 'AIP_VALIDATION_DATA_URI' in os.environ else ""
    )
    parser.add_argument(
        '--test_data_uri',
        help = 'location of test data in either gs:// uri or bigquery uri',
        type = str,
        default =  os.environ['AIP_TEST_DATA_URI'] if 'AIP_TEST_DATA_URI' in os.environ else ""
    )
    
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

    
    
    args = parser.parse_args()
    arguments = args.__dict__
    

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        
        
    logging.info('Model artifacts will be exported here: {}'.format(arguments['model_dir']))
    
    logging.info('Data format: {}'.format(arguments["data_format"]))
    logging.info('Training data uri: {}'.format(arguments['training_data_uri']) )
    logging.info('Validation data uri: {}'.format(arguments['validation_data_uri']))
    logging.info('Test data uri: {}'.format(arguments['test_data_uri']))
    
    
    '''
    Tenemos 2 maneras diferentes de cargar nuestros datos a pandas. 
    Uno es desde el almacenamiento en la nube mediante la carga de archivos csv y y la otra es conectandonos a BigQuery.
    Vertex AI soporta ambas y Aqui hemos creado un codigo que, dependiendo del conjunto de datos proporcionado, seleccionara el metodo de carga apropiado.
    '''
    logging.info('Loading {} data'.format(arguments["data_format"]))
    if(arguments['data_format']=='csv'):
        df_train = load_data_from_gcs(arguments['training_data_uri'])
        df_test = load_data_from_bq(arguments['test_data_uri'])
        df_valid = load_data_from_gcs(arguments['validation_data_uri'])
    elif(arguments['data_format']=='bigquery'):
        print(arguments['training_data_uri'])
        df_train = load_data_from_bq(arguments['training_data_uri'])
        df_test = load_data_from_bq(arguments['test_data_uri'])
        df_valid = load_data_from_bq(arguments['validation_data_uri'])
    else:
        raise ValueError("Invalid data type ")
        '''
        Como utilizaremos Cross Validation, solo tendremos un conjunto de entrenamiento y un conjunto de prueba.
        Uniremos el set de test  y validacion para conseguir una division del 80%-20%.
        '''
    df_test = pd.concat([df_test,df_valid])
    
    logging.info('Defining model parameters')    
    model_params =  {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "max_depth": -1,
        "min_child_samples": 20,
        "min_child_weight": 0.001,
        "subsample_for_bin": 200000,
        "min_split_gain": 0.0,
        "min_data_in_leaf": 20,
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": 42
    }

    df_train = clean_missing_numerics(df_train, NUMERIC_FEATURES)
    df_test = clean_missing_numerics(df_test, NUMERIC_FEATURES)
    

    logging.info('Running feature selection')    
    X_train, y_train = data_selection(df_train, ALL_COLUMNS, LABEL)
    X_test, y_test = data_selection(df_test, ALL_COLUMNS, LABEL)

    logging.info('Training pipelines in CV')   
    clf = pipeline_builder(model_params, BINARY_FEATURES_IDX, NUMERIC_FEATURES_IDX, CATEGORICAL_FEATURES_IDX)
#### ok 
    cv_score = train_pipeline(clf, X_train, y_train)
    
    
    
    logging.info('Export trained pipeline and report')   
    pipeline_export_gcs(clf, arguments['model_dir'])

    y_pred = clf.predict(X_test)
    
    
    test_score = mean_squared_error(y_test, y_pred, squared=False)
    
    
    logging.info('mean squared error: '+ str(test_score))    
    
    test_score = r2_score(y_test, y_pred)
    logging.info('R square: '+ str(test_score))   

    
    report = prepare_report(cv_score,
                        model_params,
                        str(r2_score(y_test,y_pred)),
                        ALL_COLUMNS, 
                        X_test.to_numpy()[0:2])
    
    report_export_gcs(report, arguments['model_dir'])
    
    
    logging.info('Training job completed. Exiting...')    

