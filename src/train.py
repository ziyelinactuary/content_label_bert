import sagemaker
import boto3
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_store import FeatureStore
import tensorflow as tf
import tensorflow_hub as hub
import s3fs
import fastparquet as fp
import numpy as np
import ast
import tensorflow as tf

region = "us-east-1"
featurestore_runtime = boto3.Session(region_name=region).client(
    service_name="sagemaker-featurestore-runtime", region_name=region
)
sm = boto3.Session(region_name=region).client(service_name="sagemaker", region_name=region)
s3 = boto3.Session(region_name=region).client(service_name="s3", region_name=region)
boto_session=boto3.Session(region_name=region)

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable = True )
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

def define_model():
    input_words_ids = tf.keras.layers.Input(shape =(max_seq_length),dtype  = tf.int32, name = "input_word_ids")
    input_mask = tf.keras.layers.Input(shape = (max_seq_length), dtype = tf.int32 , name = "input_mask")
    input_type_ids = tf.keras.layers.Input(shape = (max_seq_length),dtype = tf.int32,name = "input_type_ids")
    pooled_output,sequence_output = bert_layer([input_words_ids,input_mask,input_type_ids])

    drop = tf.keras.layers.Dropout(0.4)(pooled_output)
    output = tf.keras.layers.Dense(1,activation = 'sigmoid', name = "output")(drop)

    model = tf.keras.Model(
        inputs = {
            'input_word_ids' : input_words_ids,
            'input_mask' : input_mask,
            'input_type_ids' : input_type_ids
        },
        outputs = output
    )
    return model

def create_model():
    model = define_model()
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 2e-5),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = [tf.keras.metrics.BinaryAccuracy()]
    )
    return model

def load_data_as_data_frame():
    s3 = s3fs.S3FileSystem()
    fs = s3fs.core.S3FileSystem()
    s3_path = "content-descriptor-prediction-training/feature_group/280384355285/sagemaker/us-east-1/offline-store/content_metadata-1709314508/data/*/*/*/*/*.parquet"
    all_paths_from_s3 = fs.glob(path=s3_path)

    myopen = s3.open
    #use s3fs as the filesystem
    fp_obj = fp.ParquetFile(all_paths_from_s3,open_with=myopen)
    #convert to pandas dataframe
    df = fp_obj.to_pandas()
    return df

def prepare_input(raw_input):
    input_ids_ragged_tensor = tf.ragged.constant([ast.literal_eval(x) for x in np.array(raw_input.input_ids)])
    input_mask_ragged_tensor = tf.ragged.constant([ast.literal_eval(x) for x in np.array(raw_input.input_mask)])
    input_segment_ids_ragged_tensor = tf.ragged.constant([ast.literal_eval(x) for x in np.array(raw_input.segment_ids)])

    input = {
        'input_word_ids' : input_ids_ragged_tensor.to_tensor(),
        'input_mask' : input_mask_ragged_tensor.to_tensor(),
        'input_type_ids' : input_segment_ids_ragged_tensor.to_tensor()
    }
    return input


def prepare_dataset(raw_df):
    train_raw_df = raw_df.loc[df['split_type'] == "train"]
    validation_raw_df = raw_df.loc[df['split_type'] == "validation"]
    train_input = prepare_input(train_raw_df)
    validation_input = prepare_input(validation_raw_df)
    train_labels = train_raw_df.copy().pop("contain_violence")
    validation_labels = validation_raw_df.copy().pop("contain_violence")
    return train_input, train_labels, validation_input, validation_labels

if __name__ == "__main__":
    raw_df = load_data_as_data_frame()
    train_input, train_labels, validation_input, validation_labels = prepare_dataset(raw_df)
    model = create_model()
    epochs = 4
    history = model.fit(train_input, train_labels, validation_data=(validation_input, validation_labels),epochs = epochs,verbose = 2)

