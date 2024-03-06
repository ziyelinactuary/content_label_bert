import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow import keras
import collections
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import time
from datetime import datetime, timezone
import boto3
import functools
import multiprocessing
import sagemaker
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)
from official.nlp.bert import tokenization
from official.nlp.data import classifier_data_lib
import argparse

from src.s3_client import S3Client

bert_model_name="Hum-Works/lodestone-base-4096-v1"
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable = True )
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

# tokenizer = BertTokenizer.from_pretrained(bert_model_name)
region = "us-east-1"
featurestore_runtime = boto3.Session(region_name=region).client(
    service_name="sagemaker-featurestore-runtime", region_name=region
)
sm = boto3.Session(region_name=region).client(service_name="sagemaker", region_name=region)
s3 = boto3.Session(region_name=region).client(service_name="s3", region_name=region)
s3_client = S3Client()

sagemaker_session = sagemaker.Session(
    boto_session=boto3.Session(region_name=region),
    sagemaker_client=sm,
    sagemaker_featurestore_runtime_client=featurestore_runtime,
)
sts = boto3.Session(region_name=region).client(service_name="sts", region_name=region)
caller_identity = sts.get_caller_identity()
print("caller_identity: {}".format(caller_identity))

assumed_role_arn = caller_identity["Arn"]
print("(assumed_role) caller_identity_arn: {}".format(assumed_role_arn))

assumed_role_name = assumed_role_arn.split("/")[-2]

iam = boto3.Session(region_name=region).client(service_name="iam", region_name=region)
get_role_response = iam.get_role(RoleName=assumed_role_name)
print("get_role_response {}".format(get_role_response))
role = get_role_response["Role"]["Arn"]
bucket = "content-descriptor-prediction-training"
feature_group_prefix = "feature_group"
def list_arg(raw_value):
    """argparse type for a list of strings"""
    return str(raw_value).split(",")

class InputFeatures(object):
    """BERT feature vectors."""

    def __init__(self, input_ids, input_mask, segment_ids, date, asin, contain_violence):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.date = date
        self.asin = asin
        self.contain_violence = contain_violence

def wait_for_feature_group_creation_complete(feature_group):
    try:
        status = feature_group.describe().get("FeatureGroupStatus")
        print("Feature Group status: {}".format(status))
        while status == "Creating":
            print("Waiting for Feature Group Creation")
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
            print("Feature Group status: {}".format(status))
        if status != "Created":
            print("Feature Group status: {}".format(status))
            raise RuntimeError(f"Failed to create feature group {feature_group.name}")
        print(f"FeatureGroup {feature_group.name} successfully created.")
    except:
        print("No feature group created yet.")

def create_or_load_feature_group(prefix, feature_group_name):

    # Feature Definitions for our records
    feature_definitions = [
        FeatureDefinition(feature_name="input_ids", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="input_mask", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="segment_ids", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="date", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="asin", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="contain_violence", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="split_type", feature_type=FeatureTypeEnum.STRING),
    ]

    feature_group = FeatureGroup(
        name=feature_group_name, feature_definitions=feature_definitions, sagemaker_session=sagemaker_session
    )

    print("Feature Group: {}".format(feature_group))

    try:
        print(
            "Waiting for existing Feature Group to become available if it is being created by another instance in our cluster..."
        )
        wait_for_feature_group_creation_complete(feature_group)
    except Exception as e:
        print("Before CREATE FG wait exeption: {}".format(e))
    #        pass

    try:
        record_identifier_feature_name = "asin"
        event_time_feature_name = "date"

        print("Creating Feature Group with role {}...".format(role))
        feature_group.create(
            s3_uri=f"s3://{bucket}/{feature_group_prefix}",
            record_identifier_name=record_identifier_feature_name,
            event_time_feature_name=event_time_feature_name,
            role_arn=role,
            enable_online_store=False,
        )
        print("Creating Feature Group. Completed.")

        print("Waiting for new Feature Group to become available...")
        wait_for_feature_group_creation_complete(feature_group)
        print("Feature Group available.")
        feature_group.describe()

    except Exception as e:
        print("Exception: {}".format(e))

    return feature_group

def transform_inputs_to_tfrecord(inputs, output_file, max_seq_length):
    records = []
    tf_record_writer = tf.io.TFRecordWriter(output_file)

    inputs_df = inputs.reset_index()  # make sure indexes pair with number of rows

    for input_idx, row in inputs_df.iterrows():
        if input_idx % 10000 == 0:
            print("Writing input {} of {}\n".format(input_idx, len(inputs)))
        features = convert_input(row, max_seq_length)
        all_features = collections.OrderedDict()
        all_features["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.input_ids))
        all_features["input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.input_mask))
        all_features["segment_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.segment_ids))
        all_features["contain_violence"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[features.contain_violence]))

        tf_record = tf.train.Example(features=tf.train.Features(feature=all_features))
        tf_record_writer.write(tf_record.SerializeToString())

        records.append(
            {  #'tf_record': tf_record.SerializeToString(),
                "input_ids": features.input_ids,
                "input_mask": features.input_mask,
                "segment_ids": features.segment_ids,
                "asin": row.asin,
                "date": row.date,
                "contain_violence": row.contain_violence,
            }
        )

    tf_record_writer.close()

    return records

def convert_input(the_input, max_seq_length):
    example = classifier_data_lib.InputExample(guid=the_input.asin,text_a = the_input.synopsis,label = the_input.contain_violence)
    feature = classifier_data_lib.convert_single_example(
        0,
        example,
        [the_input.contain_violence],
        max_seq_length,
        tokenizer
    )

    segment_ids = [0] * max_seq_length
    features = InputFeatures(
        input_ids=feature.input_ids,
        input_mask=feature.input_mask,
        segment_ids=feature.segment_ids,
        date = the_input.date,
        asin=the_input.asin.encode('utf-8'),
        contain_violence=feature.label_id,
    )

    return features

def process(args):

    feature_group_name = args.feature_group_name
    feature_store_offline_prefix = args.feature_store_offline_prefix

    feature_group = create_or_load_feature_group(
        prefix=feature_store_offline_prefix, feature_group_name=feature_group_name
    )

    print(feature_group.as_hive_ddl())


def cast_object_to_string(data_frame):
    for label in data_frame.columns:
        if data_frame.dtypes[label] == "object":
            data_frame[label] = data_frame[label].astype("str").astype("string")
    return data_frame

def _transform_tsv_to_tfrecord(parquet_index, args, max_seq_length, balance_dataset):
    feature_group = create_or_load_feature_group(args.feature_store_offline_prefix, args.feature_group_name)
    df = s3_client.s3_to_dataframe(bucket_name="content-descriptor-prediction-training",s3_key=f"iad_training_{parquet_index+10}.parquet.gz")

    df.dropna(subset=['synopsis'], inplace=True)
    df['contain_violence'] = df.apply(lambda x: 1 if 'vcc_cd_violence_contain' in x.cd_labels.tolist() else 0, axis=1)
    timestamp = datetime.now().replace(tzinfo=timezone.utc).isoformat()
    df['date'] = timestamp
    holdout_percentage = 0.2 #arguments
    df_train, df_holdout = train_test_split(df, test_size=holdout_percentage, stratify=df['contain_violence'])
    test_holdout_percentage = 0.3 #arguments
    df_validation, df_test = train_test_split(
        df_holdout, test_size=test_holdout_percentage, stratify=df_holdout['contain_violence'])
    df_train = df_train.reset_index(drop=True)
    df_validation = df_validation.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print("Shape of train dataframe {}".format(df_train.shape))
    print("Shape of validation dataframe {}".format(df_validation.shape))
    print("Shape of test dataframe {}".format(df_test.shape))

    output_data="output"
    train_data = "{}/bert/train".format(output_data)
    validation_data = "{}/bert/validation".format(output_data)
    test_data = "{}/bert/test".format(output_data)

    # Convert our train and validation features to InputFeatures (.tfrecord protobuf) that works with BERT and TensorFlow.
    current_host="host1" #arguments
    filename_without_extension = "TetmpOutputFeature" #arguments

    train_records = transform_inputs_to_tfrecord(
        df_train,
        # "{}/part-{}-{}.tfrecord".format(train_data, current_host, filename_without_extension),
        "output/train-{}.tfrecord".format(filename_without_extension),
        max_seq_length,
    )

    validation_records = transform_inputs_to_tfrecord(
        df_validation,
        "output/validation-{}.tfrecord".format(filename_without_extension),
        max_seq_length,
    )

    test_records = transform_inputs_to_tfrecord(
        df_test,
        "output/test-{}.tfrecord".format(filename_without_extension),
        max_seq_length,
    )

    df_train_records = pd.DataFrame.from_dict(train_records)
    df_train_records["split_type"] = "train"
    df_train_records.head()

    df_validation_records = pd.DataFrame.from_dict(validation_records)
    df_validation_records["split_type"] = "validation"
    df_validation_records.head()

    df_test_records = pd.DataFrame.from_dict(test_records)
    df_test_records["split_type"] = "test"
    df_test_records.head()

    # Add record to feature store
    df_fs_train_records = cast_object_to_string(df_train_records)
    df_fs_validation_records = cast_object_to_string(df_validation_records)
    df_fs_test_records = cast_object_to_string(df_test_records)


    print("Ingesting Features...")
    feature_group.ingest(data_frame=df_fs_train_records, max_workers=3, wait=True)
    feature_group.ingest(data_frame=df_fs_validation_records, max_workers=3, wait=True)
    feature_group.ingest(data_frame=df_fs_test_records, max_workers=3, wait=True)

    offline_store_status = None
    while offline_store_status != 'Active':
        try:
            offline_store_status = feature_group.describe()['OfflineStoreStatus']['Status']
        except:
            pass
        print('Offline store status: {}'.format(offline_store_status))
    print('...features ingested!')

def parse_args():
    # Unlike SageMaker training jobs (which have `SM_HOSTS` and `SM_CURRENT_HOST` env vars), processing jobs to need to parse the resource config file directly
    resconfig = {}
    try:
        with open("/opt/ml/config/resourceconfig.json", "r") as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print("/opt/ml/config/resourceconfig.json not found.  current_host is unknown.")
        pass  # Ignore

    # Local testing with CLI args
    parser = argparse.ArgumentParser(description="Process")

    parser.add_argument(
        "--hosts",
        type=list_arg,
        default=resconfig.get("hosts", ["unknown"]),
        help="Comma-separated list of host names running the job",
    )
    parser.add_argument(
        "--current-host",
        type=str,
        default=resconfig.get("current_host", "unknown"),
        help="Name of this host running the job",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="/opt/ml/processing/input/data",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="/opt/ml/processing/output",
    )
    parser.add_argument(
        "--train-split-percentage",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--validation-split-percentage",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--test-split-percentage",
        type=float,
        default=0.05,
    )
    parser.add_argument("--balance-dataset", type=eval, default=True)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--feature-store-offline-prefix",
        type=str,
        default="offline_process_1",
    )
    parser.add_argument(
        "--feature-group-name",
        type=str,
        default="content_metadata",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    feature_group_name = "content_metadata" #arguments
    feature_store_offline_prefix = "offline_process_1"
    process(args)
    input_dfs = []
    max_seq_length = 400 #arguments
    balance_dataset = False
    transform_tsv_to_tfrecord = functools.partial(
        _transform_tsv_to_tfrecord,
        args=args,
        max_seq_length=max_seq_length,
        balance_dataset=balance_dataset
    )
    num_cpus = multiprocessing.cpu_count()
    print("num_cpus {}".format(num_cpus))
    input_files = list(range(15, 20))
    p = multiprocessing.Pool(num_cpus)
    p.map(transform_tsv_to_tfrecord, input_files)

    offline_store_contents = None
    while offline_store_contents is None:
        objects_in_bucket = s3.list_objects(Bucket=bucket, Prefix=feature_store_offline_prefix)
        if "Contents" in objects_in_bucket and len(objects_in_bucket["Contents"]) > 1:
            offline_store_contents = objects_in_bucket["Contents"]
        else:
            print("Waiting for data in offline store...\n")
            sleep(60)

    print("Data available.")

