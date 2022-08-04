# Copyright MinIO 2020
import math
import os
import random
import tarfile
import timeit
from datetime import datetime

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io
from minio import Minio
from minio.error import S3Error
from tensorflow import keras

random_seed = 44
batch_size = 128
datasets_bucket = "datasets"
preprocessed_data_folder = "preprocessed-data"
tf_record_file_size = 500
# How to access MinIO
minio_address = "localhost:9000"
minio_access_key = "minioadmin"
minio_secret_key = "minioadmin"

minioClient = Minio(minio_address,
                    access_key=minio_access_key,
                    secret_key=minio_secret_key,
                    secure=False)

try:
    minioClient.fget_object(
        datasets_bucket,
        "aclImdb_v1.tar.gz",
        "/tmp/dataset.tar.gz")
except S3Error as err:
    print(err)

extract_folder = f"/tmp/{datasets_bucket}/"

with tarfile.open("/tmp/dataset.tar.gz", "r:gz") as tar:
    tar.extractall(path=extract_folder)

# Pre-Processing

train = []
test = []

dirs_to_read = [
    "aclImdb/train/pos",
    "aclImdb/train/neg",
    "aclImdb/test/pos",
    "aclImdb/test/neg",
]

for dir_name in dirs_to_read:
    parts = dir_name.split("/")
    dataset = parts[1]
    label = parts[2]
    for filename in os.listdir(os.path.join(extract_folder, dir_name)):
        with open(os.path.join(extract_folder, dir_name, filename), "r") as f:
            content = f.read()
            if dataset == "train":
                train.append({
                    "text": content,
                    "label": label
                })
            elif dataset == "test":
                test.append({
                    "text": content,
                    "label": label
                })

random.Random(random_seed).shuffle(train)
random.Random(random_seed).shuffle(test)

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")


def _embedded_sentence_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _label_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def encode_label(label):
    if label == "pos":
        return tf.constant([1, 0])
    elif label == "neg":
        return tf.constant([0, 1])


def serialize_example(label, sentence_tensor):
    feature = {
        "sentence": _embedded_sentence_feature(sentence_tensor[0]),
        "label": _label_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto


def process_examples(records, prefix=""):
    starttime = timeit.default_timer()
    total_training = len(records)
    print(f"Total of {total_training} elements")
    total_batches = math.floor(total_training / tf_record_file_size)
    if total_training % tf_record_file_size != 0:
        total_batches += 1
    print(f"Total of {total_batches} files of {tf_record_file_size} records")

    counter = 0
    file_counter = 0
    buffer = []
    file_list = []
    for i in range(len(records)):
        counter += 1

        sentence_embedding = embed([records[i]["text"]])
        label_encoded = encode_label(records[i]["label"])
        record = serialize_example(label_encoded, sentence_embedding)
        buffer.append(record)

        if counter >= tf_record_file_size:
            print(f"Records in buffer {len(buffer)}")
            # save this buffer of examples as a file to MinIO
            counter = 0
            file_counter += 1
            file_name = f"{prefix}_file{file_counter}.tfrecord"
            with open(file_name, "w+") as f:
                with tf.io.TFRecordWriter(f.name, options="GZIP") as writer:
                    for example in buffer:
                        writer.write(example.SerializeToString())
            print(f"file size {os.stat(file_name).st_size}")
            try:
                minioClient.fput_object(datasets_bucket, f"{preprocessed_data_folder}/{file_name}", file_name)
            except S3Error as err:
                print(err)
            file_list.append(file_name)
            os.remove(file_name)
            buffer = []
            print(f"Done with batch {file_counter}/{total_batches} - {timeit.default_timer() - starttime}")
    if len(buffer) > 0:
        file_counter += 1
        file_name = f"file{file_counter}.tfrecord"
        with open(file_name, "w+") as f:
            with tf.io.TFRecordWriter(f.name) as writer:
                for example in buffer:
                    writer.write(example.SerializeToString())
        try:
            minioClient.fput_object(datasets_bucket, f"{preprocessed_data_folder}/{file_name}", file_name)
        except S3Error as err:
            print(err)
        file_list.append(file_name)
        os.remove(file_name)
        buffer = []
    print("total time is :", timeit.default_timer() - starttime)
    return file_list


process_examples(train, prefix="train")
process_examples(test, prefix="test")
print("Done!")

# List all training tfrecord files
objects = minioClient.list_objects_v2(datasets_bucket, prefix="train")
training_files_list = []
for obj in objects:
    training_files_list.append(obj.object_name)

# List all testing tfrecord files
objects = minioClient.list_objects_v2(datasets_bucket, prefix="test")
testing_files_list = []
for obj in objects:
    testing_files_list.append(obj.object_name)

os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
os.environ["AWS_REGION"] = "us-east-1"
os.environ["S3_ENDPOINT"] = minio_address
os.environ["S3_USE_HTTPS"] = "0"
os.environ["S3_VERIFY_SSL"] = "0"

all_training_filenames = [f"s3://datasets/{f}" for f in training_files_list]
testing_filenames = [f"s3://datasets/{f}" for f in testing_files_list]

total_train_data_files = math.floor(len(all_training_filenames) * 0.9)
if total_train_data_files == len(all_training_filenames):
    total_train_data_files -= 1
training_files = all_training_filenames[0:total_train_data_files]
validation_files = all_training_filenames[total_train_data_files:]

# Now let's create the `tf.data` datasets:

AUTO = tf.data.experimental.AUTOTUNE
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False

dataset = tf.data.TFRecordDataset(training_files, num_parallel_reads=AUTO, compression_type="GZIP")
dataset = dataset.with_options(ignore_order)

validation = tf.data.TFRecordDataset(validation_files, num_parallel_reads=AUTO, compression_type="GZIP")
validation = validation.with_options(ignore_order)

testing_dataset = tf.data.TFRecordDataset(testing_filenames, num_parallel_reads=AUTO, compression_type="GZIP")
testing_dataset = testing_dataset.with_options(ignore_order)


def decode_fn(record_bytes):
    schema = {
        "label": tf.io.FixedLenFeature([2], dtype=tf.int64),
        "sentence": tf.io.FixedLenFeature([512], dtype=tf.float32),
    }

    tf_example = tf.io.parse_single_example(record_bytes, schema)
    new_shape = tf.reshape(tf_example["sentence"], [1, 512])
    label = tf.reshape(tf_example["label"], [1, 2])
    return new_shape, label


model = keras.Sequential()

model.add(
    keras.layers.Dense(
        units=256,
        input_shape=(1, 512),
        activation="relu"
    )
)
model.add(
    keras.layers.Dropout(rate=0.5)
)

model.add(
    keras.layers.Dense(
        units=16,
        activation="relu"
    )
)
model.add(
    keras.layers.Dropout(rate=0.5)
)

model.add(keras.layers.Dense(2, activation="softmax"))
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(0.001),
    metrics=["accuracy"]
)

# ![Structure of our Deep Learning model](pic2.png)


model.summary()

# Let's prepare our datasets for the training stage by having them repeat
# themselves a little and batch `128` items at a time


mapped_ds = dataset.map(decode_fn)
mapped_ds = mapped_ds.repeat(5)
mapped_ds = mapped_ds.batch(128)

mapped_validation = validation.map(decode_fn)
mapped_validation = mapped_validation.repeat(5)
mapped_validation = mapped_validation.batch(128)

testing_mapped_ds = testing_dataset.map(decode_fn)
testing_mapped_ds = testing_mapped_ds.repeat(5)
testing_mapped_ds = testing_mapped_ds.batch(128)

checkpoint_path = f"s3://{datasets_bucket}/checkpoints/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model_note = "256"
logdir = f"s3://{datasets_bucket}/logs/imdb/{model_note}-" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Finally we will train the model:


history = model.fit(
    mapped_ds,
    epochs=10,
    callbacks=[cp_callback, tensorboard_callback],
    validation_data=mapped_validation,
)

# Now that we have our model, we want to save it to MinIO
model.save(f"s3://{datasets_bucket}/imdb_sentiment_analysis")

testing = model.evaluate(testing_mapped_ds)

samples = [
    "This movie sucks",
    "This was extremely good, I loved it.",
    "great acting",
    "terrible acting",
    "pure kahoot",
    "I don't know what's the point of this movie, this movie sucks but the acting is great",
    "This is not a good movie",
]
sample_embedded = embed(samples)

res = model.predict(sample_embedded)
for s in range(len(samples)):
    if res[s][0] > res[s][1]:
        print(f"{samples[s]} - positive")
    else:
        print(f"{samples[s]} - negative")
