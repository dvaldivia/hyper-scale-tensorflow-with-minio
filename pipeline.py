# Copyright MinIO 2022
import kfp
import kfp.components as comp
import requests
from kfp.compiler import Compiler
from kfp.components import InputPath, OutputPath
from kfp.dsl import pipeline


@pipeline(
    name='Pre-Process Data',
)
def preprocess_data(file: str,
                    output_text_path: comp.OutputPath(str)):
    from minio import Minio, S3Error
    import tarfile
    import os
    import math
    import tensorflow as tf

    # Config Paramters
    minio_address = "minio.ns-1.svc.cluster.local"
    minio_access_key = "kubeflow"
    minio_secret_key = "kubeflow123"
    datasets_bucket = "datasets"
    preprocessed_data_folder = "preprocessed-data"
    tf_record_file_size = 500

    minioClient = Minio(minio_address,
                        access_key=minio_access_key,
                        secret_key=minio_secret_key,
                        secure=False)
    # Download data
    try:
        minioClient.fget_object(
            datasets_bucket,
            "aclImdb_v1.tar.gz",
            "/tmp/dataset.tar.gz")
    except S3Error as err:
        print(err)

    # Extract data
    extract_folder = f"/tmp/{datasets_bucket}/"

    with tarfile.open("/tmp/dataset.tar.gz", "r:gz") as tar:
        tar.extractall(path=extract_folder)

    # Load and structure the data
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

    # Since we encode the data using the Universal Sentence Encoder model let's download it
    try:
        minioClient.fget_object(
            datasets_bucket,
            "models/universal-sentence-encoder_4.tar.gz",
            "/tmp/universal-sentence-encoder_4.tar.gz")
    except S3Error as err:
        print(err)
    se_model_prefix = "universal-sentence-encoder/4"
    extract_folder = f"/tmp/{se_model_prefix}/"

    with tarfile.open("/tmp/universal-sentence-encoder_4.tar.gz", "r:gz") as tar:
        tar.extractall(path=extract_folder)
    embed = tf.saved_model.load(extract_folder)

    def _embedded_sentence_feature(value):
        # convert tensor to list of float values
        input = value.numpy().ravel().tolist()
        return tf.train.Feature(float_list=tf.train.FloatList(value=input))

    def _label_feature(value):
        # convert tensor to list of float values
        input = value.numpy().ravel().tolist()
        return tf.train.Feature(int64_list=tf.train.Int64List(value=input))

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
        print(f"Process examples for prefix: {prefix}")
        import timeit
        starttime = timeit.default_timer()
        total_training = len(records)
        print(f"Total of {total_training} elements")
        total_batches = math.floor(total_training / tf_record_file_size)
        if total_training % tf_record_file_size != 0:
            total_batches += 1
        print(f"Total of {total_batches} files of {tf_record_file_size} records - {timeit.default_timer() - starttime}")

        counter = 0
        file_counter = 0
        buffer = []
        file_list = []
        for i in range(total_training):
            counter += 1
            sentence_embedding = embed([records[i]["text"]])
            label_encoded = encode_label(records[i]["label"])
            record = serialize_example(label_encoded, sentence_embedding)
            buffer.append(record)

            if len(buffer) >= tf_record_file_size:
                # save this buffer of examples as a file to MinIO
                counter = 0
                file_counter += 1
                file_name = f"{prefix}_file{file_counter}.tfrecord"
                with open(file_name, "w+") as f:
                    with tf.io.TFRecordWriter(f.name, options="GZIP") as writer:
                        for example in buffer:
                            writer.write(example.SerializeToString())

                try:
                    minioClient.fput_object(datasets_bucket, f"{preprocessed_data_folder}/{file_name}", file_name)
                except S3Error as err:
                    print(err)
                file_list.append(file_name)
                os.remove(file_name)
                buffer = []
                print(f"Done with batch {file_counter}/{total_batches} - {timeit.default_timer() - starttime}")
        print("")
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
    with open(output_text_path, 'w') as writer:
        writer.write("done!")

    print("Done!")


@pipeline(
    name='Training',
)
def train(train_path: InputPath(),
          output_text_path: OutputPath(str)):
    from minio import Minio
    import os
    import math
    import tensorflow as tf
    from tensorflow import keras

    # Config Paramters
    minio_address = "minio.ns-1.svc.cluster.local"
    minio_access_key = "kubeflow"
    minio_secret_key = "kubeflow123"
    datasets_bucket = "datasets"
    preprocessed_data_folder = "preprocessed-data"
    tf_record_file_size = 500

    minioClient = Minio(minio_address,
                        access_key=minio_access_key,
                        secret_key=minio_secret_key,
                        secure=False)

    # List all training tfrecord files
    objects = minioClient.list_objects(datasets_bucket, prefix=f"{preprocessed_data_folder}/train")
    training_files_list = []
    for obj in objects:
        training_files_list.append(obj.object_name)

    # Configure TF to use MinIO
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["AWS_REGION"] = "us-east-1"
    os.environ["S3_ENDPOINT"] = minio_address
    os.environ["S3_USE_HTTPS"] = "0"
    os.environ["S3_VERIFY_SSL"] = "0"

    all_training_filenames = [f"s3://datasets/{f}" for f in training_files_list]

    total_train_data_files = math.floor(len(all_training_filenames) * 0.9)
    if total_train_data_files == len(all_training_filenames):
        total_train_data_files -= 1
    training_files = all_training_filenames[0:total_train_data_files]
    validation_files = all_training_filenames[total_train_data_files:]

    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(training_files, num_parallel_reads=AUTO, compression_type="GZIP")
    dataset = dataset.with_options(ignore_order)

    validation = tf.data.TFRecordDataset(validation_files, num_parallel_reads=AUTO, compression_type="GZIP")
    validation = validation.with_options(ignore_order)

    def decode_fn(record_bytes):
        schema = {
            "label": tf.io.FixedLenFeature([2], dtype=tf.int64),
            "sentence": tf.io.FixedLenFeature([512], dtype=tf.float32),
        }

        tf_example = tf.io.parse_single_example(record_bytes, schema)
        new_shape = tf.reshape(tf_example["sentence"], [1, 512])
        label = tf.reshape(tf_example["label"], [1, 2])
        return new_shape, label

    # Build model
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

    model.summary()

    mapped_ds = dataset.map(decode_fn)
    mapped_ds = mapped_ds.repeat(5)
    mapped_ds = mapped_ds.batch(128)

    mapped_validation = validation.map(decode_fn)
    mapped_validation = mapped_validation.repeat(5)
    mapped_validation = mapped_validation.batch(128)

    checkpoint_path = f"s3://{datasets_bucket}/checkpoints/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    from datetime import datetime
    model_note = "256"
    logdir = f"s3://{datasets_bucket}/logs/imdb/{model_note}-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    history = model.fit(
        mapped_ds,
        epochs=10,
        callbacks=[cp_callback, tensorboard_callback],
        validation_data=mapped_validation,
    )

    model_destination = f"s3://{datasets_bucket}/imdb_sentiment_analysis/1"
    model.save(model_destination)
    with open(output_text_path, 'w') as writer:
        writer.write("done training!")
    print("Done!")


@pipeline(
    name='Evaluate Model',
)
def evaluate(train_path: InputPath(),
             output_text_path: OutputPath(str)):
    import tensorflow as tf
    from minio import Minio
    from tensorflow import keras
    import os

    # Config Paramters
    minio_address = "minio.ns-1.svc.cluster.local"
    minio_access_key = "kubeflow"
    minio_secret_key = "kubeflow123"
    datasets_bucket = "datasets"

    preprocessed_data_folder = "preprocessed-data"
    tf_record_file_size = 500

    # Configure TF to use MinIO
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["AWS_REGION"] = "us-east-1"
    os.environ["S3_ENDPOINT"] = minio_address
    os.environ["S3_USE_HTTPS"] = "0"
    os.environ["S3_VERIFY_SSL"] = "0"

    minioClient = Minio(minio_address,
                        access_key=minio_access_key,
                        secret_key=minio_secret_key,
                        secure=False)

    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    # List all testing tfrecord files
    objects = minioClient.list_objects(datasets_bucket, prefix=f"{preprocessed_data_folder}/test")
    testing_files_list = []
    for obj in objects:
        testing_files_list.append(obj.object_name)

    testing_filenames = [f"s3://datasets/{f}" for f in testing_files_list]

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

    testing_mapped_ds = testing_dataset.map(decode_fn)
    testing_mapped_ds = testing_mapped_ds.repeat(5)
    testing_mapped_ds = testing_mapped_ds.batch(128)

    # load model
    model_destination = f"s3://{datasets_bucket}/imdb_sentiment_analysis/1"

    model = keras.models.load_model(model_destination)

    testing = model.evaluate(testing_mapped_ds)

    print(testing)

    with open(output_text_path, 'w') as writer:
        writer.write("done eval!")


@pipeline(
    name='Deploy Model',
)
def serve(train_path: InputPath(),
          output_text_path: OutputPath(str)):
    print("Deploy")
    from kubernetes import client, config

    with open(output_text_path, 'w') as writer:
        writer.write("done training!")


preprocess_data_op = kfp.components.create_component_from_func(
    func=preprocess_data,
    output_component_file='component-pre-process.yaml',  # This is optional. It saves the component spec for future use.
    base_image='python:3.7',
    packages_to_install=[
        'minio==7.1.10',
        'tensorflow==2.2.3',
        'protobuf==3.20.0',
    ])
train_op = kfp.components.create_component_from_func(
    func=train,
    output_component_file='component-train.yaml',  # This is optional. It saves the component spec for future use.
    base_image='python:3.7',
    packages_to_install=[
        'minio==7.1.10',
        'tensorflow==2.2.3',
        'protobuf==3.20.0',
    ])
evaluate_op = kfp.components.create_component_from_func(
    func=evaluate,
    output_component_file='component-evaluate.yaml',  # This is optional. It saves the component spec for future use.
    base_image='python:3.7',
    packages_to_install=[
        'minio==7.1.10',
        'tensorflow==2.2.3',
        'protobuf==3.20.0',
        'matplotlib',
        'numpy',
    ])
serve_op = kfp.components.create_component_from_func(
    func=serve,
    output_component_file='component-serve.yaml',  # This is optional. It saves the component spec for future use.
    base_image='python:3.7',
    packages_to_install=[
        'minio==7.1.10',
        'tensorflow==2.2.3',
        'protobuf==3.20.0',
        'kubernetes',
    ])


# Define a pipeline and create a task from a component:
def my_pipeline(url):
    preprocess_task = preprocess_data_op(file="train")
    print(train_op.__code__.co_varnames)
    train_task = train_op(train=preprocess_task.output)
    evaluate_task = evaluate_op(train=train_task.output)
    serve_task = serve_op(train=evaluate_task.output)
    # The outputs of the merge_csv_task can be referenced using the
    # merge_csv_task.outputs dictionary: merge_csv_task.outputs['output_csv']


def run():
    # kfp.compiler.Compiler().compile(
    #     pipeline_func=my_pipeline,
    #     package_path='pipeline-v3.yaml')

    # HOST = "https://kubeflow.tes.dc.min.dev"
    HOST = "http://kubeflow.lulu"
    # HOST = "http://localhost:3000"
    # AUTH_HOST = "https://kubeflow.lulu"
    USERNAME = "user@example.com"
    PASSWORD = "12341234"
    NAMESPACE = "kubeflow-user-example-com"

    import urllib3
    urllib3.disable_warnings()

    session = requests.Session()
    response = session.get(HOST, verify=False)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {"login": USERNAME, "password": PASSWORD}
    session_res = session.post(response.url, headers=headers, data=data, verify=False)
    print(session_res.status_code)
    print(session.cookies.get_dict())
    session_cookie = session.cookies.get_dict()["authservice_session"]
    print(session_cookie)

    client = kfp.Client(
        cookies=f"authservice_session={session_cookie}",
        host=f"{HOST}/pipeline",
        namespace=NAMESPACE,
    )
    print("-----")
    print(client.list_experiments())
    x = client.create_run_from_pipeline_func(
        my_pipeline,
        run_name="sentiment_analysis_model",
        namespace=NAMESPACE,
        arguments={
            'url': 'https://storage.googleapis.com/ml-pipeline-playground/iris-csv-files.tar.gz'
        })
    print(x)

def compile():
    Compiler().compile(my_pipeline, 'sentiment-analysis-object-storage.yaml')


if __name__ == "__main__":
    run()
    # compile()
