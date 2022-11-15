""" Build a regression model using tf.estimator API
"""
from absl import flags
from custom_dataset import load_dataset
from custom_models import baseline, linear_reg, nn, cnn
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_float("threshold",
                   default=10.,
                   help="Energy threshold.")
flags.DEFINE_float("learning_rate",
                   default=0.001,
                   help="Initial learning rate.")
flags.DEFINE_integer("batch_size",
                     default=32,
                     help="Batch size.")
flags.DEFINE_integer("steps",
                     default=100,
                     help="Number of steps.")
flags.DEFINE_integer("checkpoints",
                     default=20,
                     help="Number of checkpoints.")
flags.DEFINE_string("model",
                    default="baseline",
                    help="Select model: linear_reg, nn or cnn")

FLAGS = flags.FLAGS


def model_fn(features, labels, mode):
    """ Model function
    """
    inputs = features['x']

    with tf.variable_scope("model"):
        with tf.variable_scope("logits"):
            logits = eval(f'{FLAGS.model}(inputs)')

        predictions = tf.squeeze(logits, axis=1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)

        average_loss = tf.losses.mean_squared_error(labels, predictions)
        batch_size = tf.shape(labels)[0]
        total_loss = tf.to_float(batch_size) * average_loss

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=tf.train.AdamOptimizer(
                    learning_rate=FLAGS.learning_rate).minimize(
                        loss=average_loss,
                        global_step=tf.train.get_global_step()))

        rmse = tf.metrics.root_mean_squared_error(labels, predictions)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops={'rmse': rmse})


def main(_):
    """ Load dataset.
        Execute custom estimator.
    """
    dataset = load_dataset(FLAGS.threshold)
    runconf = tf.estimator.RunConfig(tf_random_seed=42)
    model_dir = f'./results{int(FLAGS.threshold)}/{FLAGS.model}'

    estimator = tf.estimator.Estimator(model_fn, model_dir, runconf)

    for i in range(FLAGS.checkpoints):
        print(f'\nCheckpoint {i+1}')
        estimator.train(
            input_fn=tf.estimator.inputs.numpy_input_fn(
                x={'x': dataset.train.images},
                y=dataset.train.labels,
                batch_size=FLAGS.batch_size,
                num_epochs=None,
                shuffle=False),
            steps=FLAGS.steps)
        eval_results = estimator.evaluate(
            input_fn=tf.estimator.inputs.numpy_input_fn(
                x={'x': dataset.validation.images},
                y=dataset.validation.labels,
                num_epochs=1,
                shuffle=False))
        print(eval_results)


if __name__ == '__main__':
    tf.app.run()
