import tensorflow as tf
from src.models import inception_resnet_v1

training_checkpoint = "models/20180402-114759/model-20180402-114759.ckpt-275"
id = training_checkpoint.split("/")[-2]
print(id)
eval_checkpoint = f"{id}-saved/ckpt"

with tf.Session() as sess:
    input = tf.placeholder(name="input", dtype=tf.float32, shape=[None, 160, 160, 3])
    prelogits, _ = inception_resnet_v1.inference(
        input, keep_probability=0.8, phase_train=False, bottleneck_layer_size=512
    )
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name="embeddings")
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, training_checkpoint)
    save_path = saver.save(sess, eval_checkpoint)
    print("Model saved in file: %s" % save_path)

    print("convert to onnx by running:")
    print(
        f"python -m tf2onnx.convert --checkpoint {eval_checkpoint}.meta --output {id}.onnx --inputs input:0 --outputs embeddings:0 --opset 11 --inputs-as-nchw input:0"
)
    os.system(f"python -m tf2onnx.convert --checkpoint {eval_checkpoint}.meta --output {id}.onnx --inputs input:0 --outputs embeddings:0 --opset 11 --inputs-as-nchw input:0")
