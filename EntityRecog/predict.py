import tensorflow as tf
from .joint_m_models import BiLstmCRF
from .hyperparams import HyperParams as hp
from .utils import get_entity, decode, get_class
import pickle
tf.reset_default_graph()

model = BiLstmCRF(hp, is_training=True)
ckpt = tf.train.get_checkpoint_state(hp.checkpoint_dir)
sess = tf.Session()
model.saver.restore(sess, ckpt.model_checkpoint_path)


def predict(text):

    print('============= predict =============')
    demo_sent = text

    path = hp.vocab_dir
    with open(path, "rb") as f:
        to_idx_map, id_to_token = pickle.load(f)
    max_length = hp.max_length
    chars = [to_idx_map.get(char, to_idx_map.get("_UNK_")) for char in demo_sent.strip()]
    chars = chars[:max_length]
    chars = [chars + [0] * (hp.max_length - len(chars))]
    labels = [[0] * hp.max_length]
    categories = [0, 0, 0, 0, 0, 0]

    feed_dict = {model.inputs: chars,
                 model.targets: labels,
                 model.categories: categories,
                 model.keep_prob: 1}
    logits, accuracy, length, trans, classify_outputs = sess.run([model.logits,
                                                model.accuracy,
                                                model.length,
                                                model.trans,
                                                model.y_pred_cls], feed_dict=feed_dict)
    predict = decode(logits, length, trans)
    classify_output = int(classify_outputs)
    class_output = get_class(classify_output)
    # print("类型：{}\n".format(class_output))
    POS, DIR, TIME, DIS, RSCT, HWN, HWNB = get_entity(demo_sent.strip(), predict[0], hp.label_map)
    # print("位置: {}\n方向: {}\n时间: {}\n距离: {}\n路段: {}\n高速名称: {}\n高速编号: {}".format(POS, DIR, TIME, DIS, RSCT, HWN, HWNB))

    return  POS, DIR, TIME, DIS, RSCT, HWN, HWNB, class_output


