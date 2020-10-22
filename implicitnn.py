import tensorflow as tf
from utils.tfoperations import act_func_dict


def create_implicit_concatinput_hidden(xinput, target_input, actfunc, n_hidden1, n_hidden2):
    totalinput = tf.concat([xinput, target_input], axis=1)
    hidden1 = tf.contrib.layers.fully_connected(totalinput, n_hidden1, activation_fn=actfunc)
    hidden2 = tf.contrib.layers.fully_connected(hidden1, n_hidden2, activation_fn=actfunc)
    return hidden2


def create_implicit_nn(scopename, config):
    n_input, n_hidden1, n_hidden2 = config.dim, config.n_h1, config.n_h2
    actfunc = config.actfunctype
    with tf.variable_scope(scopename):
        input = tf.placeholder(tf.float32, [None, n_input])
        with tf.GradientTape() as g:
            target_input = tf.placeholder(tf.float32, [None, 1])
            g.watch(target_input)
            hidden2 = create_implicit_concatinput_hidden(input, target_input,
                                                             act_func_dict[actfunc], n_hidden1, n_hidden2)
            fxy = tf.contrib.layers.fully_connected(hidden2, 1, activation_fn=tf.nn.tanh)
            gradwrt_target = tf.gradients(fxy, target_input)[0]

        firstloss = tf.squeeze(tf.square(fxy))
        batch_jacobian = g.batch_jacobian(fxy, target_input)
        batch_jacobian = tf.reshape(batch_jacobian, [-1])
        derivativeloss = tf.squeeze(tf.square(batch_jacobian + 1.0))
        loss = tf.reduce_mean(firstloss) + tf.reduce_mean(derivativeloss)
        lossvec = tf.squeeze(firstloss) + derivativeloss

        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
        gradplaceholders = [tf.placeholder(tf.float32, tvar.get_shape().as_list()) for tvar in tvars]
        if config.highorder_reg > 0:
            derivative_2ndorder = tf.squeeze(tf.linalg.diag_part(tf.gradients(gradwrt_target, target_input)))
            loss = loss + tf.reduce_mean(tf.square(derivative_2ndorder)) * config.highorder_reg
    return input, target_input, loss, lossvec, fxy, hidden2, tf.reduce_mean(derivativeloss), gradplaceholders, tvars

