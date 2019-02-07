import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np

"""
mnist configs from paper:

        self.img_dim = 28

        self.enc_dim = 256
        self.z_dim = 100
        self.dec_dim = 256

        self.read_dim = 2
        self.write_dim = 5

        self.num_timesteps = 64

"""


class DRAWCell:
    def __init__(self,
                 img_width,
                 img_height,
                 img_channels,
                 enc_dim,
                 z_dim,
                 dec_dim,
                 read_dim,
                 write_dim,
                 num_timesteps,
                 reuse=False,
                 name=None):

        assert read_dim >= 2

        self.img_dim_A = img_width
        self.img_dim_B = img_height
        self.img_channels = img_channels

        self.enc_dim = enc_dim
        self.z_dim = z_dim
        self.dec_dim = dec_dim

        self.read_dim = read_dim
        self.write_dim = write_dim

        self.num_timesteps = num_timesteps

        self.kernel_initializer = tf.glorot_normal_initializer()

        self.DO_SHARE = False
        # ^ fix recurrence bug. dynamic rnn doesnt need to worry about this, but we do.
        # see https://github.com/ericjang/draw/blob/master/draw.py

        with tf.variable_scope('DRAW'):
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, img_width, img_height, img_channels])
            self.do_inference = tf.placeholder(dtype=tf.bool, shape=[])

            self.read_op_params_kernel = tf.get_variable(dtype=tf.float32, shape=[self.dec_dim, 5], name='read_op_params_kernel', initializer=self.kernel_initializer)
            self.write_op_params_kernel = tf.get_variable(dtype=tf.float32, shape=[self.dec_dim, 5], name='write_op_params_kernel', initializer=self.kernel_initializer)
            self.write_patch_kernel = tf.get_variable(dtype=tf.float32, shape=[self.dec_dim, (self.write_dim * self.write_dim * self.img_channels)], name='write_patch_kernel', initializer=self.kernel_initializer)

            self.read_op_params_bias = tf.get_variable(dtype=tf.float32, shape=[5], name='read_op_params_bias', initializer=tf.zeros_initializer())
            self.write_op_params_bias = tf.get_variable(dtype=tf.float32, shape=[5], name='write_op_params_bias', initializer=tf.zeros_initializer())
            self.write_patch_bias = tf.get_variable(dtype=tf.float32, shape=[(self.write_dim * self.write_dim * self.img_channels)], name='write_patch_bias', initializer=tf.zeros_initializer())

            self.z_mu_kernel = tf.get_variable(dtype=tf.float32, shape=[self.dec_dim, self.z_dim], name='z_mu_kernel', initializer=self.kernel_initializer)
            self.z_logsigma_kernel = tf.get_variable(dtype=tf.float32, shape=[self.dec_dim, self.z_dim], name='z_logsigma_kernel', initializer=self.kernel_initializer)

            self.z_mu_bias = tf.get_variable(dtype=tf.float32, shape=[self.z_dim], name='z_mu_bias', initializer=tf.zeros_initializer())
            self.z_logsigma_bias = tf.get_variable(dtype=tf.float32, shape=[self.z_dim], name='z_logsigma_bias', initializer=tf.zeros_initializer())

            self.h_enc_initial = tf.get_variable(dtype=tf.float32, shape=[self.enc_dim], name='h_enc_initial', initializer=tf.zeros_initializer())
            self.h_dec_initial = tf.get_variable(dtype=tf.float32, shape=[self.dec_dim], name='h_dec_initial', initializer=tf.zeros_initializer())

            #self.canvas_initial = tf.get_variable(
            #    dtype=tf.float32, shape=[self.img_dim_B, self.img_dim_A, self.img_channels], name='canvas_initial', initializer=tf.zeros_initializer())

            self.canvas_initial = tf.get_variable(
                dtype=tf.float32, shape=[self.img_dim_B, self.img_dim_A, self.img_channels], name='canvas_initial',
                initializer=tf.constant_initializer(value=0.0),
                trainable=True)

            #self.canvas_initial = tf.zeros(dtype=tf.float32, shape=[self.img_dim_B, self.img_dim_A, self.img_channels])

            self.enc_cell = tf.contrib.rnn.LSTMCell(num_units=self.enc_dim, forget_bias=0.0, activation=tf.identity, state_is_tuple=True)
            self.dec_cell = tf.contrib.rnn.LSTMCell(num_units=self.dec_dim, forget_bias=0.0, activation=tf.identity, state_is_tuple=True)

            def _compute_attn_filters(hidden_vec, params_kernel, params_bias, grid_dim):
                five_numbers = tf.matmul(hidden_vec, params_kernel) + tf.expand_dims(params_bias, 0)
                g_tilde_X = five_numbers[:, 0]
                g_tilde_Y = five_numbers[:, 1]
                log_of_sigma_squared = five_numbers[:, 2]
                log_of_delta_tilde = five_numbers[:, 3]
                log_of_gamma = five_numbers[:, 4]

                N = grid_dim
                sigma_squared = tf.exp(log_of_sigma_squared)
                delta_tilde = tf.exp(log_of_delta_tilde)
                gamma = tf.exp(log_of_gamma)

                g_X = tf.constant((float(self.img_dim_A + 1) / 2.0)) * (g_tilde_X + 1.0)   # shape: [batch_size]
                g_Y = tf.constant((float(self.img_dim_B + 1) / 2.0)) * (g_tilde_Y + 1.0)   # shape: [batch_size]

                delta = tf.constant((float(max(self.img_dim_A, self.img_dim_B) - 1.0) / float(N - 1))) * delta_tilde

                window_ints = tf.cumsum(tf.ones(dtype=tf.int32, shape=(N))) - 1
                # a vector containing [0, 1, ..., N-1]
                # shape: [N]

                window_ints = tf.cast(window_ints, dtype=tf.float32)
                i_indexed_vec = window_ints - tf.constant((float(N) / 2.0)) - 0.5
                j_indexed_vec = window_ints - tf.constant((float(N) / 2.0)) - 0.5

                mu_X_vec = tf.expand_dims(g_X, 1) + tf.expand_dims(i_indexed_vec, 0) * tf.expand_dims(delta, 1)    # shape: [batch_size, N]
                mu_Y_vec = tf.expand_dims(g_Y, 1) + tf.expand_dims(j_indexed_vec, 0) * tf.expand_dims(delta, 1)    # shape: [batch_size, N]

                img_position_ints_A = tf.cumsum(tf.ones(dtype=tf.int32, shape=(self.img_dim_A))) - 1  # shape: [A]
                img_position_ints_B = tf.cumsum(tf.ones(dtype=tf.int32, shape=(self.img_dim_B))) - 1  # shape: [B]

                img_position_ints_A = tf.cast(img_position_ints_A, dtype=tf.float32)    # shape: [A]
                img_position_ints_B = tf.cast(img_position_ints_B, dtype=tf.float32)    # shape: [B]

                img_position_ints_A = tf.expand_dims(tf.expand_dims(img_position_ints_A, 0), 1) # shape: [1, 1, A]
                img_position_ints_B = tf.expand_dims(tf.expand_dims(img_position_ints_B, 0), 1)  # shape: [1, 1, B]

                mu_X_vec = tf.expand_dims(mu_X_vec, 2)  # shape: [batch_size, N, 1]
                mu_Y_vec = tf.expand_dims(mu_Y_vec, 2)  # shape: [batch_size, N, 1]

                # F_X and F_Y are filterbanks dynamically determined per image
                #
                # for each image, F_X is a horizontal filterbank matrix. it contains N filters of size A.
                #     the n-th of the N filters is a horizontal 1D gaussian kernel.
                #     it will be used to attend to rows of the image.
                #     the center of the n-th filter coincides with the n-th tick of the horizontal axis of the attention grid.
                #
                # for each image, F_Y is a vertical filterbank matrix. it contains N filters of size B.
                #     the n-th of the N filters is a vertical 1D gaussian kernel.
                #     it will be used to attend to columns of the image.
                #     the center of the n-th filter coincides with the n-th tick of the vertical axis of the attention grid.

                F_X_exp_arg_numerators = tf.square(img_position_ints_A - mu_X_vec)     # shape: [batch_size, N, A]
                F_Y_exp_arg_numerators = tf.square(img_position_ints_B - mu_Y_vec)     # shape: [batch_size, N, B]

                F_X_exp_arg_denominators = 2.0 * tf.expand_dims(tf.expand_dims(sigma_squared, -1), -1)   # shape: [batch_size, 1, 1]
                F_Y_exp_arg_denominators = 2.0 * tf.expand_dims(tf.expand_dims(sigma_squared, -1), -1)   # shape: [batch_size, 1, 1]

                F_X_exp_args = -(F_X_exp_arg_numerators / F_X_exp_arg_denominators)  # shape: [batch_size, N, A]
                F_Y_exp_args = -(F_Y_exp_arg_numerators / F_Y_exp_arg_denominators)  # shape: [batch_size, N, B]

                F_X_exps = tf.exp(F_X_exp_args)
                F_Y_exps = tf.exp(F_Y_exp_args)

                Z_X = tf.reduce_sum(F_X_exps, axis=2, keep_dims=True)  # shape: [batch_size, N, 1]
                Z_Y = tf.reduce_sum(F_Y_exps, axis=2, keep_dims=True)  # shape: [batch_size, N, 1]

                F_X = (F_X_exps / (Z_X + 1e8))         # shape: [batch_size, N, A]
                F_Y = (F_Y_exps / (Z_Y + 1e8))         # shape: [batch_size, N, B]

                F_X = tf.expand_dims(F_X, 3)   # shape: [batch_size, N, A, 1]
                F_Y = tf.expand_dims(F_Y, 3)   # shape: [batch_size, N, B, 1]

                return F_X, F_Y, gamma

            def _compute_read_filters(hidden_dec_tm1):
                F_X, F_Y, gamma = _compute_attn_filters(hidden_dec_tm1, self.read_op_params_kernel, self.read_op_params_bias, grid_dim=self.read_dim)
                return F_X, F_Y, gamma

            def _compute_write_filters(hidden_dec_t):
                F_hat_X, F_hat_Y, gamma_hat = _compute_attn_filters(hidden_dec_t, self.write_op_params_kernel, self.write_op_params_bias, grid_dim=self.write_dim)
                return F_hat_X, F_hat_Y, gamma_hat

            def _read_op(x_batch, F_X, F_Y):
                # for each image in the batch, the read op uses the attn matrixes FX and FY,
                # and returns an NxN patch
                # for each image

                # note that
                # draw for mnist assumes only one channel, but they apply it to datasets with color as well.
                # they apply the same convolutions independently to each channel, (see section 3.3).

                conv_1 = tf.reduce_sum(tf.expand_dims(F_Y, 3) * tf.expand_dims(x_batch, 1), axis=2)

                # shape [batch_size, N, B, 1, 1] x shape [batch_size, 1, B, A, channels]
                # get [batch_size, N, B, A, channels]
                #
                # reduce axis 2, get [batch_size, N, A, channels]

                conv_2 = tf.reduce_sum(tf.expand_dims(F_X, 1) * tf.expand_dims(conv_1, 2), axis=3)

                # shape [batch_size, 1, N, A, 1] x [batch_size, N, 1, A, channels]
                # get [batch_size, N, N, A, channels]
                #
                # reduce axis 3, get [batch_size, N, N, channels]

                return conv_2

            def _write_op(w_t, F_X_hat, F_Y_hat):
                # for each image in the batch, the read op uses the attn matrixes FX_hat and FY_hat,
                # and returns an NxN patch
                # for each image

                # note that
                # draw for mnist assumes only one channel, but they apply it to datasets with color as well.
                # they apply the same convolutions independently to each channel, (see section 3.3).

                conv_1 = tf.reduce_sum(tf.expand_dims(F_Y_hat, 2) * tf.expand_dims(w_t, 3), axis=1)

                # shape [batch_size, N, 1, B, 1] x shape [batch_size, N, N, 1, channels]
                # get [batch_size, N, N, B, channels]
                #
                # reduce axis 2, get [batch_size, N, B, channels]

                conv_2 = tf.reduce_sum(tf.expand_dims(F_X_hat, 2) * tf.expand_dims(conv_1, 3), axis=1)

                # shape [batch_size, N, 1, A, 1] x [batch_size, N, B, 1, channels]
                # get [batch_size, N, B, A, channels]
                #
                # reduce axis 1, get [batch_size, B, A, channels]

                return conv_2

            batch_size = tf.shape(self.x)[0]

            self.epsilons = tf.random_normal(shape=[batch_size, self.num_timesteps, self.z_dim])

            drawings_over_time = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.num_timesteps,
                                                              dynamic_size=False, infer_shape=True)

            def _recurrence(t, canvas_tm1, encoder_state_tm1, decoder_state_tm1, kl_div_0_thru_tm1, drawings_over_time, reuse=self.DO_SHARE):
                with tf.variable_scope('step', reuse=reuse):
                    x_hat_t = self.x - tf.nn.sigmoid(canvas_tm1)
                    #x_hat_t = self.x - canvas_tm1

                    h_dec_tm1 = decoder_state_tm1.h

                    # compute params for read attn
                    F_X, F_Y, gamma = _compute_read_filters(h_dec_tm1)

                    # read from the input image and the error image according using the (read) attention
                    read_x = _read_op(self.x, F_X, F_Y)
                    read_x_hat_t = _read_op(x_hat_t, F_X, F_Y)
                    r_t = tf.reshape(gamma, [-1, 1, 1, 1]) * tf.concat([read_x, read_x_hat_t], axis=3)
                    r_t = tf.reshape(r_t, [-1, (2 * self.read_dim * self.read_dim * self.img_channels)])

                    # update the encoder cell
                    encoder_inputs = tf.concat([r_t, h_dec_tm1], axis=1)
                    h_enc_t, encoder_state_t = self.enc_cell(inputs=encoder_inputs, state=encoder_state_tm1)

                    # inference: compute posterior for z distribution; our recognition model assumes it's a diagonal gaussian.
                    z_t_mu = tf.matmul(h_enc_t, self.z_mu_kernel) + tf.expand_dims(self.z_mu_bias, 0)
                    z_t_log_sigma = tf.matmul(h_enc_t, self.z_logsigma_kernel) + tf.expand_dims(self.z_logsigma_bias, 0)

                    epsilons_t = self.epsilons[:, t, :]
                    z_prior_sample_t = epsilons_t
                    z_posterior_sample_t = z_t_mu + tf.exp(z_t_log_sigma) * epsilons_t

                    # sample the z's for this timestep from the appropriate distribution
                    # our prior at every timestep is an isotropic gaussian.
                    z_t = tf.cond(self.do_inference, true_fn=lambda: z_posterior_sample_t, false_fn=lambda: z_prior_sample_t)

                    # update the decoder cell
                    h_dec_t, decoder_state_t = self.dec_cell(inputs=z_t, state=decoder_state_tm1)

                    # compute what to write to the canvas, these are basically brushstrokes without a specified location or scale
                    w_t = tf.matmul(h_dec_t, self.write_patch_kernel) #+ tf.expand_dims(self.write_patch_bias, 0)
                    w_t = tf.reshape(w_t, [-1, self.write_dim, self.write_dim, self.img_channels])

                    # compute params for write attn
                    F_X_hat, F_Y_hat, gamma_hat = _compute_write_filters(h_dec_t)

                    gamma_hat_inverse = (1.0 / gamma_hat)

                    # write the computed brushstrokes to the canvas in the manner determine by the (write) attention
                    write_w_t = tf.reshape(gamma_hat_inverse, [-1, 1, 1, 1]) * _write_op(w_t, F_X_hat, F_Y_hat)

                    canvas_t = canvas_tm1 + write_w_t
                    #canvas_t = canvas_tm1 + (1 - canvas_tm1) * tf.nn.sigmoid(write_w_t)

                    canvas_t.set_shape([None, self.img_dim_B, self.img_dim_A, self.img_channels])

                    # compute the kl divergence from the prior for the latent variables at this timestep
                    # the kl div is between our inferred diagonal gaussian and the isotropic gaussian prior.
                    #
                    # the notation in Gregor et al. absent mindedly used t as the index of summation, but this is misleading, because the final term is a constant in their formula and is written as -T/2.
                    # they also use T as the total number of timesteps.
                    # in fact, the correct formula for the kl divergence has a -1/2 for every element in z_dim, and thus the constant is significantly larger (by a factor of z_dim).
                    # had they been summing over the z dim index like they were supposed to, they would not have made this mistake.
                    #
                    # So, that's a longwinded way of saying that my formula below is correct, even though it does not match the paper.
                    # I am relying instead on Kingma and Welling, who invented VAEs,
                    # and whose paper includes a derivation of the appropriate kl divergence formula.
                    #
                    # Reference: appendix B of https://arxiv.org/pdf/1312.6114.pdf
                    #
                    z_t_kl_div = tf.reduce_sum(0.5 * (tf.square(z_t_mu) + tf.square(tf.exp(z_t_log_sigma)) - 2.0 * z_t_log_sigma - 1.0), axis=1)
                    # shape: [batch_size]

                    kl_div_0_thru_t = kl_div_0_thru_tm1 + z_t_kl_div

                    drawing = tf.nn.sigmoid(canvas_t)
                    #drawing = canvas_t
                    drawings_over_time_updated = drawings_over_time.write(t, drawing)

                self.DO_SHARE = True

                return t+1, canvas_t, encoder_state_t, decoder_state_t, kl_div_0_thru_t, drawings_over_time_updated

        canvas_initial_state = tf.tile(
            tf.expand_dims(self.canvas_initial, 0),
            multiples=[batch_size, 1, 1, 1])

        encoder_initial_state = tf.contrib.rnn.LSTMStateTuple(
            c=tf.tile(tf.expand_dims(self.h_enc_initial, 0), multiples=[batch_size, 1]),
            h=tf.tile(tf.expand_dims(self.h_enc_initial, 0), multiples=[batch_size, 1])
        )
        #encoder_initial_state = self.enc_cell.zero_state(dtype=tf.float32, batch_size=batch_size)

        decoder_initial_state = tf.contrib.rnn.LSTMStateTuple(
            c = tf.tile(tf.expand_dims(self.h_dec_initial, 0), multiples=[batch_size, 1]),
            h = tf.tile(tf.expand_dims(self.h_dec_initial, 0), multiples=[batch_size, 1])
        )
        #decoder_initial_state = self.dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size)

        _, self.canvas_Tm1, _, _, self.kl_div_0_thru_Tm1, self.drawings_over_time = tf.while_loop(
              cond=lambda t, *_: t < self.num_timesteps,
              body=_recurrence,
              loop_vars=(tf.constant(0, dtype=tf.int32),
                         canvas_initial_state,
                         encoder_initial_state,
                         decoder_initial_state,
                         tf.zeros(dtype=tf.float32, shape=[batch_size]),
                         drawings_over_time),
              shape_invariants=(
                  tf.TensorShape([]),
                  tf.TensorShape([None, self.img_dim_B, self.img_dim_A, self.img_channels]),
                  tf.contrib.rnn.LSTMStateTuple(c=tf.TensorShape([None, self.enc_dim]), h=tf.TensorShape([None, self.enc_dim])),
                  tf.contrib.rnn.LSTMStateTuple(c=tf.TensorShape([None, self.dec_dim]), h=tf.TensorShape([None, self.dec_dim])),
                  tf.TensorShape([None]),
                  tf.TensorShape(None)
              )
        )

        self.drawings_over_time = self.drawings_over_time.stack()
        self.drawings_over_time = tf.transpose(self.drawings_over_time, perm=[1, 0, 2, 3, 4])

        self.D_X_given_canvas_Tm1 = tf.nn.sigmoid(self.canvas_Tm1)

        cross_entropy_terms = -((self.x) * tf.log(self.D_X_given_canvas_Tm1 + 1e-8) + (1.0 - self.x) * (tf.log(1.0 - self.D_X_given_canvas_Tm1 + 1e-8)))
        cross_entropy_per_image = tf.reduce_sum(cross_entropy_terms, axis=[1,2,3])
        self.elbo = tf.reduce_mean(-cross_entropy_per_image - self.kl_div_0_thru_Tm1, axis=0)
        self.loss = -self.elbo

        self.optimizer = tf.train.AdamOptimizer(1e-3)

        tvars = tf.trainable_variables()
        gradients, _ = zip(*self.optimizer.compute_gradients(loss=self.loss, var_list=tvars))
        gradients, _ = tf.clip_by_global_norm(gradients, 100.0)
        self.train_op = self.optimizer.apply_gradients(zip(gradients, tvars))

    def train(self, sess, imgs):
        feed_dict = {
            self.x: imgs,
            self.do_inference: True
        }
        _, elbo = sess.run([self.train_op, self.elbo], feed_dict=feed_dict)
        return elbo

    def reconstruct(self, sess, imgs):
        feed_dict = {
            self.x: imgs,
            self.do_inference: True
        }
        drawings_over_time = sess.run(self.drawings_over_time, feed_dict=feed_dict)
        return drawings_over_time

    def sample(self, sess, num_imgs):
        # use some empty_images for self.x placeholder
        # this is simply so we dont have to write a different computation graph for sampling.
        # the priors are what is sampled from, and the input images and the encoder rnn have no impact on the output.
        #
        empty_images = np.zeros(dtype=np.float32, shape=[num_imgs, self.img_dim_B, self.img_dim_A, self.img_channels])
        feed_dict = {
            self.x: empty_images,
            self.do_inference: False
        }
        drawings_over_time = sess.run(self.drawings_over_time, feed_dict=feed_dict)
        return drawings_over_time
