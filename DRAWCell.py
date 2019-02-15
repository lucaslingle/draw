import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np


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

        self.kernel_initializer = tf.random_uniform_initializer(-0.1, 0.1)

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.img_dim_B, self.img_dim_A, self.img_channels])
        self.do_inference = tf.placeholder(dtype=tf.bool, shape=[])

        self.read_op_params_kernel = tf.get_variable(dtype=tf.float32, shape=[self.enc_dim, 5], name='read_op_params_kernel', initializer=self.kernel_initializer)
        self.write_op_params_kernel = tf.get_variable(dtype=tf.float32, shape=[self.dec_dim, 5], name='write_op_params_kernel', initializer=self.kernel_initializer)

        self.write_patch_kernel = tf.get_variable(dtype=tf.float32, shape=[self.dec_dim, (self.write_dim * self.write_dim * self.img_channels)], name='write_patch_kernel', initializer=self.kernel_initializer)

        self.read_op_params_bias = tf.get_variable(dtype=tf.float32, shape=[5], name='read_op_params_bias', initializer=tf.zeros_initializer())
        self.write_op_params_bias = tf.get_variable(dtype=tf.float32, shape=[5], name='write_op_params_bias', initializer=tf.zeros_initializer())

        self.write_patch_bias = tf.get_variable(dtype=tf.float32, shape=[(self.write_dim * self.write_dim * self.img_channels)], name='write_patch_bias', initializer=tf.zeros_initializer())

        self.z_mu_kernel = tf.get_variable(dtype=tf.float32, shape=[self.enc_dim, self.z_dim], name='z_mu_kernel', initializer=self.kernel_initializer)
        self.z_logsigma_kernel = tf.get_variable(dtype=tf.float32, shape=[self.enc_dim, self.z_dim], name='z_logsigma_kernel', initializer=self.kernel_initializer)

        self.z_mu_bias = tf.get_variable(dtype=tf.float32, shape=[self.z_dim], name='z_mu_bias', initializer=tf.zeros_initializer())
        self.z_logsigma_bias = tf.get_variable(dtype=tf.float32, shape=[self.z_dim], name='z_logsigma_bias', initializer=tf.zeros_initializer())

        self.canvas_initial = tf.get_variable(
            dtype=tf.float32, shape=[self.img_dim_B, self.img_dim_A, self.img_channels], name='canvas_initial',
            initializer=tf.constant_initializer(value=0.0),
            trainable=True)

        batch_size = tf.shape(self.x)[0]
        self.epsilons = tf.random_normal(shape=[batch_size, self.num_timesteps, self.z_dim])

        self.enc_rnn_h_kernel = tf.get_variable(dtype=tf.float32, shape=[self.enc_dim, 4 * self.enc_dim], name='enc_rnn_h_kernel')
        self.enc_rnn_x_kernel = tf.get_variable(dtype=tf.float32, shape=[(2 * self.read_dim * self.read_dim * self.img_channels + self.dec_dim), 4 * self.enc_dim], name='enc_rnn_x_kernel')

        self.dec_rnn_h_kernel = tf.get_variable(dtype=tf.float32, shape=[self.dec_dim, 4 * self.dec_dim], name='dec_rnn_h_kernel')
        self.dec_rnn_x_kernel = tf.get_variable(dtype=tf.float32, shape=[self.z_dim, 4 * self.dec_dim], name='dec_rnn_x_kernel')

        self.enc_rnn_bias = tf.get_variable(dtype=tf.float32, shape=[4 * self.enc_dim], name='enc_rnn_bias')
        self.dec_rnn_bias = tf.get_variable(dtype=tf.float32, shape=[4 * self.dec_dim], name='dec_rnn_bias')

        self.encoder_initial = tf.get_variable(dtype=tf.float32, shape=[2 * self.enc_dim], name='enc_initial_state',
                                               initializer=tf.constant_initializer(0.0), trainable=True)

        self.decoder_initial = tf.get_variable(dtype=tf.float32, shape=[2 * self.dec_dim], name='dec_initial_state',
                                               initializer=tf.constant_initializer(0.0), trainable=True)

        drawings_over_time = tensor_array_ops.TensorArray(dtype=tf.float32, size=(self.num_timesteps+1),
                                                          dynamic_size=False, infer_shape=True)

        canvas_initial_state = tf.tile(
            tf.expand_dims(self.canvas_initial, 0),
            multiples=[batch_size, 1, 1, 1])

        drawings_over_time = drawings_over_time.write(0, tf.nn.sigmoid(canvas_initial_state))

        encoder_initial_state = tf.tile(
            tf.expand_dims(self.encoder_initial, 0),
            multiples=[batch_size, 1])

        decoder_initial_state = tf.tile(
            tf.expand_dims(self.decoder_initial, 0),
            multiples=[batch_size, 1])

        canvases_over_time = {}
        encoder_states = {}
        decoder_states = {}
        kl_divs_thus_far = {}

        canvases_over_time[0] = canvas_initial_state
        encoder_states[0] = encoder_initial_state
        decoder_states[0] = decoder_initial_state
        kl_divs_thus_far[0] = tf.zeros(dtype=tf.float32, shape=[batch_size])

        for t in range(1, self.num_timesteps+1):
            tp1, canvases_over_time[t], encoder_states[t], decoder_states[t], kl_divs_thus_far[t], drawings_over_time = self._recurrence(
                t, canvases_over_time[t-1], encoder_states[t-1], decoder_states[t-1], kl_divs_thus_far[t-1], drawings_over_time)

        self.canvas_T = canvases_over_time[self.num_timesteps]

        self.drawings_over_time = drawings_over_time.stack()
        self.drawings_over_time = tf.transpose(self.drawings_over_time, perm=[1, 0, 2, 3, 4])

        self.D_X_given_canvas_T = tf.nn.sigmoid(self.canvas_T)

        self.kl_div_1_thru_T = kl_divs_thus_far[self.num_timesteps]

        # reconstruction loss is based on binary cross entropy:
        cross_entropy_terms = -(
            (self.x) * tf.log(self.D_X_given_canvas_T + 1e-8) + \
            (1.0 - self.x) * (tf.log(1.0 - self.D_X_given_canvas_T + 1e-8))
        )

        cross_entropy_per_image = tf.reduce_sum(cross_entropy_terms, axis=[1,2,3])
        self.elbo = tf.reduce_mean((-cross_entropy_per_image - self.kl_div_1_thru_T), axis=0)
        self.loss = -self.elbo

        self.optimizer = tf.train.AdamOptimizer(1e-3)

        tvars = tf.trainable_variables()
        gradients, _ = zip(*self.optimizer.compute_gradients(loss=self.loss, var_list=tvars))
        gradients = [None if g is None else tf.clip_by_norm(g, 5.0) for g in gradients]
        self.train_op = self.optimizer.apply_gradients(zip(gradients, tvars))

        self.gradnorm = tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in gradients if g is not None])

    def _compute_five_numbers(self, hidden_vec, params_kernel, params_bias):
        five_numbers = tf.matmul(hidden_vec, params_kernel) + tf.expand_dims(params_bias, 0)
        g_tilde_X = five_numbers[:, 0]
        g_tilde_Y = five_numbers[:, 1]
        log_of_sigma_squared = five_numbers[:, 2]
        log_of_delta_tilde = five_numbers[:, 3]
        log_of_gamma = five_numbers[:, 4]

        sigma_squared = tf.exp(log_of_sigma_squared)
        delta_tilde = tf.exp(log_of_delta_tilde)
        gamma = tf.exp(log_of_gamma)

        return g_tilde_X, g_tilde_Y, sigma_squared, delta_tilde, gamma

    def _compute_attn_filters(self, five, N):
        g_tilde_X, g_tilde_Y, sigma_squared, delta_tilde, gamma = five

        g_X = tf.constant((float(self.img_dim_A + 1) / 2.0)) * (g_tilde_X + 1.0)  # shape: [batch_size]
        g_Y = tf.constant((float(self.img_dim_B + 1) / 2.0)) * (g_tilde_Y + 1.0)  # shape: [batch_size]

        delta = tf.constant((float(max(self.img_dim_A, self.img_dim_B) - 1.0) / float(N - 1))) * delta_tilde

        window_ints = tf.cumsum(tf.ones(dtype=tf.int32, shape=(N)))
        # a vector containing [1, 2, ..., N]
        # shape: [N]

        window_ints = tf.cast(window_ints, dtype=tf.float32)
        i_indexed_vec = window_ints - tf.constant((float(N) / 2.0)) - 0.5  # [-(N/2-0.5), ..., (N/2-0.5)]
        j_indexed_vec = window_ints - tf.constant((float(N) / 2.0)) - 0.5  # [-(N/2-0.5), ..., (N/2-0.5)]

        mu_X_vec = tf.expand_dims(g_X, 1) + tf.expand_dims(i_indexed_vec, 0) * tf.expand_dims(delta, 1)  # shape: [batch_size, N]
        mu_Y_vec = tf.expand_dims(g_Y, 1) + tf.expand_dims(j_indexed_vec, 0) * tf.expand_dims(delta, 1)  # shape: [batch_size, N]

        img_position_ints_A = tf.cumsum(tf.ones(dtype=tf.int32, shape=(self.img_dim_A))) - 1  # shape: [A]
        img_position_ints_B = tf.cumsum(tf.ones(dtype=tf.int32, shape=(self.img_dim_B))) - 1  # shape: [B]

        img_position_ints_A = tf.cast(img_position_ints_A, dtype=tf.float32)  # shape: [A]
        img_position_ints_B = tf.cast(img_position_ints_B, dtype=tf.float32)  # shape: [B]

        img_position_ints_A = tf.expand_dims(tf.expand_dims(img_position_ints_A, 0), 1)  # shape: [1, 1, A]
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

        F_X_exp_arg_numerators = tf.square(img_position_ints_A - mu_X_vec)  # shape: [batch_size, N, A]
        F_Y_exp_arg_numerators = tf.square(img_position_ints_B - mu_Y_vec)  # shape: [batch_size, N, B]

        F_X_exp_arg_denominators = 2.0 * tf.expand_dims(tf.expand_dims(sigma_squared, -1), -1)  # shape: [batch_size, 1, 1]
        F_Y_exp_arg_denominators = 2.0 * tf.expand_dims(tf.expand_dims(sigma_squared, -1), -1)  # shape: [batch_size, 1, 1]

        F_X_exp_args = -(F_X_exp_arg_numerators / F_X_exp_arg_denominators)  # shape: [batch_size, N, A]
        F_Y_exp_args = -(F_Y_exp_arg_numerators / F_Y_exp_arg_denominators)  # shape: [batch_size, N, B]

        F_X_exps = tf.exp(F_X_exp_args)
        F_Y_exps = tf.exp(F_Y_exp_args)

        Z_X = tf.maximum(1e-8, tf.reduce_sum(F_X_exps, axis=2, keep_dims=True))  # shape: [batch_size, N, 1]
        Z_Y = tf.maximum(1e-8, tf.reduce_sum(F_Y_exps, axis=2, keep_dims=True))  # shape: [batch_size, N, 1]

        F_X = (F_X_exps / Z_X)  # shape: [batch_size, N, A]
        F_Y = (F_Y_exps / Z_Y)  # shape: [batch_size, N, B]

        return F_X, F_Y, gamma

    def _compute_read_filters(self, h_dec_tm1):
        five = self._compute_five_numbers(h_dec_tm1, self.read_op_params_kernel, self.read_op_params_bias)
        F_X, F_Y, gamma = self._compute_attn_filters(five, N=self.read_dim)
        return F_X, F_Y, gamma

    def _compute_write_filters(self, h_dec_t):
        five = self._compute_five_numbers(h_dec_t, self.write_op_params_kernel, self.write_op_params_bias)
        F_hat_X, F_hat_Y, gamma_hat = self._compute_attn_filters(five, N=self.write_dim)
        return F_hat_X, F_hat_Y, gamma_hat

    def _read_op(self, x_batch, F_X, F_Y):
        x_batch = tf.transpose(x_batch, [0, 3, 1, 2])  # [Batch, channels, B, A]
        F_X = tf.expand_dims(F_X, 1)  # [Batch, 1, N, A]
        F_Y = tf.expand_dims(F_Y, 1)  # [Batch, 1, N, B]

        F_X = tf.tile(F_X, multiples=[1, self.img_channels, 1, 1])    # [Batch, channels, N, A]
        F_Y = tf.tile(F_Y, multiples=[1, self.img_channels, 1, 1])    # [Batch, channels, N, B]

        F_X_T = tf.transpose(F_X, perm=[0, 1, 3, 2])  # [Batch, channels, A, N]

        read_x_t = tf.matmul(
            F_Y,  # [Batch, channels, N, B]
            tf.matmul(x_batch, F_X_T)  # [Batch, channels, B, A] x [Batch, channels, A, N] = [Batch, channels, B, N]
        )
        # [Batch, channels, N, B] x [Batch, channels, B, N] = [Batch, channels, N, N]

        read_x_t = tf.transpose(read_x_t, [0, 2, 3, 1])  # [Batch, N, N, channels]

        return read_x_t

    def _write_op(self, w_t, F_X_hat, F_Y_hat):
        w_t = tf.transpose(w_t, [0, 3, 1, 2])  # [Batch, channels, N, N]
        F_X_hat = tf.expand_dims(F_X_hat, 1)  # [Batch, 1, N, A]
        F_Y_hat = tf.expand_dims(F_Y_hat, 1)  # [Batch, 1, N, B]

        F_X_hat = tf.tile(F_X_hat, multiples=[1, self.img_channels, 1, 1])
        F_Y_hat = tf.tile(F_Y_hat, multiples=[1, self.img_channels, 1, 1])

        F_Y_hat_T = tf.transpose(F_Y_hat, perm=[0, 1, 3, 2])  # [Batch, channels, B, N]

        write_w_t = tf.matmul(
            F_Y_hat_T,  # [Batch, channels, B, N]
            tf.matmul(w_t, F_X_hat)  # [Batch, channels, N, N] x [Batch, channels, N, A] = [Batch, channels, N, A]
        )
        # [Batch, channels, B, N] x [Batch, channels, N, A] = [Batch, channels, B, A]

        write_w_t = tf.transpose(write_w_t, [0, 2, 3, 1])  # [Batch, B, A, channels]

        return write_w_t

    def _recurrence(self, t, canvas_tm1, enc_state_tm1, dec_state_tm1, kl_div_0_thru_tm1, drawings_over_time):
        x_hat_t = self.x - tf.nn.sigmoid(canvas_tm1)

        h_enc_tm1 = enc_state_tm1[:, 0:self.enc_dim]
        c_enc_tm1 = enc_state_tm1[:, -self.enc_dim:]
        h_dec_tm1 = dec_state_tm1[:, 0:self.dec_dim]
        c_dec_tm1 = dec_state_tm1[:, -self.dec_dim:]

        # compute params for read attn
        F_X, F_Y, gamma = self._compute_read_filters(h_dec_tm1)

        # read from the input image and the error image according using the (read) attention
        read_x = self._read_op(self.x, F_X, F_Y)
        read_x_hat_t = self._read_op(x_hat_t, F_X, F_Y)
        r_t = tf.reshape(gamma, [-1, 1, 1, 1]) * tf.concat([read_x, read_x_hat_t], axis=3)
        r_t = tf.reshape(r_t, [-1, (2 * self.read_dim * self.read_dim * self.img_channels)])

        # update the encoder cell
        encoder_inputs = tf.concat([r_t, h_dec_tm1], axis=1)
        enc_gates = tf.matmul(encoder_inputs, self.enc_rnn_x_kernel) + tf.matmul(h_enc_tm1, self.enc_rnn_h_kernel) + tf.expand_dims(self.enc_rnn_bias, 0)
        f, i, o, j = tf.split(enc_gates, 4, axis=1)
        f = tf.nn.sigmoid(f + 1.0)
        i = tf.nn.sigmoid(i)
        o = tf.nn.sigmoid(o)
        j = tf.nn.tanh(j)
        c_enc_t = f * c_enc_tm1 + i * j
        h_enc_t = o * c_enc_t
        enc_state_t = tf.concat([h_enc_t, c_enc_t], axis=1)

        # inference: compute posterior for z distribution; our recognition model assumes it's a diagonal gaussian.
        z_t_mu = tf.matmul(h_enc_t, self.z_mu_kernel) + tf.expand_dims(self.z_mu_bias, 0)
        z_t_log_sigma = tf.matmul(h_enc_t, self.z_logsigma_kernel) + tf.expand_dims(self.z_logsigma_bias, 0)

        epsilons_t = self.epsilons[:, (t-1), :]   # recurrence time index t starts at 1, gotta subtract to get appropriate index.

        z_prior_sample_t = epsilons_t
        z_posterior_sample_t = z_t_mu + tf.exp(z_t_log_sigma) * epsilons_t

        z_t = tf.cond(self.do_inference, true_fn=lambda: z_posterior_sample_t, false_fn=lambda: z_prior_sample_t)

        # update the decoder cell
        dec_gates = tf.matmul(z_t, self.dec_rnn_x_kernel) + tf.matmul(h_dec_tm1, self.dec_rnn_h_kernel) + tf.expand_dims(self.dec_rnn_bias, 0)
        f, i, o, j = tf.split(dec_gates, 4, axis=1)
        f = tf.nn.sigmoid(f + 1.0)
        i = tf.nn.sigmoid(i)
        o = tf.nn.sigmoid(o)
        j = tf.nn.tanh(j)
        c_dec_t = f * c_dec_tm1 + i * j
        h_dec_t = o * c_dec_t
        dec_state_t = tf.concat([h_dec_t, c_dec_t], axis=1)

        # compute what to write to the canvas, these are basically brushstrokes without a specified location or scale
        w_t = tf.matmul(h_dec_t, self.write_patch_kernel) + tf.expand_dims(self.write_patch_bias, 0)
        w_t = tf.reshape(w_t, [-1, self.write_dim, self.write_dim, self.img_channels])

        # compute params for write attn
        F_X_hat, F_Y_hat, gamma_hat = self._compute_write_filters(h_dec_t)

        gamma_hat_inverse = (1.0 / gamma_hat)

        # write the computed brushstrokes to the canvas in the manner determine by the (write) attention
        write_w_t = tf.reshape(gamma_hat_inverse, [-1, 1, 1, 1]) * self._write_op(w_t, F_X_hat, F_Y_hat)

        canvas_t = canvas_tm1 + write_w_t

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
        z_t_kl_div = tf.reduce_sum(
            0.5 * (tf.square(z_t_mu) + tf.square(tf.exp(z_t_log_sigma)) - 2.0 * z_t_log_sigma - 1.0), axis=1)
        # shape: [batch_size]

        kl_div_0_thru_t = kl_div_0_thru_tm1 + z_t_kl_div

        drawing = tf.nn.sigmoid(canvas_t)
        drawings_over_time = drawings_over_time.write(t, drawing)

        return t + 1, canvas_t, enc_state_t, dec_state_t, kl_div_0_thru_t, drawings_over_time

    def train(self, sess, imgs):
        # train the DRAW model for one gradient step
        feed_dict = {
            self.x: imgs,
            self.do_inference: True
        }
        _, elbo, gradnorm = sess.run([self.train_op, self.elbo, self.gradnorm], feed_dict=feed_dict)
        return elbo, gradnorm

    def reconstruct(self, sess, imgs):
        # use DRAW model to generate reconstructions of given images

        feed_dict = {
            self.x: imgs,
            self.do_inference: True
        }
        drawings_over_time = sess.run(self.drawings_over_time, feed_dict=feed_dict)
        return drawings_over_time

    def sample(self, sess, num_imgs):
        # use DRAW model to generate some new images by sampling from the prior

        # when running this, the generation process does not depend on any input image.
        # to enforce this, we use a boolean placeholder 'do_inference', and set it to False.
        # this causes the model to sample epsilons directly rather than sample mu + sigma * epsilons.
        #
        # we also pass in some empty_images for self.x, the placeholder for the input images.
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