# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:43:42 2021

@author: Declan
"""

import tensorflow as tf
import support
import numpy as np
import pandas as pd
from support import *

train = np.load("np_train_data.npy")
ucb_matrix = np.load("ucb_matrix_test.npy")
ui_matrix = np.load("ui_matrix.npy")


brand_num = 254 
class_num =  178
user_emb_dim = brand_num + class_num

D_brand_emb_dim = 128
D_class_emb_dim = 128

G_brand_emb_dim = 128
G_class_emb_dim = 128

hidden_dim = 128
alpha = 0

# Initializer
init = tf.initializers.glorot_normal()

'''Generator and Discriminator Attribute Embeddings'''
D_brand_embs = tf.keras.layers.Embedding(input_dim = brand_num, output_dim = D_brand_emb_dim,
                                          trainable=True, weights = [init(shape=(brand_num,D_brand_emb_dim))])
D_class_embs = tf.keras.layers.Embedding(input_dim = class_num, output_dim = D_class_emb_dim,
                                          trainable=True, weights = [init(shape=(class_num,D_class_emb_dim))])

G_brand_embs = tf.keras.layers.Embedding(input_dim = brand_num, output_dim = G_brand_emb_dim,
                                          trainable=True, weights = [init(shape=(brand_num,G_brand_emb_dim))])
G_class_embs = tf.keras.layers.Embedding(input_dim = class_num, output_dim = G_class_emb_dim,
                                          trainable=True, weights = [init(shape=(class_num,G_class_emb_dim))])
G_input_size =  G_brand_emb_dim + G_class_emb_dim
D_input_size = user_emb_dim + D_brand_emb_dim + D_class_emb_dim



def generator_input(brand_id, class_id):
    brand_emb = G_brand_embs(tf.constant(brand_id))
    class_emb = G_class_embs(tf.constant(class_id))
    brand_class_emb = tf.keras.layers.concatenate([brand_emb, class_emb], 1)
    return brand_class_emb

# Generates user based on concatenation of all attributes
def generator():
    bc_input = tf.keras.layers.Input(shape=(G_input_size))
    x = tf.keras.layers.Dense(hidden_dim, activation ='sigmoid')(bc_input)
    x = tf.keras.layers.Dense(hidden_dim, activation ='sigmoid')(x)
    x = tf.keras.layers.Dense(user_emb_dim, activation ='sigmoid')(x)
    g_model = tf.keras.models.Model(bc_input, x, name = 'generator')
    return g_model
g_model = generator()

# Dictionary of attribute embeddings for attribute generators
att_dict = {"brand":G_brand_embs, "class":G_class_embs}
# Generates user based on one attribute
def att_gen(att_id, att):
    att = att_dict[att]
    att_emb = tf.reshape(G_brand_embs(att_id), shape=(1,G_brand_emb_dim))
    att_input = tf.keras.layers.Input(shape=(128))
    x = tf.keras.layers.Dense(hidden_dim, activation ='sigmoid')(att_input)
    x = tf.keras.layers.Dense(hidden_dim, activation ='sigmoid')(x)
    x = tf.keras.layers.Dense(user_emb_dim, activation ='sigmoid')(x)
    model = tf.keras.models.Model(att_input, x, name = 'generator')
    return model

'''D'''
def discriminator_old(brand_id, class_id, user_emb):
    brand_emb = tf.reshape(G_brand_embs(brand_id), shape=(1,G_brand_emb_dim))
    class_emb = tf.reshape(G_class_embs(class_id), shape=(1,D_brand_emb_dim))
    user = tf.reshape(user_emb, shape = (1,user_emb_dim))
    emb = tf.concat([class_emb, brand_emb, user], 1)
    l1_outputs = tf.nn.sigmoid(tf.matmul(emb, D_W1) + D_b1)
    l2_outputs = tf.nn.sigmoid(tf.matmul(l1_outputs, D_W2) + D_b2)
    D_logit = tf.matmul(l2_outputs, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit

def discriminator_input(brand_id, class_id, user_emb):
    brand_emb = G_brand_embs(tf.constant(brand_id))
    class_emb = G_class_embs(tf.constant(class_id))
    user_emb = tf.cast(user_emb, dtype=float)
    d_input = tf.keras.layers.concatenate([brand_emb, class_emb, user_emb], 1)
    return d_input

def discriminator():
    d_input = tf.keras.layers.Input(shape=(D_input_size))
    x = tf.keras.layers.Dense(hidden_dim, activation ='sigmoid')(d_input)
    x = tf.keras.layers.Dense(hidden_dim, activation ='sigmoid')(x)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(d_input, x, name = 'discriminator')
    return model
d_model = discriminator()

'''Loss functions'''
def generator_loss(fake_user):
    return -tf.reduce_mean(fake_user)

def discriminator_loss(real, fake):
    r = tf.reduce_mean(real)
    f = tf.reduce_mean(fake)
    logit = tf.reduce_mean(fake-real)
    return logit

'''optimizer'''
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.0005)
#discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.0005)

# WGAN Class
class WGAN(tf.keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        discriminator_extra_steps=5,
        batch_size = 577
    ):
        super(WGAN, self).__init__()
        self.discriminator = d_model
        self.generator = g_model
        self.d_steps = discriminator_extra_steps
        self.batch_size = batch_size
        self.k = 10
        self.ucb_matrix = ucb_matrix
        self.ui_matrix = ui_matrix
        self.sim = get_intersection_similar_user
        self.index = 0 
        self.gp_weight = 10
    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, run_eagerly):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.run_eagerly = run_eagerly
        #self.d_loss_metric = tf.keras.metrics.Precision(name="d_loss")
        #self.g_loss_metric = tf.keras.metrics.Precision(name="g_loss")
    def precision_at_k(self, generated_users, item_id, k):
        sim_users = self.sim(generated_users, k)
        count = 0
        for i, userlist in zip(item_id, sim_users):       
            for u in userlist:
                if ui_matrix[u, i] == 1:
                    count = count + 1            
        p_k = round(count/(self.batch_size * 10), 4)
        return p_k

    '''@property
    def metrics(self):
        return [precision_at_k] '''
    
    def gradient_penalty(self, batch_size, real_users, fake_users, brand_id, class_id):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size,1], 0.0, 1.0)
        diff = fake_users - real_users
        interpolated = real_users + alpha * diff

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            interpolated_input = discriminator_input(brand_id, class_id, interpolated)
            pred = self.discriminator(interpolated_input)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def get_intersection_similar_user(self, G_user, k):
        user_emb_matrix = self.ucb_matrix
        user_emb_matrixT = np.transpose(user_emb_matrix)
        A = np.matmul(G_user, user_emb_matrixT)
        intersection_rank_matrix = np.argsort(-A)
        return intersection_rank_matrix[:, 0:k]
    
    def train_step(self, real_users):
        
        for i in range(self.d_steps):
            
            with tf.GradientTape() as tape:
                item_id, brand_id, class_id, real_users = support.get_batchdata(self.index, self.index + self.batch_size)
                # Generate fake users from attributes
                g_input0 = generator_input(brand_id, class_id)
                fake_users = self.generator(g_input0)
                # Get the logits for the fake users
                d_input0 = discriminator_input(brand_id, class_id, fake_users)
                fake_logits = self.discriminator(d_input0)
                # Get the logits for the real user
                d_input1 = discriminator_input(brand_id, class_id, real_users)
                real_logits = self.discriminator(d_input1)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_logits, fake_logits)
                # Get gradient penalty
                gp = self.gradient_penalty(self.batch_size, real_users, fake_users, brand_id, class_id)
                # Later add counter loss
                d_loss = d_cost + gp*self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as tape:
            
            # Generate fake images using the generator
            g_input1 = generator_input(brand_id, class_id)
            gen_users = self.generator(g_input1)
            # Get the discriminator logits for fake images
            d_input2 = discriminator_input(brand_id, class_id, gen_users)
            gen_logits = self.discriminator(d_input2)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        # Training precision at k
        if self.eval_steps%100 == 0:
            p_k = self.precision_at_k(gen_users, item_id, self.k)
            return {"d_loss": d_loss, "g_loss": g_loss, "p_k":p_k}
        else:
            return {"d_loss": d_loss, "g_loss": g_loss}

# Fit 
epochs = 500

# Instantiate the WGAN model.
wgan = WGAN(
    discriminator=discriminator,
    generator=generator,
    discriminator_extra_steps=3
)

# Compile the WGAN model.
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
    run_eagerly=True
)

# Start training the model.
fit = wgan.fit(train, batch_size=577, epochs=epochs, verbose=True)